"""
optuna_tune.py

Optuna search over preprocessing parameters on whole images.

Search space:
- color_space:  rgb | green | hsv | lab
                hsv and lab use full 3-channel conversions (not single-channel
                extraction). green extracts the G channel and replicates it to
                3 channels for ImageNet-pretrained backbone compatibility.
- blur_kernel:  0 (off) | 3 | 5 | 7 | 9 | 11  (Gaussian, sigma auto-derived)

Model: frozen DenseNet-121 backbone + linear head.
Whole-image input at 512x512 (resized from cropped native size).

Device is hard-required to be CUDA. Script exits with a clear error if
CUDA is unavailable so you don't silently fall back to CPU.

Progress: prints trial start, per-epoch train/val metrics, prune events,
and a running best. A summary prints at the end including mean val_loss
per color_space and per blur_kernel for interpretability.
"""

import sys
import time
import optuna
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split


CONFIG = {
    "proxy_csv":    "datasets/proxy/proxy_sample.csv",
    "n_trials":     30,
    "epochs":       5,
    "batch_size":   8,
    "lr":           1e-3,
    "image_size":   512,
    "device":       "cuda",
    "seed":         42,
    "output_csv":   "datasets/proxy/optuna_results.csv",
    "num_workers":  0,
}


# -----------------------------
# Device check
# -----------------------------
def setup_device(requested):
    print("=" * 60)
    print("Device check")
    print("=" * 60)
    print(f"torch version:        {torch.__version__}")
    print(f"CUDA available:       {torch.cuda.is_available()}")
    print(f"CUDA version (torch): {torch.version.cuda}")

    if requested == "cuda":
        if not torch.cuda.is_available():
            print("\n[ERROR] CUDA requested but not available.")
            print("Reinstall PyTorch with CUDA support:")
            print("  pip uninstall torch torchvision -y")
            print("  pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu126")
            sys.exit(1)

        n = torch.cuda.device_count()
        print(f"GPU count:            {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name}  "
                  f"{props.total_memory / 1024**3:.1f} GB  "
                  f"compute capability {props.major}.{props.minor}")

        device = torch.device("cuda:0")
        try:
            x = torch.zeros(1, device=device)
            del x
            torch.cuda.synchronize()
            print(f"\n[check] CUDA smoke test passed on {device}")
        except Exception as e:
            print(f"\n[ERROR] CUDA smoke test failed: {e}")
            sys.exit(1)
    else:
        device = torch.device("cpu")
        print("[warn] running on CPU — this will be slow")

    print(f"\nUsing device: {device}")
    return device


# -----------------------------
# Preprocessing
# -----------------------------
def apply_color_space(img_bgr, mode):
    """
    Return a 3-channel uint8 image.

    - rgb:   standard RGB conversion
    - green: extract G channel, replicate to 3 channels
    - hsv:   full HSV conversion (3 channels)
    - lab:   full LAB conversion (3 channels)

    Note: HSV and LAB are full conversions, not single-channel extractions.
    These produce channel distributions that differ from ImageNet RGB stats,
    which may hurt pretrained feature quality. If RGB wins by a wide margin,
    that's likely why — see report for discussion.
    """
    if mode == "rgb":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if mode == "green":
        g = img_bgr[:, :, 1]
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    if mode == "hsv":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if mode == "lab":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    raise ValueError(f"Unknown color_space: {mode}")


def apply_blur(img, kernel_size):
    k = int(kernel_size)
    if k <= 1:
        return img
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)


def preprocess_image(img_bgr, params):
    img = apply_color_space(img_bgr, params["color_space"])
    img = apply_blur(img, params["blur_kernel"])
    return img


# -----------------------------
# Dataset
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, df, params, image_size, transform):
        self.df = df.reset_index(drop=True)
        self.params = params
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = preprocess_image(img, self.params)
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA)
        return self.transform(img), int(row["label"])


def make_transform(image_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# -----------------------------
# Model
# -----------------------------
class WholeImageClassifier(nn.Module):
    def __init__(self, feature_dim=1024, n_classes=5):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        return self.head(feats)


# -----------------------------
# Train / eval
# -----------------------------
def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad()
        logits = model(imgs)
        loss = ce(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def objective(trial, df_train, df_val, device):
    assert device.type == "cuda", \
        f"Trial {trial.number} running on {device.type} — expected cuda"

    params = {
        "color_space": trial.suggest_categorical(
            "color_space", ["rgb", "green", "hsv", "lab"]),
        "blur_kernel": trial.suggest_categorical(
            "blur_kernel", [0, 3, 5, 7, 9, 11]),
    }

    print(f"\n{'=' * 60}")
    print(f"[Trial {trial.number}] color_space={params['color_space']}  "
          f"blur_kernel={params['blur_kernel']}")
    print(f"{'=' * 60}")

    transform = make_transform(CONFIG["image_size"])
    ds_train = ImageDataset(df_train, params, CONFIG["image_size"], transform)
    ds_val   = ImageDataset(df_val,   params, CONFIG["image_size"], transform)

    dl_train = DataLoader(ds_train, batch_size=CONFIG["batch_size"],
                          shuffle=True, num_workers=CONFIG["num_workers"],
                          pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=CONFIG["batch_size"],
                          shuffle=False, num_workers=CONFIG["num_workers"],
                          pin_memory=True)

    model = WholeImageClassifier().to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=CONFIG["lr"])

    best_val_loss = float("inf")
    t0 = time.time()
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = run_epoch(model, dl_train, optimizer, device,
                                          train=True)
        val_loss, val_acc = run_epoch(model, dl_val, optimizer, device,
                                      train=False)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1}/{CONFIG['epochs']}  "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}  "
              f"[{elapsed:.1f}s]")
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"  -> pruned at epoch {epoch + 1}")
            raise optuna.TrialPruned()

    print(f"  -> best_val_loss={best_val_loss:.4f}  "
          f"(total {time.time() - t0:.1f}s)")
    return best_val_loss


# -----------------------------
# Main
# -----------------------------
def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])

    proxy_path = Path(CONFIG["proxy_csv"])
    if not proxy_path.exists():
        sys.exit(f"Missing proxy CSV: {proxy_path}")

    df = pd.read_csv(proxy_path)
    print(f"Proxy sample: {len(df)} images")

    df_train, df_val = train_test_split(
        df, test_size=0.25, stratify=df["label"], random_state=CONFIG["seed"]
    )
    print(f"Train: {len(df_train)}  Val: {len(df_val)}")

    device = setup_device(CONFIG["device"])

    print("\n" + "=" * 60)
    print(f"Starting Optuna: {CONFIG['n_trials']} trials, "
          f"{CONFIG['epochs']} epochs each")
    print(f"Search space: 4 color spaces x 6 blur kernels = 24 combos")
    print("=" * 60)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=CONFIG["seed"])
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="minimize",
                                sampler=sampler, pruner=pruner)

    start = time.time()
    study.optimize(lambda t: objective(t, df_train, df_val, device),
                   n_trials=CONFIG["n_trials"])
    total_time = time.time() - start

    print("\n" + "=" * 60)
    print("Search complete")
    print("=" * 60)
    print(f"Total time:   {total_time / 60:.1f} min "
          f"({total_time / max(1, CONFIG['n_trials']):.1f}s per trial avg)")
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best value:  {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value)
    print("\nTop 10 trials:")
    print(f"  {'rank':<5} {'trial':<7} {'val_loss':<10} "
          f"{'color_space':<14} {'blur_kernel':<12}")
    for rank, t in enumerate(completed[:10], 1):
        print(f"  {rank:<5} {t.number:<7} {t.value:<10.4f} "
              f"{t.params['color_space']:<14} {t.params['blur_kernel']:<12}")

    if len(completed) >= 5:
        print("\nMean val_loss by color_space:")
        by_cs = {}
        for t in completed:
            by_cs.setdefault(t.params["color_space"], []).append(t.value)
        for cs, vals in sorted(by_cs.items(),
                               key=lambda kv: np.mean(kv[1])):
            print(f"  {cs:<10}  n={len(vals):<3}  "
                  f"mean={np.mean(vals):.4f}  best={min(vals):.4f}")

        print("\nMean val_loss by blur_kernel:")
        by_bk = {}
        for t in completed:
            by_bk.setdefault(t.params["blur_kernel"], []).append(t.value)
        for bk, vals in sorted(by_bk.items(),
                               key=lambda kv: np.mean(kv[1])):
            print(f"  blur={bk:<3}  n={len(vals):<3}  "
                  f"mean={np.mean(vals):.4f}  best={min(vals):.4f}")

    rows = []
    for t in study.trials:
        r = {"number": t.number, "state": t.state.name, "value": t.value}
        r.update(t.params)
        rows.append(r)
    out = Path(CONFIG["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved all trials to {out}")


if __name__ == "__main__":
    main()