"""
train_mil.py

Attention-based Multiple Instance Learning (MIL) for diabetic retinopathy
grading with WeightedRandomSampler to force the model to see rare classes.

Key change from cost-sensitive version:
  - Replaced CostSensitiveLoss with plain weighted CE
  - Added WeightedRandomSampler to oversample grades 3 and 4 during training
  - Sampling weights are the inverse of class frequency raised to a power
  - Model selection criterion unchanged: QWK + 0.5 * PDR sensitivity

Reads:  datasets/bags/bag_index.csv
Writes: runs/mil_densenet121_sampler/best.pt
        runs/mil_densenet121_sampler/best_metrics.json

Device: hard-required CUDA.
"""

import sys
import json
import time
import math
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    roc_auc_score,
)
from tqdm import tqdm


CONFIG = {
    "bag_index":        "datasets/bags/bag_index.csv",
    "run_dir":          "runs/mil_densenet121_sampler",
    "device":           "cuda",

    "bag_size_cap":     64,
    "bag_size_floor":   3,
    "patch_batch":      16,

    "epochs":                   6,
    "freeze_backbone_epochs":   2,
    "bags_per_step":            8,
    "lr_backbone":              1e-5,
    "lr_head":                  1e-3,
    "weight_decay":             1e-4,
    "warmup_epochs":            1,
    "grad_clip":                1.0,
    "label_smoothing":          0.05,
    "freeze_bn_stats":          True,

    # Milder class weights since sampler handles the imbalance.
    # These still apply in the loss to give a small extra push.
    "class_weights":    [1.0, 1.5, 1.2, 2.0, 2.5],

    # Sampler strength: sample_prob ∝ (1 / count)^sampler_power
    # 0.0 = uniform (no reweighting)
    # 1.0 = fully inverse frequency (rare classes sampled as often as common ones)
    # Typical sweet spot: 0.5-1.0
    "sampler_power":    1.0,

    "seed":             42,
    "num_workers":      0,
    "log_every_n_bags": 50,
}


def setup_device(requested):
    print("=" * 60)
    print("Device check")
    print("=" * 60)
    print(f"torch version:  {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        sys.exit("CUDA required but not available. Reinstall PyTorch with CUDA.")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:            {props.name}  "
          f"{props.total_memory / 1024**3:.1f} GB  "
          f"cc {props.major}.{props.minor}")
    device = torch.device("cuda:0")
    x = torch.zeros(1, device=device); del x; torch.cuda.synchronize()
    print(f"[check] CUDA smoke test passed on {device}")
    return device


# =============================
# Transforms
# =============================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=(90, 90)),
    ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =============================
# Dataset
# =============================
class BagDataset(Dataset):
    def __init__(self, bag_index_df, split, cap, floor, transform):
        df = bag_index_df[bag_index_df["split"] == split].copy()
        groups = df.groupby("parent_image_id")
        self.bags = []
        for image_id, grp in groups:
            if len(grp) < floor:
                continue
            self.bags.append({
                "image_id": image_id,
                "label":    int(grp["label"].iloc[0]),
                "dataset":  grp["dataset"].iloc[0],
                "paths":    grp["patch_path"].tolist(),
            })
        self.cap = cap
        self.floor = floor
        self.transform = transform
        print(f"[{split}] {len(self.bags)} bags after floor={floor}")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        paths = bag["paths"]
        if len(paths) > self.cap:
            paths = random.sample(paths, self.cap)

        tensors = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            tensors.append(self.transform(img))
        if not tensors:
            tensors = [torch.zeros(3, 224, 224)]

        return {
            "patches":  torch.stack(tensors, dim=0),
            "label":    bag["label"],
            "image_id": bag["image_id"],
            "dataset":  bag["dataset"],
        }


def collate_bags(batch):
    return batch


def build_weighted_sampler(dataset, sampler_power, seed):
    """
    WeightedRandomSampler with per-sample weights = (1 / class_count)^power.

    sampler_power = 0.0 -> uniform sampling (no effect)
    sampler_power = 1.0 -> inverse-frequency sampling (rarest class sampled
                           as often as the most common)
    Values in between interpolate smoothly.
    """
    labels = [b["label"] for b in dataset.bags]
    counts = Counter(labels)

    # Per-sample weight: (1 / count_of_its_class)^power
    class_weight = {c: (1.0 / counts[c]) ** sampler_power for c in counts}
    sample_weights = [class_weight[l] for l in labels]

    total = sum(sample_weights)
    sample_weights = [w / total for w in sample_weights]

    print(f"\nSampler config (power={sampler_power}):")
    print(f"  Class counts: {dict(sorted(counts.items()))}")
    print(f"  Class weights: "
          f"{ {c: round(class_weight[c], 6) for c in sorted(class_weight)} }")

    # Expected number of samples per class per epoch
    print(f"  Expected samples/class in one epoch of {len(labels)} draws:")
    expected = {c: round(sample_weights[i] * len(labels) * counts[c]
                         / sum(1 for l in labels if l == c), 1)
                for i, c in enumerate(labels[:1])}  # dummy
    # Cleaner: recompute
    expected = {}
    for c in sorted(counts):
        idxs = [i for i, l in enumerate(labels) if l == c]
        prob_mass = sum(sample_weights[i] for i in idxs)
        expected[c] = round(prob_mass * len(labels), 1)
    print(f"    {expected}")

    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )


# =============================
# Model
# =============================
class AttentionMIL(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )
        self.backbone.classifier = nn.Identity()
        feature_dim = 1024

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.att_V = nn.Linear(feature_dim, 128)
        self.att_U = nn.Linear(feature_dim, 128)
        self.att_w = nn.Linear(128, 1)
        self.classifier = nn.Linear(feature_dim, n_classes)

    def set_backbone_trainable(self, trainable):
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def freeze_bn_stats(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, patches, patch_batch_size):
        N = patches.shape[0]
        feats = []
        for i in range(0, N, patch_batch_size):
            chunk = patches[i:i + patch_batch_size]
            feats.append(self.backbone(chunk))
        feats = torch.cat(feats, dim=0)

        a_V = torch.tanh(self.att_V(feats))
        a_U = torch.sigmoid(self.att_U(feats))
        a = self.att_w(a_V * a_U).squeeze(-1)
        a = F.softmax(a, dim=0)

        bag_repr = (feats * a.unsqueeze(-1)).sum(dim=0)
        logits = self.classifier(bag_repr)
        return logits, a


# =============================
# Scheduler
# =============================
def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# =============================
# Metrics
# =============================
def compute_metrics(all_labels, all_preds, all_probs, n_classes=5):
    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    mae = mean_absolute_error(labels, preds)

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    per_class_recall, per_class_precision, per_class_f1 = {}, {}, {}
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class_recall[c] = float(recall)
        per_class_precision[c] = float(precision)
        per_class_f1[c] = float(f1)

    ref_true = (labels >= 2).astype(int)
    ref_pred = (preds >= 2).astype(int)
    ref_cm = confusion_matrix(ref_true, ref_pred, labels=[0, 1])
    ref_tn, ref_fp, ref_fn, ref_tp = ref_cm.ravel()
    ref_sens = ref_tp / (ref_tp + ref_fn) if (ref_tp + ref_fn) > 0 else 0.0
    ref_spec = ref_tn / (ref_tn + ref_fp) if (ref_tn + ref_fp) > 0 else 0.0
    ref_auc = (roc_auc_score(ref_true, probs[:, 2:].sum(axis=1))
               if len(np.unique(ref_true)) > 1 else 0.0)

    pdr_true = (labels == 4).astype(int)
    pdr_pred = (preds == 4).astype(int)
    pdr_cm = confusion_matrix(pdr_true, pdr_pred, labels=[0, 1])
    pdr_tn, pdr_fp, pdr_fn, pdr_tp = pdr_cm.ravel()
    pdr_sens = pdr_tp / (pdr_tp + pdr_fn) if (pdr_tp + pdr_fn) > 0 else 0.0
    pdr_spec = pdr_tn / (pdr_tn + pdr_fp) if (pdr_tn + pdr_fp) > 0 else 0.0
    pdr_auc = (roc_auc_score(pdr_true, probs[:, 4])
               if len(np.unique(pdr_true)) > 1 else 0.0)

    severe_under = int(((labels == 4) & (preds <= 2)).sum())

    return {
        "accuracy": float((labels == preds).mean()),
        "qwk": float(qwk),
        "mae": float(mae),
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "referable_dr": {
            "sensitivity": float(ref_sens),
            "specificity": float(ref_spec),
            "auc": float(ref_auc),
        },
        "pdr": {
            "sensitivity": float(pdr_sens),
            "specificity": float(pdr_spec),
            "auc": float(pdr_auc),
            "true_positives": int(pdr_tp),
            "true_negatives": int(pdr_tn),
            "false_positives": int(pdr_fp),
            "false_negatives": int(pdr_fn),
        },
        "severe_undergrades": severe_under,
    }


# =============================
# Train / eval
# =============================
def run_train_epoch(model, loader, optimizer, scheduler, scaler, loss_fn,
                    device, config, epoch):
    model.train()
    if config["freeze_bn_stats"]:
        model.freeze_bn_stats()

    total_loss = 0.0
    total_bags = 0
    correct = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Train epoch {epoch}")
    for batch in pbar:
        for bag in batch:
            patches = bag["patches"].to(device, non_blocking=True)
            label = torch.tensor([bag["label"]], dtype=torch.long, device=device)

            with autocast(dtype=torch.float16):
                logits, _ = model(patches, config["patch_batch"])
                loss = loss_fn(logits.unsqueeze(0), label)
                loss = loss / config["bags_per_step"]

            scaler.scale(loss).backward()

            total_loss += loss.item() * config["bags_per_step"]
            total_bags += 1
            correct += int(logits.argmax().item() == bag["label"])

            if total_bags % config["bags_per_step"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if total_bags % config["log_every_n_bags"] == 0:
                avg_loss = total_loss / total_bags
                acc = correct / total_bags
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{acc:.3f}",
                    "vram": f"{gpu_mem:.1f}G",
                })

    return total_loss / max(1, total_bags), correct / max(1, total_bags)


@torch.no_grad()
def run_eval_epoch(model, loader, device, config):
    model.eval()
    all_labels, all_preds, all_image_ids, all_probs = [], [], [], []

    pbar = tqdm(loader, desc="Eval")
    for batch in pbar:
        for bag in batch:
            patches = bag["patches"].to(device, non_blocking=True)
            with autocast(dtype=torch.float16):
                logits, _ = model(patches, config["patch_batch"])
            probs = F.softmax(logits, dim=0).float().cpu().numpy()
            all_labels.append(bag["label"])
            all_preds.append(int(probs.argmax()))
            all_probs.append(probs.tolist())
            all_image_ids.append(bag["image_id"])
        pbar.set_postfix({"done": len(all_labels)})

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["image_ids"] = all_image_ids
    metrics["probs"] = all_probs
    return metrics


# =============================
# Main
# =============================
def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    run_dir = Path(CONFIG["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(CONFIG["device"])

    bag_path = Path(CONFIG["bag_index"])
    if not bag_path.exists():
        sys.exit(f"Missing bag index: {bag_path}")
    df = pd.read_csv(bag_path)
    print(f"Loaded {len(df)} patch rows from {bag_path}")

    train_ds = BagDataset(df, split="train",
                          cap=CONFIG["bag_size_cap"],
                          floor=CONFIG["bag_size_floor"],
                          transform=TRAIN_TRANSFORM)
    eval_ds  = BagDataset(df, split="eval",
                          cap=CONFIG["bag_size_cap"],
                          floor=CONFIG["bag_size_floor"],
                          transform=EVAL_TRANSFORM)

    if len(train_ds) == 0 or len(eval_ds) == 0:
        sys.exit("Empty split after filtering.")

    train_labels = [b["label"] for b in train_ds.bags]
    train_dist = (pd.Series(train_labels).value_counts().sort_index().to_dict())
    print(f"\nTrain label distribution: {train_dist}")

    # ---- WeightedRandomSampler for training ----
    sampler = build_weighted_sampler(
        train_ds,
        sampler_power=CONFIG["sampler_power"],
        seed=CONFIG["seed"],
    )

    train_loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                              num_workers=CONFIG["num_workers"],
                              collate_fn=collate_bags)
    eval_loader  = DataLoader(eval_ds, batch_size=1, shuffle=False,
                              num_workers=CONFIG["num_workers"],
                              collate_fn=collate_bags)

    print("\nBuilding model...")
    model = AttentionMIL(n_classes=5).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (backbone frozen): {n_trainable:,}")

    class_weights = torch.tensor(CONFIG["class_weights"],
                                 dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=CONFIG["label_smoothing"],
    )
    print(f"\nLoss: weighted CE + label_smoothing={CONFIG['label_smoothing']}")
    print(f"Class weights (in loss): {CONFIG['class_weights']}")

    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.att_V.parameters())
                   + list(model.att_U.parameters())
                   + list(model.att_w.parameters())
                   + list(model.classifier.parameters()))

    optimizer = AdamW([
        {"params": backbone_params, "lr": CONFIG["lr_backbone"]},
        {"params": head_params,     "lr": CONFIG["lr_head"]},
    ], weight_decay=CONFIG["weight_decay"])

    steps_per_epoch = max(1, len(train_ds) // CONFIG["bags_per_step"])
    total_steps = steps_per_epoch * CONFIG["epochs"]
    warmup_steps = steps_per_epoch * CONFIG["warmup_epochs"]
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler()

    epoch_log = []
    best_score = -1.0
    best_epoch_metrics = None
    best_epoch_num = -1
    t0 = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        if epoch == CONFIG["freeze_backbone_epochs"] + 1:
            print(f"\n>>> Unfreezing backbone at epoch {epoch}")
            model.set_backbone_trainable(True)
            n_trainable = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
            print(f"    Trainable params now: {n_trainable:,}")

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{CONFIG['epochs']}  "
              f"elapsed_total={(time.time() - t0) / 60:.1f}min")
        print(f"{'=' * 60}")

        epoch_t0 = time.time()
        train_loss, train_acc = run_train_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn,
            device, CONFIG, epoch
        )
        train_time = time.time() - epoch_t0
        print(f"Train: loss={train_loss:.4f}  acc={train_acc:.3f}  "
              f"[{train_time / 60:.1f}min]")

        eval_t0 = time.time()
        metrics = run_eval_epoch(model, eval_loader, device, CONFIG)
        eval_time = time.time() - eval_t0
        print(f"Eval:  acc={metrics['accuracy']:.3f}  "
              f"qwk={metrics['qwk']:.4f}  [{eval_time / 60:.1f}min]")
        print("Per-class recall:",
              {k: round(v, 3) for k, v in metrics["per_class_recall"].items()})
        print("Per-class precision:",
              {k: round(v, 3) for k, v in metrics["per_class_precision"].items()})
        print(f"PDR sens: {metrics['pdr']['sensitivity']:.3f}  "
              f"PDR AUC: {metrics['pdr']['auc']:.3f}  "
              f"severe under-grades: {metrics['severe_undergrades']}")
        print("Confusion matrix:")
        for row in metrics["confusion_matrix"]:
            print(f"  {row}")

        epoch_time = train_time + eval_time
        epoch_log.append({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "train_acc":    train_acc,
            "eval_acc":     metrics["accuracy"],
            "eval_qwk":     metrics["qwk"],
            "eval_mae":     metrics["mae"],
            "pdr_sens":     metrics["pdr"]["sensitivity"],
            "pdr_auc":      metrics["pdr"]["auc"],
            "ref_sens":     metrics["referable_dr"]["sensitivity"],
            "severe_under": metrics["severe_undergrades"],
            "train_time_s": train_time,
            "eval_time_s":  eval_time,
        })

        # Selection: prefer models that detect PDR, not just maximize QWK.
        selection_score = metrics["qwk"] + 0.5 * metrics["pdr"]["sensitivity"]
        if selection_score > best_score:
            best_score = selection_score
            best_epoch_num = epoch
            best_epoch_metrics = {k: v for k, v in metrics.items()
                                  if k not in ("probs", "image_ids")}
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": CONFIG,
            }, run_dir / "best.pt")
            print(f"  -> saved new best "
                  f"(score={best_score:.4f}, "
                  f"QWK={metrics['qwk']:.4f}, "
                  f"PDR_sens={metrics['pdr']['sensitivity']:.3f})")

        avg_epoch_time = (time.time() - t0) / epoch
        remaining = avg_epoch_time * (CONFIG["epochs"] - epoch)
        print(f"  Epoch time: {epoch_time / 60:.1f}min  |  "
              f"Est. remaining: {remaining / 60:.1f}min")

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print("Training complete")
    print(f"{'=' * 60}")
    print(f"Total time:   {total_time / 60:.1f} min")
    print(f"Best epoch:   {best_epoch_num}  "
          f"(selection score={best_score:.4f})")
    print(f"Run directory: {run_dir}")

    # =============================
    # Final: reload best checkpoint, run eval, dump metrics JSON
    # =============================
    print("\n" + "=" * 60)
    print(f"Loading best checkpoint (epoch {best_epoch_num}) and evaluating")
    print("=" * 60)

    best_ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    final = run_eval_epoch(model, eval_loader, device, CONFIG)

    true_labels = [bag["label"] for bag in eval_ds.bags]
    probs = np.array(final["probs"])
    preds = probs.argmax(axis=1).tolist()

    report_str = classification_report(
        true_labels, preds,
        labels=[0, 1, 2, 3, 4],
        target_names=["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"],
        digits=4,
        zero_division=0,
    )
    print(report_str)

    out = {
        "run_info": {
            "run_dir":          str(run_dir),
            "total_time_min":   round(total_time / 60, 2),
            "best_epoch":       best_epoch_num,
            "selection_score":  round(best_score, 4),
            "selection_rule":   "QWK + 0.5 * PDR_sensitivity",
            "train_bags":       len(train_ds),
            "eval_bags":        len(eval_ds),
            "train_distribution": train_dist,
            "sampler_power":    CONFIG["sampler_power"],
        },
        "config": CONFIG,
        "epoch_history": epoch_log,
        "best_epoch_metrics": best_epoch_metrics,
        "final_eval_on_best": {
            k: v for k, v in final.items()
            if k not in ("probs", "image_ids")
        },
        "classification_report": report_str,
    }

    metrics_path = run_dir / "best_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved:")
    print(f"  {run_dir / 'best.pt'}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()