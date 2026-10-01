# Diabetic Retinopathy Detection

Attention-based Multiple Instance Learning (MIL) for automated 5-class diabetic retinopathy grading on retinal fundus images, trained on the combined APTOS 2019 and Messidor-2 datasets.

## Results

Best model (eval split, n=1,080 images):

| Metric | Value |
|---|---|
| Quadratic Weighted Kappa (QWK) | 0.731 |
| Accuracy | 0.615 |
| Referable DR sensitivity (grade ≥ 2) | ~0.84 |
| PDR sensitivity (grade 4) | 0.606 |
| PDR AUC | 0.873 |
| Severe under-grades (grade 4 → 0/1/2) | 14 / 66 |

Per-class performance:

| Grade | Severity | Recall | Precision | F1 |
|---|---|---|---|---|
| 0 | No DR | 0.782 | 0.942 | 0.855 |
| 1 | Mild NPDR | 0.578 | 0.308 | 0.402 |
| 2 | Moderate NPDR | 0.328 | 0.722 | 0.433 |
| 3 | Severe NPDR | 0.537 | 0.299 | 0.384 |
| 4 | Proliferative DR | 0.606 | 0.231 | 0.327 |

The model prioritizes PDR detection (the clinically critical class) over raw QWK. A model with higher QWK but zero PDR sensitivity would be clinically useless for screening.

## Datasets

| Dataset | Images | Labels | Notes |
|---|---|---|---|
| APTOS 2019 | 3,662 | 0–4 ICDR | Kaggle; mixed resolutions (381×381 to 2848×3614) |
| Messidor-2 | 1,744 | 0–4 ICDR | ADCIS images + Abràmoff et al. adjudication CSV |

Combined class distribution is severely imbalanced — grade 0 is 52% of data, grades 3+4 are only 11%. Split: stratified 80/20, performed per dataset. Final: 4,317 training bags, 1,080 eval bags.

## Architecture
Retinal image → crop retina + circular mask
→ green channel + Gaussian blur (kernel=5)
→ slice into 224×224 non-overlapping patches
→ bag of up to 64 patches
→ DenseNet-121 (ImageNet pretrained) → 1024-D per patch
→ gated attention pooling → bag representation
→ linear classifier → 5-class logits


## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Merge datasets | `unify.py` | `unified_index.csv` with train/eval split |
| 2. Crop retina | `crop_images.py` | Cropped images + circular mask |
| 3. Build proxy sample | `make_proxy_sample.py` | 400-image class-balanced sample |
| 4. Tune preprocessing | `optuna_tune.py` | Winning `green + blur=5` |
| 5. Apply preprocessing | `preprocess.py` | Green channel + blur at native resolution |
| 6. Slice into bags | `slice_bags.py` | 224×224 patches + bag index |
| 7. Train model | `train_mil.py` | `best.pt` + `best_metrics.json` |

Optuna result: green channel (mean val_loss 1.362) beat rgb (1.389), lab (1.514), hsv (1.550). Blur=5 won the best trial.

## Training Evolution

The model was iterated four times, each fixing a specific failure:

| Iteration | Approach | Outcome | Root Cause |
|---|---|---|---|
| 1 | Ordinal CDW-CE loss | QWK −0.117 | Stacked class weights × distance penalty → model avoided extreme classes |
| 2 | Plain CE, unfrozen from start | Train 95% / Eval 18% | No augmentation, no regularization → overfitting |
| 3 | Overfitting fixes (augmentation, frozen backbone, BN stats frozen) | QWK 0.776, PDR sens 0.0 | Class weights alone can't overcome 8.6× frequency gap |
| 4a | Cost-sensitive loss with cost matrix | QWK 0.779, PDR sens 0.0 | Cost term satisfiable without learning grade-4 features |
| 4b | WeightedRandomSampler | QWK 0.731, PDR sens 0.606 | Works — model actually sees grade 4 often enough to learn it |

Key insight: loss re-weighting cannot fix extreme class imbalance, but sampling can. Model selection uses `QWK + 0.5 × PDR_sensitivity` so a model that detects PDR beats one with higher QWK but zero grade-4 recall.

## Usage

py unify.py              # Build unified index
py crop_images.py        # Crop retinal circles
py make_proxy_sample.py  # Build Optuna sample
py optuna_tune.py        # Hyperparameter search (~55 min)
py preprocess.py         # Apply winning preprocessing
py slice_bags.py         # Slice into patches
py train_mil.py          # Train model (~2.3 hours)
