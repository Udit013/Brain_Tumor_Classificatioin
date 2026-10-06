# Model Card — NeuroClass (EfficientNetB3 Brain Tumor MRI Classifier)

**Live demo:** https://huggingface.co/spaces/Udit013/brain-tumor-mri-classifier
· **Model:** https://huggingface.co/Udit013/brain-tumor-efficientnetb3
· **Code:** https://github.com/Udit013/Brain_Tumor_Classificatioin

## Model details
- **Architecture:** EfficientNetB3 (ImageNet-pretrained, fully trainable) with
  `pooling='max'` → BatchNorm → Dense(512, L2/L1-regularised) → Dropout(0.4) →
  Dense(256, regularised) → Dropout(0.2) → Dense(4, softmax). 11.7M params.
- **Input:** 256×256×3 RGB, **raw [0,255] pixels** (EfficientNet applies its own
  rescaling internally — do not divide by 255).
- **Output:** softmax over `[glioma, meningioma, notumor, pituitary]`, then a
  fitted temperature (T = 0.561) for calibrated confidence.
- **Training:** Adamax(1e-3), categorical cross-entropy, 20 epochs,
  EarlyStopping(loss), ReduceLROnPlateau(val_loss), best-`val_accuracy`
  checkpoint, followed by BatchNorm running-statistics recalibration.
- **Serving formats:** Keras `.h5` weights (Grad-CAM) and ONNX (inference). The
  ONNX export matches CPU Keras to within 2.4e-5 per output probability.
- **Origin:** reproduces the best model from "Identifying Various Types of Brain
  Tumors using Deep Neural Network based Image Features," 2024 International
  Conference on Cognitive Robotics and Intelligent Systems (ICC-ROBINS), IEEE,
  2024. DOI: 10.1109/ICC-ROBINS60238.2024.10533941. The original notebooks are
  preserved byte-for-byte under [`/legacy`](legacy/).

## Intended use
- **Intended:** research, education, and ML-engineering demonstration of an
  end-to-end evaluated and served classification pipeline.
- **Out of scope:** **NOT a medical device.** Must not be used for diagnosis,
  triage, or any clinical decision-making. No regulatory clearance.

## Training data
- Msoud Nickparvar **"Brain Tumor MRI Dataset"** (Kaggle), a compilation of
  figshare CE-MRI + SARTAJ + Br35H. The paper used 7,023 images (5,712 train /
  1,311 test); the dataset as currently distributed has 7,200 (5,600 / 1,600).
- **Preprocessing:** resize 256×256, RGB; horizontal-flip augmentation (train).

## Evaluation data & metrics
- **In-distribution:** the full 1,600-image Kaggle test split.
- **Out-of-distribution:** a separate public dataset (Navoneel Chakrabarty
  binary tumor/no-tumor set, n=253), scored by collapsing the three tumor
  classes into one.
- **Device:** all metrics are measured on **CPU** — the same computation the
  deployed ONNX model performs. See "Metal vs CPU" below.
- Every number here is produced by `scripts/reproduce.sh`; none is transcribed
  by hand.

## Quantitative results (measured on CPU)
| Metric | Value |
|---|---|
| Accuracy (full 1,600-image test set) | 91.25% |
| Macro ROC-AUC / average precision | 0.980 / 0.961 |
| Accuracy: test images with a near-duplicate in training vs novel images | 97.2% vs 86.5% |
| Near-duplicate / exact-duplicate test images | 714 (44.6%) / 114 (7.1%) |
| External out-of-distribution accuracy (n=253) | 70.8% |
| ECE before → after temperature scaling | 0.196 → 0.073 (held-out: 0.074 ± 0.008) |
| Clean vs mean corrupted accuracy (6 corruptions × 5 severities) | 90.5% vs 68.3% |
| TTA predictive entropy, correct vs wrong predictions | 0.82 vs 1.00 |
| Single-image latency, Keras-CPU / ONNX-CPU / ONNX-CoreML | 79 / 34 / 18.6 ms |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 0.996 | **0.677** | 0.807 |
| meningioma | 0.792 | 0.990 | 0.880 |
| notumor | 0.926 | 1.000 | 0.962 |
| pituitary | 0.992 | 0.983 | 0.987 |

The paper's published 99.844% is not directly comparable: it was scored on
640 of 1,311 test images, from an older version of the dataset (see below).

## Limitations & ethical considerations
- **Missed tumors (glioma).** Glioma recall is 67.7%: of 400 glioma test images,
  97 are predicted as meningioma and **32 (8%) as `notumor`**. A missed tumor is
  the most consequential error for any screening use.
- **Train/test leakage:** the published split is image-level over a multi-source
  compilation of 3-D volumes, so near-identical slices from one scan can appear
  on both sides. Patient/scan IDs are **not recoverable** from the Kaggle
  release, so the leakage module reports an image-similarity **lower bound**.
  The model is 10.7 points less accurate on novel images than on leaked ones.
- **Test set used for model selection:** the best checkpoint is chosen by
  `val_accuracy` computed on (part of) the test set, both in the paper and in
  this parity reproduction. Test metrics are therefore optimistically biased.
- **Published metrics on partial test sets:** each paper notebook evaluates
  `1311 // 32 = 40` batches. With batch size 16, EfficientNetB3's 99.844% is
  exactly 639/640 — 49% of the test set, unshuffled. The other three models
  were scored on 1,280 images, so the comparison used different denominators.
- **Metal vs CPU:** on Apple Silicon, `tensorflow-metal` computes different
  outputs for these weights (93.94% vs 91.25% on CPU, disagreeing on 2.9% of
  test images). CPU TensorFlow and ONNX Runtime agree with each other, so
  inference is pinned to CPU and Metal is used for training only.
- **Train/serve preprocessing skew:** metrics are measured with Keras
  nearest-neighbour resizing (the published pipeline), but serving resizes with
  PIL's bicubic default; under the serving preprocessing test accuracy is
  **90.25%**, a point below the reported 91.25%. Averaging the Space's 6
  test-time-augmentation views raises that to 92.31% (glioma recall 64.3% →
  71.5%) — measured, not yet adopted.
- **Distribution shift:** out-of-distribution accuracy (70.8%) is far below the
  in-distribution number — the honest generalisation signal.
- **Robustness:** near chance (~25%) under additive Gaussian noise at every
  severity tested; degrades sharply under heavy blur and JPEG compression;
  stays above 70% under brightness, contrast, and rotation changes.
- **Uncertainty:** TTA entropy is higher on wrong predictions, but the
  correct/wrong distributions overlap heavily, so entropy alone is a weak
  abstain signal.
- **Bias:** demographic, scanner, and acquisition-protocol distributions of the
  compiled dataset are undocumented; performance across subgroups is unknown.

## How to reproduce
```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh      # needs Kaggle credentials
bash scripts/reproduce.sh
python scripts/fill_readme_results.py
```
