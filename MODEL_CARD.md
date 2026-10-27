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
  fitted temperature for calibrated confidence (deployed: T = 0.561, fit before
  evaluation moved to the serving preprocessing; re-fit on it: T = 0.577).
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
  Training resizes with Keras nearest-neighbour (as published); evaluation and
  serving use the same PIL resize (`btc/preprocess.py`).

## Evaluation data & metrics
- **In-distribution:** the full 1,600-image Kaggle test split.
- **Out-of-distribution:** a separate public dataset (Navoneel Chakrabarty
  binary tumor/no-tumor set, n=253), scored by collapsing the three tumor
  classes into one.
- **Device and preprocessing:** all metrics are measured on **CPU** with the
  serving preprocessing — the same computation the deployed ONNX model performs.
  See "Metal vs CPU" below.
- Every number here is produced by `scripts/reproduce.sh`; none is transcribed
  by hand.

## Quantitative results (measured on CPU, serving preprocessing)
| Metric | Value |
|---|---|
| Accuracy (full 1,600-image test set) | 90.25% |
| Macro ROC-AUC / average precision | 0.981 / 0.963 |
| Accuracy: test images with a near-duplicate in training vs novel images | 95.9% vs 85.7% |
| Near-duplicate / exact-duplicate test images | 714 (44.6%) / 114 (7.1%) |
| External out-of-distribution accuracy (n=253) | 69.6% (tumor recall 50.3%, no-tumor specificity 100%) |
| ECE before → after temperature scaling | 0.188 → 0.068 (fit on half of test, scored on the other half: 0.194 → 0.065) |
| Clean vs mean corrupted accuracy (6 corruptions × 5 severities) | 90.25% vs 67.3% |
| TTA predictive entropy, correct vs wrong predictions | 0.81 vs 1.02 |
| Single-image latency, Keras-CPU / ONNX-CPU / ONNX-CoreML | 79 / 34 / 18.6 ms |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 1.000 | **0.642** | 0.782 |
| meningioma | 0.771 | 0.985 | 0.865 |
| notumor | 0.924 | 1.000 | 0.960 |
| pituitary | 0.985 | 0.983 | 0.984 |

Until October 2026 evaluation resized images differently from serving, and
these numbers were reported as 91.25% accuracy, 97.2% vs 86.5%, OOD 70.8%,
ECE 0.196 → 0.073 and glioma recall 67.7%; the values above replace them.

The paper's published 99.844% is not directly comparable: it was scored on
640 of 1,311 test images, from an older version of the dataset (see below).

## Limitations & ethical considerations
- **Missed tumors (glioma).** Glioma recall is 64.2%: of 400 glioma test images,
  110 are predicted as meningioma and **33 (8.2%) as `notumor`**. A missed tumor
  is the most consequential error for any screening use. On the external set
  the model finds only half the tumors (50.3%).
- **Train/test leakage:** the published split is image-level over a multi-source
  compilation of 3-D volumes, so near-identical slices from one scan can appear
  on both sides. Patient/scan IDs are **not recoverable** from the Kaggle
  release, so the leakage module reports an image-similarity **lower bound**.
  The model is 10.3 points less accurate on novel images than on leaked ones.
- **Test set used for model selection:** the best checkpoint is chosen by
  `val_accuracy` computed on (part of) the test set, both in the paper and in
  this parity reproduction. Test metrics are therefore optimistically biased.
- **Published metrics on partial test sets:** each paper notebook evaluates
  `1311 // 32 = 40` batches. With batch size 16, EfficientNetB3's 99.844% is
  exactly 639/640 — 49% of the test set, unshuffled. The other three models
  were scored on 1,280 images, so the comparison used different denominators.
- **Metal vs CPU:** on Apple Silicon, `tensorflow-metal`'s *graph* mode (used
  by Keras `predict` and `fit`) computes a different function: 93.81% vs 90.25%
  on CPU for these weights, 3.6% of predictions flipped; eager mode matches CPU.
  Disabling grappler's meta-optimizer removes the forward-pass difference. The
  model was trained through the defective path. CPU TensorFlow and ONNX Runtime
  agree, so inference is pinned to CPU.
- **Train/serve preprocessing skew:** the model was trained with Keras
  nearest-neighbour resizing but is served (and now evaluated) with PIL's
  bicubic resize; serving costs it about a point of accuracy (91.25% under the
  training-style resize). Averaging the Space's 6 test-time-augmentation views
  helps this model (external OOD 69.6% → 71.9%) but is not used for its
  predictions; the Space uses those views only for its stability indicator.
- **Distribution shift:** out-of-distribution accuracy (69.6%) is far below the
  in-distribution number — the honest generalisation signal.
- **Robustness:** near chance (~25%) under additive Gaussian noise at every
  severity tested; degrades sharply under heavy blur and JPEG compression;
  stays above 70% under brightness, contrast, and rotation changes.
- **Uncertainty:** TTA entropy is higher on wrong predictions, but the
  correct/wrong distributions overlap heavily, so entropy alone is a weak
  abstain signal.
- **Bias:** demographic, scanner, and acquisition-protocol distributions of the
  compiled dataset are undocumented; performance across subgroups is unknown.

## Candidate replacement: NeuroClass v2 (staged, not deployed)
Same architecture and weight layout; retrained with `btc/train_v2.py` (head on
cached features, then fine-tuning from `block6a` with augmentation including
Gaussian noise; BatchNorm frozen; CPU, serving preprocessing; 21.5 min on a
laptop CPU). Selected on a leakage-safe validation split of Kaggle Training
(841 images; near-duplicate clusters never straddle the split) and compared
with the deployed model once on the test set:

| Metric (CPU, serving preprocessing) | Deployed | v2 |
|---|---|---|
| Test accuracy | 90.25% | **93.44%** |
| Novel-image subset (n=886) | 85.67% | **89.39%** |
| Glioma recall / gliomas → `notumor` | 64.25% / 33 | **79.50% / 25** |
| Macro F1 | 0.898 | **0.933** |
| ECE (raw) | 0.188 | **0.022** |
| Mean accuracy, 30 corruptions | 67.3% | **90.7%** |
| External OOD accuracy (n=253) | 69.6% | **72.7%** |
| External tumor recall / no-tumor specificity | 50.3% / 100% | 56.1% / 99.0% |

Accuracy gains are paired-significant (exact McNemar: full test p = 1.4e-6,
novel subset p = 2.6e-4, glioma p = 7.8e-13). v2 trained on fewer images
(4,759 vs 5,600) and, unlike the deployed model, never saw the test set during
model selection. Its weaknesses: glioma recall is still the lowest (79.5%), it
still misses 44% of tumors in the external set, and results are from a single
training seed. Full numbers: `results/metrics/v2_comparison.json`.

## How to reproduce
```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh      # needs Kaggle credentials
bash scripts/reproduce.sh
python scripts/fill_readme_results.py
```
