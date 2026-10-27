# Model Card — NeuroClass (EfficientNetB3 Brain Tumor MRI Classifier)

**Live demo:** https://huggingface.co/spaces/Udit013/brain-tumor-mri-classifier
· **Model:** https://huggingface.co/Udit013/brain-tumor-efficientnetb3
· **Code:** https://github.com/Udit013/Brain_Tumor_Classificatioin

## Model details
- **Version:** retrained October 2026 (replaced the paper-recipe model, which is
  preserved — see "Previous model" below). Same architecture, file names and
  serving stack.
- **Architecture:** EfficientNetB3 (ImageNet-pretrained) with `pooling='max'` →
  BatchNorm → Dense(512, L2/L1-regularised) → Dropout(0.4) → Dense(256,
  regularised) → Dropout(0.2) → Dense(4, softmax). 11.7M params.
- **Input:** 256×256×3 RGB, **raw [0,255] pixels** (EfficientNet applies its own
  rescaling internally — do not divide by 255). Load the `.h5` weights into the
  architecture built with `weights=None` (`btc.model.build_model`).
- **Output:** softmax over `[glioma, meningioma, notumor, pituitary]`, then a
  temperature (T = 0.937, fit on the validation split) for calibrated confidence.
- **Training** (`btc/finetune.py`, 21.5 min on an Apple M3 Pro CPU, seed 75):
  1. head only on cached features of the frozen ImageNet backbone (each image
     and its mirror), Adamax 1e-3, early-stopped on validation accuracy;
  2. fine-tune from `block6a` up with flips, small rotations/zooms/shifts,
     brightness/contrast and Gaussian noise, Adamax 1e-4 reduced on validation
     plateaus, BatchNorm frozen throughout.

  Checkpoint, learning-rate schedule and early stopping are all driven by the
  validation split; the test set was used once, after selection.
- **Serving formats:** Keras `.h5` weights (Grad-CAM) and ONNX (inference). The
  ONNX export matches CPU Keras to within 5.3e-6 per output probability on all
  1,600 test images.
- **Origin:** builds on the best model from "Identifying Various Types of Brain
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
- Msoud Nickparvar **"Brain Tumor MRI Dataset"** (Kaggle, CC BY 4.0), a
  compilation of figshare CE-MRI + SARTAJ + Br35H. The paper used 7,023 images
  (5,712 train / 1,311 test); the dataset as currently distributed has 7,200
  (5,600 / 1,600).
- **Split:** Kaggle Training is divided into 4,759 training / 841 validation
  images (about 210 per class) by `btc/splits.py`, which groups near-duplicate
  images (perceptual-hash distance ≤ 5, or identical pixels) and keeps each
  group on one side: zero near-duplicate pairs straddle the split.
- **Preprocessing:** resize 256×256 RGB with PIL (`btc/preprocess.py`) — the
  same function used for evaluation and serving.

## Evaluation data & metrics
- **In-distribution:** the full 1,600-image Kaggle test split, plus its 886
  images that have no near-duplicate in Kaggle Training ("novel subset").
- **Out-of-distribution:** a separate public dataset (Navoneel Chakrabarty
  binary tumor/no-tumor set, n=253), scored by collapsing the three tumor
  classes into one.
- **Device and preprocessing:** all metrics are measured on **CPU** with the
  serving preprocessing — the same computation the deployed ONNX model performs.
- Every number here is produced by `scripts/reproduce.sh`
  (`results/metrics/*.json`); none is transcribed by hand.

## Quantitative results (measured on CPU, serving preprocessing)
| Metric | Value |
|---|---|
| Accuracy (full 1,600-image test set) | 93.44% |
| Macro ROC-AUC / average precision | 0.993 / 0.983 |
| Accuracy: novel images (n=886) vs images with a near-duplicate in training (n=714) | 89.4% vs 98.5% |
| External out-of-distribution accuracy (n=253) | 72.7% (tumor recall 56.1%, no-tumor specificity 99.0%) |
| ECE, raw → temperature-scaled (T fit on validation, reported on test) | 0.022 → 0.024 |
| Clean vs mean corrupted accuracy (6 corruptions × 5 severities, n=400) | 92.0% vs 90.7% |
| Single-image latency, Keras-CPU / ONNX-CPU / ONNX-CoreML | 78 / 34.5 / 20.3 ms |

| Class | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| glioma | 0.978 | **0.795** | 0.877 | 0.984 |
| meningioma | 0.872 | 0.950 | 0.909 | 0.990 |
| notumor | 0.930 | 0.998 | 0.963 | 0.999 |
| pituitary | 0.971 | 0.995 | 0.983 | 0.999 |

The paper's published 99.844% is not directly comparable: it was scored on
640 of 1,311 test images, from an older version of the dataset (see below).

## Comparison with the previous model
Scored once on the test set with the same code, before the upgrade
(`results/metrics/model_comparison.json`):

| Metric | Previous (paper recipe) | Current |
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
novel subset p = 2.6e-4, glioma p = 7.8e-13). The current model trained on
fewer images (4,759 vs 5,600), and the previous model's checkpoint was selected
on the test set, so the comparison if anything favours the previous model.

## Limitations & ethical considerations
- **Missed tumors (glioma).** Glioma is the weakest class: recall 79.5%; of 400
  glioma test images, 54 are predicted as meningioma and **25 (6.25%) as
  `notumor`**. On the external set the model finds only 56% of tumors. A missed
  tumor is the most consequential error for any screening use.
- **Train/test leakage:** the published split is image-level over a
  multi-source compilation of 3-D volumes, so near-identical slices from one
  scan can appear on both sides. Patient/scan IDs are **not recoverable** from
  the Kaggle release, so the leakage module reports an image-similarity **lower
  bound**. The model is 9.1 points less accurate on novel images than on leaked
  ones; the novel-subset accuracy (89.4%) is the more honest in-distribution
  number.
- **Distribution shift:** out-of-distribution accuracy (72.7%) is far below the
  in-distribution number — the honest generalisation signal.
- **Single training run:** all numbers come from one training seed; run-to-run
  variation has not been measured.
- **Published metrics on partial test sets:** each paper notebook evaluates
  `1311 // 32 = 40` batches. With batch size 16, EfficientNetB3's 99.844% is
  exactly 639/640 — 49% of the test set, unshuffled. The other three models
  were scored on 1,280 images, so the comparison used different denominators.
- **Metal vs CPU:** on Apple Silicon, `tensorflow-metal`'s *graph* mode (used
  by Keras `predict` and `fit`) computes a different function from CPU
  (the previous model: 93.81% vs 90.25%); eager mode matches CPU. This model was
  trained and evaluated on CPU; CPU TensorFlow and ONNX Runtime agree.
- **Robustness:** above 80% at every tested severity of Gaussian noise, blur,
  brightness, contrast, JPEG compression and rotation, lowest under heavy blur
  (81%). Other shifts (different scanners, sequences, slice orientations) are
  untested beyond the external set.
- **Uncertainty:** the Space's stability level (agreement of 6 slightly altered
  views plus calibrated confidence) is a fixed heuristic that flags fragile
  answers; a "High" level is not evidence that an answer is correct.
- **Bias:** demographic, scanner, and acquisition-protocol distributions of the
  compiled dataset are undocumented; performance across subgroups is unknown.

## Previous model (preserved for rollback)
The exact reproduction of the paper's recipe, deployed until October 2026
(trained on all of Kaggle Training through `tensorflow-metal`, checkpoint
selected on the test set, BatchNorm statistics re-estimated after training).
On the same evaluation it scores 90.25% accuracy (85.7% novel), glioma recall
64.2% with 33 gliomas called `notumor`, ECE 0.188 → 0.068 after temperature
scaling fit on the test set, and 69.6% out of distribution. Its weights, ONNX
export and live temperature (T = 0.561) are on the Hub at tag `parity-2024` and
in `models/archive/parity-2024/` (with SHA-256 checksums); its metrics and
figures are in `results/archive/parity-2024/`. The Space serves it again with
the variable `NEUROCLASS_REVISION=parity-2024` (or, for a full rollback, restore
the model repo's `main` and the Space code from their `parity-2024` tags). `python -m btc.train`
reproduces it into `models/EfficientNetB3_parity_weights.h5`.

## How to reproduce
```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh      # needs Kaggle credentials
bash scripts/reproduce.sh          # trains with btc.finetune if no weights exist
python scripts/fill_readme_results.py
```
