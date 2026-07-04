# Model Card — EfficientNetB3 Brain Tumor Classifier

**Live demo:** https://huggingface.co/spaces/Udit013/brain-tumor-mri-classifier
· **Model:** https://huggingface.co/Udit013/brain-tumor-efficientnetb3
· **Code:** https://github.com/Udit013/Brain_Tumor_Classificatioin

## Model details
- **Architecture:** EfficientNetB3 (ImageNet-pretrained, fully trainable) with
  `pooling='max'` → BatchNorm → Dense(512, L2/L1-regularised) → Dropout(0.4) →
  Dense(256, regularised) → Dropout(0.2) → Dense(4, softmax). ~11.7M params.
- **Input:** 256×256×3 RGB, **raw [0,255] pixels** (EfficientNet applies its own
  rescaling internally — do not divide by 255).
- **Output:** softmax over `[glioma, meningioma, notumor, pituitary]`.
- **Optimizer / schedule:** Adamax(1e-3), categorical cross-entropy, 20 epochs,
  EarlyStopping(monitor=loss), ReduceLROnPlateau(val_loss), best-val_accuracy
  checkpoint.
- **Origin:** reproduces the best model from "Identifying Various Types of Brain
  Tumors using Deep Neural Network based Image Features," 2024 International
  Conference on Cognitive Robotics and Intelligent Systems (ICC-ROBINS), IEEE,
  2024. DOI: 10.1109/ICC-ROBINS60238.2024.10533941. The original notebooks are
  preserved byte-for-byte under [`/legacy`](legacy/).

## Intended use
- **Intended:** research, education, and ML-engineering demonstration of an
  end-to-end evaluated/served classification pipeline.
- **Out of scope:** **NOT a medical device.** Must not be used for diagnosis,
  triage, or any clinical decision-making. No regulatory clearance.

## Training data
- Msoud Nickparvar **"Brain Tumor MRI Dataset"** (Kaggle), a compilation of
  figshare CE-MRI + SARTAJ + Br35H. 7023 T1-weighted MRI images, 4 classes,
  predefined split 5712 train / 1311 test.
- **Preprocessing:** resize 256×256, RGB; horizontal flip augmentation (train).

## Evaluation data & metrics
- In-distribution: the predefined Kaggle test split (**1600 images** as
  currently distributed; the paper used 1311 — see dataset-drift limitation).
  Reported metrics: accuracy, per-class precision/recall/F1, confusion matrix,
  per-class ROC-AUC and PR/AP, ECE (pre/post temperature scaling).
- Out-of-distribution: a **separate** public dataset (Navoneel Chakrabarty
  binary tumor/no-tumor set, n=253), evaluated via binary collapse.
- **All numbers below are measured by `scripts/reproduce.sh`; none are
  transcribed by hand.**

## Limitations & ethical considerations
- **Train/test leakage:** the published split is image-level over a multi-source
  compilation of 3-D volumes, so near-identical slices from one scan can appear
  on both sides of the split. Patient/scan IDs are **not recoverable** from the
  Kaggle release, so a rigorous patient-level audit is impossible; the leakage
  module reports an image-similarity **lower bound** instead. The headline
  in-distribution accuracy should be read in this light. See README §2
  *Evaluation Limitations*.
- **Distribution shift (OOD gap):** out-of-distribution accuracy (72.3% binary
  collapse) is materially lower than the in-distribution number — expected, and
  the honest generalisation signal.
- **Apple-Metal BatchNorm recalibration:** on `tensorflow-metal` the backbone's
  BatchNorm moving averages do not converge under the published recipe, so
  inference-mode `predict()` initially collapsed toward `notumor` (53% accuracy).
  A standard BN-recalibration pass (forward passes over training data, no weight
  update) recovered 0.53 → 0.94. Folded into `train.py`; not needed on the CUDA
  setup used for the paper.
- **Calibration:** raw softmax is over-confident; temperature scaling is applied
  and the fitted T is reported. By default T is fit on the same test split it is
  reported on (caveat documented).
- **Bias:** demographic, scanner, and acquisition-protocol distributions of the
  compiled dataset are undocumented; performance across subgroups is unknown.
- **Robustness:** near-random under additive Gaussian noise (~25% at all
  severities) and degrades under heavy blur/JPEG; robust to brightness,
  contrast, and rotation. Mean accuracy across 6 corruptions x 5 severities is
  72.5% (vs 93.0% clean). See `results/metrics/robustness.json`.
- **Uncertainty:** TTA predictive entropy is higher on wrong (0.78) than correct
  (0.36) predictions, so entropy is a usable abstain/flag signal.

## Quantitative results (measured)
| Metric | Value |
|---|---|
| Reproduced accuracy (full test set) | 93.94% |
| Published reference (paper) | 99.844% |
| Macro ROC-AUC / avg-precision | 0.985 / 0.967 |
| Exact cross-split duplicates | 114 / 1600 (7.1%) |
| Near-duplicates (phash≤5) | 714 / 1600 (44.6%) |
| Accuracy on near-duplicate vs leak-free images | 98.9% vs 90.0% |
| External OOD accuracy (binary collapse, n=253) | 72.3% |
| ECE before → after temperature scaling | 0.0425 → 0.0136 (T=0.819) |
| Inference latency, Keras / ONNX-CPU / ONNX-CoreML | 118ms / 33ms / 18.5ms |

Full breakdown (per-class P/R/F1, confusion matrices, curves) in
`results/metrics/*.json` and `results/figures/`, regenerated by
`scripts/fill_readme_results.py` for [README.md](README.md) §2.

## How to reproduce
```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh      # needs Kaggle credentials
bash scripts/reproduce.sh
python scripts/fill_readme_results.py
```
