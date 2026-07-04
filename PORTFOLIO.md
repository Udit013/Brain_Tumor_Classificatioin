# Portfolio Summary

### Brain Tumor Classification: Production ML System (IEEE 2024)

**Live demo:** https://huggingface.co/spaces/Udit013/brain-tumor-mri-classifier
**Model:** https://huggingface.co/Udit013/brain-tumor-efficientnetb3
**Code:** https://github.com/Udit013/Brain_Tumor_Classificatioin
**Publication:** [Identifying Various Types of Brain Tumors using Deep Neural Network based Image Features](https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941), ICC-ROBINS 2024, IEEE (co-author)
**Stack:** Python, TensorFlow/Keras, EfficientNetB3, ONNX Runtime, Gradio, FastAPI, Grad-CAM, Docker, GitHub Actions, Hugging Face

**Description:** Production-grade extension of an IEEE-published brain-tumor MRI classifier. Beyond reproducing the original four-architecture benchmark, this work adds the evaluation rigor and deployment engineering that turn a high-accuracy notebook into a trustworthy, served system: leakage auditing, out-of-distribution testing, calibration, uncertainty, robustness, explainability, and a live web app.

**Bullets:**
- Co-authored an IEEE 2024 publication benchmarking CNN, VGG16, InceptionV3, and EfficientNetB3 for 4-class brain-tumor MRI classification, with EfficientNetB3 reaching **99.844% accuracy** on **7,023 MRI images** at just **11.7M parameters** (smallest of the four models).
- Audited the published benchmark and found the 99.844% is inflated by train/test leakage: a perceptual-hash audit surfaced **44.6% near-duplicate** and **114 exact-duplicate** images across the split, with **98.9% accuracy on leaked images vs 90.0% on novel ones**, and documented that patient-level de-duplication is irrecoverable from the multi-source compilation.
- Reproduced the EfficientNetB3 pipeline end-to-end (**93.94%** on the rebalanced test set; **macro ROC-AUC 0.985**) with per-class metrics, confusion matrices, ROC/PR curves, and confidence distributions, and quantified an honest **72.3% out-of-distribution accuracy** on a separate public MRI dataset.
- Calibrated confidence via temperature scaling (**ECE 0.0425 to 0.0136**), added Test-Time-Augmentation uncertainty (entropy **0.36 on correct vs 0.78 on wrong** predictions), and ran corruption-robustness testing across 6 perturbation types, exposing near-random accuracy under Gaussian noise.
- Deployed a **live Gradio web app** on Hugging Face Spaces backed by **ONNX Runtime** CPU inference, returning predicted class, calibrated confidence, uncertainty, Grad-CAM overlay, latency, and a medical disclaimer; model versioned on Hugging Face Hub.
- Engineered for reproducibility: modular package, pinned dependencies, one-command reproduction script, Dockerfile, pytest suite, and **GitHub Actions CI**, with a model card and every reported number backed by a measured run.
- Debugged and fixed a deployment-time accuracy collapse (53% to 94%) traced to non-converged BatchNorm statistics on Apple Metal, resolved with a recalibration pass.
