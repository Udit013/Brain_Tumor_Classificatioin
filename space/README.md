---
title: NeuroClass
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# NeuroClass — Brain Tumor MRI Classifier (EfficientNetB3)

Upload a 2-D brain MRI slice (or pick one of the four examples) to get the
predicted class (`glioma`, `meningioma`, `notumor`, `pituitary`), a
temperature-calibrated confidence, a plain-language stability level (do 6
slightly altered versions of the image agree?), ONNX-Runtime CPU latency, and a
Grad-CAM heatmap.

This Space is the deployed front end of a production extension of an IEEE 2024
publication; the model was retrained in October 2026 (93.4% test accuracy,
89.4% on test images with no near-duplicate in training). Inference runs on **ONNX Runtime**; Grad-CAM uses the Keras model.
Weights are pulled from the [model repo](https://huggingface.co/Udit013/brain-tumor-efficientnetb3)
(branch `main`; the optional Space variable `NEUROCLASS_REVISION` selects another
branch, tag or commit — `parity-2024` is the previous model). Example images: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset),
Masoud Nickparvar, CC BY 4.0.

**⚠️ Not a medical device.** Research/education only. The underlying dataset has
documented train/test leakage, so real-world accuracy on novel patients is
materially lower than benchmark numbers. See the
[code + full evaluation](https://github.com/Udit013/Brain_Tumor_Classificatioin).
