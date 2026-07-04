---
title: Brain Tumor MRI Classifier
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

# Brain Tumor MRI Classifier (EfficientNetB3)

Upload a T1-weighted brain MRI to get the predicted tumor class
(`glioma`, `meningioma`, `notumor`, `pituitary`), a temperature-calibrated
confidence score, TTA-based uncertainty, ONNX-Runtime CPU latency, and a
Grad-CAM attribution overlay.

This Space is the deployed front end of a production extension of an IEEE 2024
publication. Inference runs on **ONNX Runtime**; Grad-CAM uses the Keras model.
Weights are pulled from the [model repo](https://huggingface.co/Udit013/brain-tumor-efficientnetb3).

**⚠️ Not a medical device.** Research/education only. The underlying dataset has
documented train/test leakage, so real-world accuracy on novel patients is
materially lower than benchmark numbers. See the
[code + full evaluation](https://github.com/Udit013/Brain_Tumor_Classificatioin).
