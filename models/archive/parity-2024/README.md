---
license: mit
tags: [image-classification, medical-imaging, brain-tumor, efficientnet, onnx]
library_name: keras
---
# EfficientNetB3 Brain Tumor MRI Classifier

4-class T1-weighted brain MRI classifier (glioma, meningioma, notumor, pituitary).
Reproduces the best model from an IEEE 2024 paper
(DOI: 10.1109/ICC-ROBINS60238.2024.10533941) and extends it with production
evaluation. Input: 256x256x3 RGB, raw [0,255] pixels (EfficientNet rescales
internally). Files: `efficientnetb3.onnx` (serving), `EfficientNetB3_model_weights.h5`
(Grad-CAM), `temperature.json` (calibration T).

**Not a medical device** — research/education only; dataset has documented
train/test leakage. Full evaluation + code:
https://github.com/Udit013/Brain_Tumor_Classificatioin
