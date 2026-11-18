---
title: NeuroClass
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
startup_duration_timeout: 30m
pinned: false
license: mit
short_description: Research MRI classifier with Grad-CAM, cited literature, API
---

# NeuroClass — research platform API + demo

**⚠️ Research and education only. Not a medical device and not clinically validated.**
Outputs are an experimental model's predictions, never diagnoses.

This free CPU Space serves two things from one process:

- `/` — a small demo: upload a brain MRI slice, get the model-predicted class
  (glioma, meningioma, pituitary tumor, no tumor), a temperature-calibrated
  confidence, a Grad-CAM heatmap and a reliability level from measured rules.
- `/api` — the NeuroClass platform API ([docs](/api/docs)): LangGraph analysis
  workflow with human review, retrieval over open-access literature with
  verified citations, reports, and benchmark data. The web app calls it.

Inference is TensorFlow-free: ONNX Runtime plus Grad-CAM computed in NumPy,
verified against TensorFlow before export. The model extends an
[IEEE ICC-ROBINS 2024 paper](https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941);
the paper's reported accuracy is its own claim, measured results are separate.

**Free-tier behaviour.** The Space sleeps after about two days without visitors;
the first request then takes a minute or two. Without a `DATABASE_URL` secret,
analysis history is kept on the Space's temporary disk and resets on restart.
Without a `GEMINI_API_KEY` secret, explanations are extractive (quoted,
cited passages) rather than LLM-written. Nothing here can incur a charge.

Source, evaluations and documentation:
[github.com/Udit013/Brain_Tumor_Classificatioin](https://github.com/Udit013/Brain_Tumor_Classificatioin).
Example images: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset),
Masoud Nickparvar, CC BY 4.0.
