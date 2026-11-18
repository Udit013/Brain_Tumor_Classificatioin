---
title: NeuroClass API
emoji: 🧠
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
startup_duration_timeout: 30m
pinned: false
license: mit
short_description: Backend API of the NeuroClass research platform
---

# NeuroClass API

**The NeuroClass app is at [neuroclass-lab.vercel.app](https://neuroclass-lab.vercel.app).**
This Space is its backend: a FastAPI service at [`/api`](https://udit013-brain-tumor-mri-classifier.hf.space/api/docs)
for model inference with Grad-CAM, the LangGraph analysis workflow with human
review, literature retrieval with verified citations, and reports.

**⚠️ Research and education only. Not a medical device and not clinically validated.**

- Inference: EfficientNetB3 exported to ONNX, Grad-CAM computed in NumPy
  (verified against TensorFlow before export).
- Storage: PostgreSQL with pgvector (Neon, free plan).
- Explanations: Gemini API free tier when configured, otherwise quoted
  passages; every statement is checked against its sources.
- The model extends an [IEEE ICC-ROBINS 2024 paper](https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941);
  the paper's reported accuracy is its own claim, measured results are separate.

Runs on free CPU hardware; after about two days without traffic it sleeps and
the first request takes a minute or two. The Space uses the Gradio SDK as a
plain Python host (free accounts can't create Docker Spaces); it serves no
Gradio interface.

Source and documentation: [github.com/Udit013/Brain_Tumor_Classificatioin](https://github.com/Udit013/Brain_Tumor_Classificatioin).
