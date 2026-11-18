# Demo script (3–5 minutes)

**Before you start (2 minutes earlier):** open
https://udit013-brain-tumor-mri-classifier.hf.space/api/ready to wake the Space
(it sleeps after ~48 h idle; waking takes about a minute), then open
https://neuroclass-gilt.vercel.app in another tab.

## 0:00 — Frame it (20 s)
"NeuroClass started as my IEEE 2024 paper on brain-tumour MRI classification.
This is what it became when I asked how to show one prediction *with its
limits* — and run it for zero dollars. It's research and education only, not a
diagnostic tool."

## 0:20 — The honest benchmark (40 s) — *Benchmarks* page
- Point at the separation: the paper's reported 99.844% vs measured results.
- "The paper's number was scored on 640 of 1,311 test images, and 45% of test
  images have a near-duplicate in training. The retrained model scores 93.4%
  overall, 89.4% on genuinely new images, 72.7% on a different dataset."
- Toggle "Show as table" on a chart (accessibility: every chart has one).

## 1:00 — An analysis (80 s) — *Workbench*
- Pick the glioma example, add the question "How reliable is Grad-CAM for
  brain tumour MRI?", run it. Point at the live step chips (SSE from the
  LangGraph workflow).
- Result: glioma at 35% calibrated confidence, **Unreliable**. "The rules say
  this one shouldn't be trusted: confidence near guessing, 4 of 6 augmented
  views agree, and glioma is the model's weakest class. On test, the rules'
  'standard' results are right 96% of the time and 'caution' ones 47%."
- Grad-CAM: "where the score came from — not a tumour outline."
- Explanation: click a citation to open the exact passage with its licence;
  show the removed-statements box. "The verifier deletes anything whose
  citation wasn't retrieved or whose numbers aren't in the source."
- Download the HTML report.

## 2:20 — Human in the loop (40 s) — signed in as reviewer
- Run the same image with review on; status becomes *awaiting review*.
- Open *Review queue*, approve with a note; the workflow resumes and completes.
- "That pause is a LangGraph interrupt with a database checkpoint; I verified
  it survives an API restart on Postgres."

## 3:00 — Research assistant (30 s)
- Ask "What share of NeuroClass test images have a near-duplicate in training?"
  → the answer cites the project's own documentation.
- "Retrieval routes project questions to project documents; that took recall@10
  from 0.79 to 1.00 on my labelled set."

## 3:30 — How it runs for $0 (45 s) — *System* page
- Show the live status: PostgreSQL (Neon, persistent), pgvector index, Gemini
  free tier with today's call budget, `paid_allowed: false`.
- "TensorFlow was only needed for Grad-CAM, so I export one ONNX graph and
  compute Grad-CAM in closed form — 2.2× faster, a third of the memory, verified
  against TensorFlow. The backend is a free Hugging Face Space, the frontend
  Vercel Hobby, the database Neon's free tier with pgvector. Gemini's free tier
  drafts explanations behind a quota guard; when the quota runs out it falls
  back to quoted, cited passages. It can degrade, but it can't bill."

## 4:15 — Close (15 s)
"Everything I showed is measured, tested in CI — including a job that installs
exactly what Hugging Face installs — and documented, including what I couldn't
measure yet."

**If something fails live:** the Space is asleep → open the Space URL, wait for
the landing page to show "Operational", then retry; or walk through the recorded
numbers in `docs/EVALUATION.md`.
