# Project Documentation: NeuroClass — Brain Tumor MRI Classification System

**A single source of truth for understanding this repository end to end** — for interview prep, for recruiters, for a new contributor, and for anyone with zero programming background who wants to understand what this system actually does.

> **Important framing note before you read further.** This document follows a comprehensive documentation template that assumes a typical web application (frontend framework, database, user authentication, React components, hooks, etc.). **This repository is not that kind of project.** It is a Python machine-learning system: a computer-vision classifier, a battery of evaluation/audit tooling, and two thin serving layers (a Gradio web app and a FastAPI service). There is **no database, no user accounts, no login, no frontend framework, no React**. Every section below is mapped onto what **actually exists** in this codebase — where a requested topic (e.g. "database schema," "JWT auth," "Redux state") has no real equivalent here, that is stated explicitly rather than invented. Nothing in this document describes code that isn't in the repository.

---

## Table of Contents

1. [The Story: What This Project Is, From Zero](#1-the-story-what-this-project-is-from-zero)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Layout](#3-repository-layout)
4. [The Two Layers of This Repo: Published Research vs. Production Extension](#4-the-two-layers-of-this-repo-published-research-vs-production-extension)
5. [The Core Package: `src/btc`, File by File](#5-the-core-package-srcbtc-file-by-file)
6. [The Serving Layer: Gradio App & FastAPI Service](#6-the-serving-layer-gradio-app--fastapi-service)
7. [Automation Layer: `scripts/`](#7-automation-layer-scripts)
8. [Tests & Continuous Integration](#8-tests--continuous-integration)
9. [Configuration & Packaging Files](#9-configuration--packaging-files)
10. [Data Storage: How This Project Handles "The Database Question"](#10-data-storage-how-this-project-handles-the-database-question)
11. [Authentication & Secrets: How This Project Handles "The Auth Question"](#11-authentication--secrets-how-this-project-handles-the-auth-question)
12. [State Management: How This Project Handles "The State Question"](#12-state-management-how-this-project-handles-the-state-question)
13. [Every Important Function, Explained](#13-every-important-function-explained)
14. [Deep Technical Concepts (Beginner → Advanced)](#14-deep-technical-concepts-beginner--advanced)
15. [Design Decisions and Trade-offs (the WHY)](#15-design-decisions-and-trade-offs-the-why)
16. [API Reference](#16-api-reference)
17. [Execution Traces: Tracing Every Major Action Through the System](#17-execution-traces-tracing-every-major-action-through-the-system)
18. [Glossary](#18-glossary)
19. [Interview Question Bank](#19-interview-question-bank)
20. [Common Follow-Up / "What If" Questions](#20-common-follow-up-what-if-questions)
21. [Complete End-to-End Walkthrough](#21-complete-end-to-end-walkthrough)

---

## 1. The Story: What This Project Is, From Zero

Imagine you're a doctor looking at an MRI scan of someone's brain. You need to figure out: is there a tumor here, and if so, what kind? There are three common tumor types this matters for — **glioma**, **meningioma**, and **pituitary tumor** — plus the case of **no tumor** at all. Telling these apart by eye takes years of training, and even trained radiologists can disagree.

**The problem this project solves:** can a computer look at a brain MRI image and predict which of these four categories it falls into, accurately enough to be useful as a second opinion or research tool?

**Who it's built for:**
- **Researchers and ML engineers** who want to see a rigorously-evaluated, honestly-reported medical imaging classifier (not just a headline accuracy number).
- **Recruiters/interviewers** evaluating the author's ability to build and reason about production ML systems.
- **Developers** who want to reproduce or extend the pipeline.
- **Anyone curious** who wants to try uploading an MRI to the [live demo](https://huggingface.co/spaces/Udit013/brain-tumor-mri-classifier) and see a real prediction with an explanation of *why* the model predicted it.

**This project has two distinct historical layers**, and understanding the difference is the single most important thing for understanding everything else:

1. **Published research (2024).** Four different neural network architectures were trained and benchmarked on a public Kaggle dataset of ~7,000 brain MRI images. This work was published at an IEEE conference. The best model, **EfficientNetB3**, reached 99.844% accuracy. These four original Jupyter notebooks are preserved, unmodified, in this repo's `legacy/` folder.

2. **Production extension (2026, this repo's main content).** Long after publication, the original author went back and asked: *is that 99.8% number actually trustworthy?* The answer, discovered through original investigation, was **no** — a meaningful chunk of it is inflated by a data-leakage problem in the public dataset itself. The rest of this repository (`src/btc/`, the live demo, the audits) is the honest, rebuilt version: it reproduces the original model, proves and quantifies exactly how much the published number is inflated, adds calibration/uncertainty/robustness testing that the original paper never had, and deploys the model as a live, usable web application.

### What happens when a user opens the application (the simplest possible flow)

There are actually **two different "users" of this system**, and they have completely different experiences:

**User type A — someone visiting the live demo website (no code, no setup):**
1. They open a URL in their browser: `https://huggingface.co/spaces/Udit013/brain-tumor-mri-classifier`.
2. They see a page with a title, an image upload box, and a "Classify" button.
3. They upload (or drag-and-drop) an MRI image.
4. They click "Classify."
5. A few hundred milliseconds to a couple of seconds later, they see: the predicted tumor class, a confidence percentage, a heatmap image showing *which part of the MRI* the model focused on, an uncertainty score, how long the prediction took, and a clear disclaimer that this is not a medical device.

**User type B — a developer or researcher reproducing the pipeline (has a terminal, wants to retrain/re-evaluate everything):**
1. They clone this Git repository.
2. They run a setup script that creates a Python virtual environment with the exact right dependency versions.
3. They run a download script that pulls the training dataset from Kaggle.
4. They run one command, `bash scripts/reproduce.sh`, which trains the model from scratch, evaluates it, audits it for data leakage, tests it against a completely separate dataset, calibrates its confidence, explains its predictions visually, exports it to a fast inference format, stress-tests it against corrupted images, measures its uncertainty, and builds a drift-detection baseline — ten sequential steps, each one producing a JSON file of real, measured numbers and (often) a chart.
5. A final script reads all those JSON files and stamps the real numbers into this project's `README.md`, so the documentation can never contain a number that wasn't actually computed on that machine.

Everything else in this document is about *how* those two flows actually work under the hood.

---

## 2. High-Level Architecture

There is no client-server web app with a database in the traditional sense. Instead, picture **two independent consumers reading from a shared, versioned "model artifact"** produced by a training/evaluation pipeline:

```text
                         ┌─────────────────────────────────────┐
                         │   TRAINING & EVALUATION PIPELINE     │
                         │        (src/btc/*.py, offline,       │
                         │     run via scripts/reproduce.sh)    │
                         │                                       │
                         │  data.py → model.py → train.py        │
                         │       │                               │
                         │       ▼                               │
                         │  evaluate.py, leakage.py,              │
                         │  external_validation.py,               │
                         │  calibration.py, explain.py,           │
                         │  robustness.py, uncertainty.py,        │
                         │  monitoring.py, export_onnx.py         │
                         └───────────────┬───────────────────────┘
                                         │  produces artifacts
                     ┌───────────────────┼────────────────────────┐
                     ▼                   ▼                        ▼
           models/*.h5, *.onnx   results/metrics/*.json    results/figures/*.png
           (weights, ONNX graph,  (every measured number,   (confusion matrix, ROC,
            calibration temp)     no hand-typed numbers)     Grad-CAM, robustness…)
                     │
                     │  models/*.h5 + *.onnx + temperature.json are
                     │  uploaded once, manually, to Hugging Face Hub
                     ▼
        ┌────────────────────────────────────────┐
        │   Hugging Face Hub model repo            │
        │   Udit013/brain-tumor-efficientnetb3     │
        │   (artifact storage — the closest thing  │
        │    this project has to a "database")     │
        └───────────┬──────────────────┬──────────┘
                     │ downloaded at    │ loaded on disk
                     │ startup          │ (models/ folder, local)
                     ▼                  ▼
   ┌───────────────────────────┐  ┌──────────────────────────┐
   │  space/app.py               │  │  src/btc/serve/app.py     │
   │  Gradio web app              │  │  FastAPI service           │
   │  deployed on HF Spaces       │  │  (uvicorn, local/Docker)   │
   │  = the LIVE PUBLIC DEMO      │  │  = alternative REST API    │
   └──────────────┬────────────┘  └────────────┬──────────────┘
                  │  browser HTTP               │  HTTP (multipart file)
                  ▼                              ▼
            End user's browser              curl / any HTTP client
```

**Client:** a plain web browser (for the Gradio demo) or any HTTP client such as `curl` (for the FastAPI service). There is no custom frontend JavaScript framework — Gradio auto-generates the entire browser UI from a Python description of the page.

**Server / compute:** Python processes. Two kinds run in production:
- The **Gradio app** (`space/app.py`), hosted on Hugging Face Spaces' free CPU tier — this *is* the live, public-facing "backend + frontend combined" for the demo.
- The **FastAPI app** (`src/btc/serve/app.py`), an alternative REST API meant to be run locally or in the provided Docker container.

**"Database":** none. All persistent state is either (a) files on disk (`models/`, `results/`), or (b) the Hugging Face Hub model repository, which acts as a simple, versioned artifact store (analogous to an S3 bucket or a Docker registry, not a relational database).

**AI model:** a single convolutional neural network, **EfficientNetB3** (a well-known, publicly pretrained image-classification architecture), fine-tuned on brain MRI images. It exists in two interchangeable formats: the original Keras/TensorFlow format (`.h5` weights file) and an exported **ONNX** format (`.onnx` file) used for faster serving.

**External services actually used:**
- **Kaggle** — source of the training dataset (`masoudnickparvar/brain-tumor-mri-dataset`) and the external out-of-distribution validation dataset.
- **Hugging Face Hub** — hosts the trained model files.
- **Hugging Face Spaces** — hosts the live Gradio demo (free CPU compute).
- **GitHub Actions** — runs the automated test suite on every push (Continuous Integration).

**Deployment architecture:** the Gradio Space is a single Docker-like container HF Spaces builds automatically from `space/requirements.txt` and `space/app.py`, pinned to Python 3.11. A separate, standalone `Dockerfile` in the repo root packages the FastAPI service for self-hosting (this Dockerfile is written and valid but has not been build-tested on the development machine, since Docker itself was not installed there — this is stated plainly rather than glossed over).

---

## 3. Repository Layout

```text
Brain_Tumor_Classificatioin/
├── CNN_Brain_Tumor.ipynb              ← original published notebook #1 (root copy, untouched)
├── VGG16_Brain_Tumor.ipynb            ← original published notebook #2
├── InceptionV3_Brain_Tumor.ipynb      ← original published notebook #3
├── EfficientNetB3_Brain_Tumor.ipynb   ← original published notebook #4 (the best model)
├── legacy/                            ← byte-for-byte backup copies of the four notebooks above
│   └── *.ipynb                        (checksummed identical to the root copies; read-only reference)
│
├── src/btc/                           ← THE CORE PACKAGE (installable as `btc`)
│   ├── __init__.py                    package docstring/version
│   ├── config.py                      central configuration: every constant, path, and hyper-parameter
│   ├── data.py                        loads the Kaggle dataset into TensorFlow generators/arrays
│   ├── model.py                       defines the EfficientNetB3 architecture, exactly
│   ├── train.py                       trains the model (+ BatchNorm recalibration fix)
│   ├── evaluate.py                    computes accuracy, per-class metrics, ROC/PR curves
│   ├── leakage.py                     THE core audit: detects train/test image duplication
│   ├── external_validation.py         tests the model on a totally separate dataset
│   ├── calibration.py                 fixes overconfident predictions (temperature scaling)
│   ├── explain.py                     Grad-CAM: visualizes what the model is "looking at"
│   ├── export_onnx.py                 converts the model to ONNX + benchmarks latency
│   ├── monitoring.py                  input-drift detection stub (production monitoring)
│   ├── robustness.py                  stress-tests the model against corrupted images
│   ├── uncertainty.py                 Test-Time Augmentation-based uncertainty estimation
│   └── serve/
│       ├── __init__.py
│       └── app.py                     FastAPI REST service
│
├── space/                             ← the live Gradio demo, deployed separately to HF Spaces
│   ├── app.py                         self-contained Gradio application
│   ├── requirements.txt               pinned dependencies FOR THE SPACE (not the whole repo)
│   └── README.md                      HF Spaces config header (YAML front-matter) + demo description
│
├── scripts/                           ← automation / one-command workflows
│   ├── setup_env.sh                   provisions a Python 3.11 virtualenv with pinned deps
│   ├── download_data.sh               downloads both datasets from Kaggle
│   ├── reproduce.sh                   THE master script: runs all 10 pipeline steps in order
│   ├── fill_readme_results.py         injects real measured numbers into README.md
│   ├── diagnose_bn.py                 one-off diagnostic script for the BatchNorm bug (see §14)
│   └── recalibrate_bn.py              one-off standalone BatchNorm-fix script
│
├── tests/
│   ├── test_core.py                   27 pure-logic unit tests (no TensorFlow/data/weights)
│   └── test_api.py                    5 FastAPI contract tests (stubbed model, no TensorFlow)
│
├── .github/workflows/ci.yml           GitHub Actions: runs tests on every push
│
├── models/                            ← GENERATED artifacts (git-ignored except a .gitkeep)
│   ├── EfficientNetB3_model_weights.h5    the trained model (canonical, BN-recalibrated)
│   ├── EfficientNetB3_raw_bnbroken.h5     the same model BEFORE the BN fix (kept for transparency)
│   ├── efficientnetb3.onnx                ONNX-exported version for fast inference
│   ├── temperature.json                    the single calibration scalar
│   └── drift_reference.npz                 PCA + histogram baseline for drift detection
│
├── results/                           ← GENERATED evidence (tracked in git — this IS the portfolio)
│   ├── metrics/*.json                 one JSON file per pipeline stage (evaluation, leakage, etc.)
│   └── figures/*.png                  confusion matrix, ROC curves, Grad-CAM overlays, etc.
│
├── Dockerfile                         packages the FastAPI service into a container
├── requirements.txt                   pinned deps for Linux/Docker/CUDA
├── requirements-macos.txt             pinned deps for Apple Silicon (tensorflow-macos/metal)
├── pyproject.toml                     makes `src/btc` pip-installable as the `btc` package
├── .gitignore                         excludes secrets, datasets, weights, caches
├── README.md                          the public-facing project overview + measured results table
├── MODEL_CARD.md                      formal model card (intended use, limitations, metrics)
└── PROJECT_DOCUMENTATION.md           this document
```

### Why this structure?

- **`src/` layout** (not a flat `btc/` folder at the repo root): this is a well-established Python packaging convention. It prevents accidentally importing the *uninstalled* source directory instead of the *installed* package, which is a common and confusing bug in flat layouts. `pyproject.toml` points `setuptools` at `src/` explicitly.
- **`legacy/` is a hard boundary.** The four original research notebooks must never be edited (their exact bytes are the citable record behind a real publication). Keeping them in an explicitly separate, clearly-named folder — never touched by any script — makes that guarantee visible just from the folder structure, not just a promise in a comment.
- **`space/` is separate from `src/btc/`** even though it duplicates some model-loading logic. This is deliberate: Hugging Face Spaces deploys *only* the contents of `space/` (plus its own `requirements.txt`), so it needs to be self-contained and not depend on the rest of this repository being present. See §15 for the trade-off this creates (some code duplication) and why it was accepted anyway.
- **`scripts/` vs `src/btc/`:** anything in `src/btc/` is importable Python library code (`python -m btc.evaluate`, etc.); anything in `scripts/` is a *workflow* — shell scripts or one-off Python scripts that orchestrate the library code, download external data, or perform a specific one-time fix. This separation means the core package stays a clean, reusable library, while the "how do I actually run this" logic lives in an obviously-named, disposable-feeling location.
- **`results/` is tracked in git; `models/` and `data/` are not.** This is a size/purpose trade-off: `results/` contains only small JSON files and PNG charts — this is the *evidence* the whole project's credibility rests on, so it belongs in version control. `models/` contains large binary weight files (tens of megabytes each) that are trivially regenerable by running the pipeline and are better hosted on Hugging Face Hub, which is built for large binary artifact hosting; committing them to git would bloat the repository for no benefit.

---

## 4. The Two Layers of This Repo: Published Research vs. Production Extension

This section exists because interviewers and readers consistently need this distinction spelled out explicitly.

| | **Section 1: Published Work** | **Section 2: Production Extension** |
|---|---|---|
| **When** | 2024 (IEEE conference paper) | 2026 (this repository's main content) |
| **Where the code lives** | `legacy/*.ipynb` (also duplicated, unmodified, at repo root) | `src/btc/`, `space/`, `scripts/`, `tests/` |
| **What it did** | Trained and benchmarked 4 architectures (CNN, VGG16, InceptionV3, EfficientNetB3) on a Kaggle dataset; reported headline accuracy numbers | Re-trained the best model (EfficientNetB3) with exact parity to the original recipe, then rigorously **audited** it and **deployed** it |
| **Headline number** | 99.844% accuracy (EfficientNetB3) — which turns out to be 639/640, scored on 49% of the test set (§4a) | 91.25% on the full 1,600-image test set, measured on CPU (the deployed computation) — **and** the finding that even that is inflated by leakage (97.2% on leaked test images vs 86.5% on genuinely novel ones) |
| **Is it still true?** | The number is real (it's what the model scored on that exact split at that exact moment), but this repo's own investigation later showed *why* it's an overly optimistic number for real-world use | This is the "honest" layer — every claim here traces back to a JSON file produced by running actual code |
| **Can I modify it?** | **No.** Never. Byte-for-byte preservation is enforced (see below) | Yes, this is the actively maintained code |

### Byte-for-byte notebook preservation, and why it matters

The four original Jupyter notebooks exist in **two identical copies**: one at the repository root (kept there because that's where they originally lived and where a reader would expect to find "the paper's code"), and one inside `legacy/` (kept as an explicitly-labeled, obviously-off-limits reference copy). Every time significant work has been done on this repository, the SHA-256 (and MD5) checksums of all four notebooks have been re-verified to prove they are unchanged, character for character, from the moment they were first copied in. This matters because these notebooks are the artifact behind a real, citable academic publication (IEEE, DOI `10.1109/ICC-ROBINS60238.2024.10533941`) — silently "improving" or "fixing" them after the fact would misrepresent what was actually published and reviewed.

### 4a. Audit of the published numbers (what 99.844% actually measures)

**Simple:** the paper's accuracy figures were each computed on only part of the test set, and the best model's on only about half of it.

**Technical:** every notebook calls `model.evaluate(test_generator, steps=test_generator.samples // 32)`, i.e. `1311 // 32 = 40` batches. Three notebooks use `batch_size=32`, so they score 1,280 images. The EfficientNetB3 notebook uses `batch_size=16`, so it scores only **640**, and the test generator is built with `shuffle=False`, so those are simply the first 640 files in order — not a stratified sample. Each published figure is an exact fraction of those denominators:

| Model | Images scored | Published | Exact fraction |
|---|---|---|---|
| CNN | 1,280 | 98.359% | 1,259 / 1,280 |
| VGG16 | 1,280 | 99.297% | 1,271 / 1,280 |
| InceptionV3 | 1,280 | 97.734% | 1,251 / 1,280 |
| EfficientNetB3 | 640 | 99.844% | **639 / 640** |

So the four-model comparison used different denominators. The EfficientNetB3 notebook's own `classification_report` on the full 1,311 images rounds to 0.99, so the magnitude is roughly right but the third decimal is unsupported. Separately, all four notebooks pass the test set as `validation_data` and keep the checkpoint with the best `val_accuracy` — the test set influenced model selection, which biases test metrics upward. `src/btc/train.py` reproduces both behaviours deliberately for parity; the fix (a validation split carved out of Training) is on the roadmap.

### 4b. Metal vs CPU: why every metric is measured on CPU

**Simple:** the same trained model gave different answers depending on which chip computed it. The live demo uses the CPU answers, so the documentation does too.

**Technical:** on Apple Silicon, `tensorflow-metal` 1.1.0 runs Keras ops on the GPU. For these weights it produces materially different outputs from TensorFlow's CPU kernels: **93.94% vs 91.25%** accuracy, disagreeing on 2.9% of test images, with a maximum per-class probability difference of 0.76. Two independent CPU runtimes — TensorFlow-CPU and ONNX Runtime — agree with each other to 2.4e-5 (100% identical predictions), and CPU is what the deployed Space (ONNX Runtime) and any Linux/Docker host run. Metal is therefore the outlier, and earlier versions of this repo were reporting a number the deployed model doesn't achieve.

The fix is `config.configure_inference_device()`, called inside `model.load_trained_model()` — the single function every evaluation and serving path uses to load the model. It hides the GPU from TensorFlow (`tf.config.set_visible_devices([], "GPU")`) before any op runs, so all metrics come from CPU; `BTC_INFERENCE_DEVICE=gpu` opts out. Training still uses Metal for speed. Re-estimating the BatchNorm statistics on CPU was also tried and scored *lower* on CPU (89.19% vs 91.25%), so the existing weights — which are exactly what the Space serves — were kept.

| Metric | On Metal (previously reported) | On CPU (deployed computation) |
|---|---|---|
| Accuracy | 93.94% | **91.25%** |
| Macro ROC-AUC | 0.985 | 0.980 |
| Near-dup vs leak-free accuracy | 98.9% vs 90.0% | **97.2% vs 86.5%** |
| ECE before → after temperature scaling | 0.0425 → 0.0136 (T = 0.819) | **0.196 → 0.073** (T = 0.561) |
| External OOD accuracy | 72.3% | 70.8% |
| Clean / mean-corrupted accuracy | 93.0% / 72.5% | 90.5% / 68.3% |
| TTA entropy, correct vs wrong | 0.36 vs 0.78 | 0.82 vs 1.00 |

---

## 5. The Core Package: `src/btc`, File by File

This is the heart of the repository. Every file here is a Python module; most are also runnable as standalone scripts via `python -m btc.<modulename>`.

### 5.1 `src/btc/config.py` — Central Configuration

**Location:** `src/btc/config.py`
**Purpose:** the single source of truth for every path, dataset constant, class name, and hyper-parameter used anywhere else in the package. Nothing else in `src/btc` hard-codes a magic number that also appears here.
**When it executes:** imported (not run directly) by every other module in the package, at the top of each file (`from . import config`).
**What imports it:** literally every other file in `src/btc/`.
**Major contents:**

```python
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # PARITY
IMG_SIZE = (256, 256)          # PARITY  target_size
IMG_SHAPE = (256, 256, 3)      # PARITY  RGB
BATCH_SIZE = 16                # PARITY  EfficientNetB3 notebook batch_size

N_TRAIN_PUBLISHED = 5712       # paper's train split size
N_TEST_PUBLISHED = 1311        # paper's test split size

EPOCHS = 20
STEPS_PER_EPOCH = N_TRAIN // 32        # -> 178 (see the "quirk" explained in §14)
```

Every constant tagged `# PARITY` in a comment is one that must **never be changed** without breaking the ability to faithfully reproduce the original published training recipe — this tag is a deliberate, greppable marker so a future editor knows immediately which lines are "sacred."

This file has two small functions. `ensure_dirs()` it creates the `models/`, `results/`, `results/figures/`, and `results/metrics/` directories if they don't already exist, using `Path.mkdir(parents=True, exist_ok=True)`. Every pipeline script calls this once at the start so it never has to worry about "directory doesn't exist yet" errors.

`configure_inference_device()` pins TensorFlow inference to CPU by hiding the GPU (`tf.config.set_visible_devices([], "GPU")`) before any op runs, unless `BTC_INFERENCE_DEVICE=gpu` is set. It exists because Apple's Metal GPU backend computes materially different outputs for this model than CPU (§4b); if it's called after TensorFlow has already initialised its devices, it warns instead of silently measuring on the GPU.

**Why a dedicated config module instead of scattering constants across files?** If `IMG_SIZE` were hard-coded independently in `data.py`, `model.py`, and `explain.py`, a future change (or a copy-paste typo) could silently desynchronize preprocessing between training and inference — exactly the kind of bug that's invisible until it causes a real accuracy regression. Centralizing it means there is exactly one place to look, and one place to change, per concept.

---

### 5.2 `src/btc/data.py` — Dataset Loading

**Location:** `src/btc/data.py`
**Purpose:** load the Kaggle brain-tumor MRI dataset from disk into the exact format the original published notebook used, plus a couple of extra helpers the new evaluation code needs.
**When it executes:** called by `train.py`, `evaluate.py`, `leakage.py`, `calibration.py`, `explain.py`, and `monitoring.py` — anything that needs images.
**Major functions:**

- **`_scan_split(split_dir)`** — walks a directory of class-subfolders (e.g. `Training/glioma/`, `Training/meningioma/`, …) and builds a pandas DataFrame with two columns: `filepaths` and `label`. This directly mirrors the `os.listdir` loop in the original notebook, so the resulting dataframe is structured identically.
- **`build_dataframes()`** — returns `(train_df, test_df)` by calling `_scan_split` on the Training and Testing folders.
- **`make_generators(seed)`** — the most important function in this file. It builds two Keras `ImageDataGenerator` objects and calls `.flow_from_dataframe(...)` on each, with `target_size=(256,256)`, `color_mode="rgb"`, and **no rescaling** (a deliberate, easy-to-miss detail — EfficientNet's own preprocessing layer does the normalization internally, so re-scaling to `[0,1]` here would double-normalize the pixels and quietly wreck accuracy). It then runs a defensive check: it re-derives the expected `{"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}` label mapping and raises a `RuntimeError` immediately if the generator's actual mapping doesn't match — this converts a silent, hours-later "why is my confusion matrix garbage" bug into an instant, loud failure at data-loading time.
- **`load_test_arrays()`** — unlike `make_generators`, this loads the *entire* test set into a single in-memory NumPy array (not a lazy generator). This is needed because several downstream tools (calibration, ROC curves, Grad-CAM, leakage cross-referencing) need random access to specific images by index, which a one-directional Keras generator doesn't support.

**One important piece of honesty logic added in the production layer** (not present in the original notebook): `make_generators` now checks whether the *actual* number of images found on disk matches the *published* split sizes (5,712 train / 1,311 test), and if they don't match, it raises a Python `warnings.warn(...)` — not silently, not hidden in a log file, but as a visible runtime warning. This exists because an investigation (see §14, "Dataset Drift") discovered that Kaggle's maintainer rebalanced the dataset after the paper was published, so today's download is actually 5,600/1,600, not 5,712/1,311. Rather than let that discrepancy silently confuse future reproduction attempts, the code detects and announces it every single time.

**Interview questions on this file:**
- *Beginner:* "What's the difference between a generator and loading everything into memory?" — A generator (`flow_from_dataframe`) yields one batch at a time on demand, using very little memory, but you can only iterate through it in order. Loading into a plain array (`load_test_arrays`) uses more memory up front but lets any piece of code grab any specific image instantly, which several of the audit tools need.
- *Intermediate:* "Why does `make_generators` raise on a class-index mismatch instead of just logging a warning?" — Because a silently-wrong label mapping doesn't crash anything; it just makes every subsequent number (accuracy, confusion matrix, ROC curve) wrong in a way that looks plausible. Raising immediately turns a subtle correctness bug into a fast, obvious, fixable one.
- *Advanced:* "Walk me through exactly why *not* rescaling to `[0,1]` matters here." — EfficientNet's Keras implementation includes its own internal `Rescaling`/normalization layer at the front of the network, calibrated to expect raw `[0, 255]` pixel values. If the data pipeline *also* divides by 255 before feeding images in, the network receives pixel values in roughly `[0, 1/255]` — far outside the distribution it was pretrained on — which silently and severely degrades accuracy without throwing any error. It's the kind of bug you only find by comparing your inference-time pipeline against the original training recipe line by line.

---

### 5.3 `src/btc/model.py` — Model Architecture

**Location:** `src/btc/model.py`
**Purpose:** define the exact EfficientNetB3-based neural network architecture used in the published paper, so that a fresh training run is architecturally identical.
**Major functions:**

- **`build_model(weights="imagenet")`** — constructs a Keras `Sequential` model:
  ```
  EfficientNetB3(pretrained on ImageNet, pooling='max')
    → BatchNormalization
    → Dense(512, L2 + L1 regularization) → Dropout(0.4)
    → Dense(256, L2 + L1 regularization) → Dropout(0.2)
    → Dense(4, softmax)
  ```
  compiled with the **Adamax** optimizer at learning rate `0.001` and `categorical_crossentropy` loss. Total parameters: ~11.7 million.
- **`load_trained_model(weights_path=None)`** — rebuilds the same architecture (with `weights=None`, skipping the slow ImageNet download since real weights are about to overwrite everything anyway) and loads the fine-tuned weights from disk. If the weights file doesn't exist, it raises a clear `FileNotFoundError` telling the user to run `python -m btc.train` first, rather than silently falling back to an untrained (ImageNet-only) model that would produce plausible-looking but meaningless predictions. It also calls `config.configure_inference_device()` first — because every evaluation and serving path loads the model through this one function, that single call pins all reported metrics to CPU.
- **`find_last_conv_layer_name(model)`** — a small utility used by the Grad-CAM explainability module. It walks the layers of the nested EfficientNetB3 backbone *in reverse* and returns the name of the last layer whose output is 4-dimensional (i.e. still a spatial feature map, shape `(batch, height, width, channels)`, as opposed to a flattened vector). This is the layer Grad-CAM needs to hook into.

**Why `Sequential` and not the Keras Functional API?** The original published notebook used `Sequential`, so this file matches it exactly for parity. (The Functional API is generally more flexible for multi-input/multi-output models, but this is a straightforward single-input, single-output classifier, so `Sequential` is the simpler, sufficient choice — and more importantly here, it's what needs to be reproduced exactly.)

**Why regularization (L1 + L2) on the dense layers?** With only ~7,000 training images and an 11.7-million-parameter network, overfitting is a real risk. L2 regularization (`kernel_regularizer`) discourages any single weight from growing too large; L1 regularization (`activity_regularizer`, `bias_regularizer`) encourages sparsity. This was a design choice in the *original* paper, preserved here for parity — not something added by the production extension.

---

### 5.4 `src/btc/train.py` — Training Loop (+ the BatchNorm Fix)

**Location:** `src/btc/train.py`
**Purpose:** run the actual training loop, reproducing the published recipe's callbacks and schedule, then apply a post-training fix that the original recipe never needed (because it was trained on different hardware).
**Execution:** `python -m btc.train`.

**Major function: `main()`**
1. Builds the train/test generators (`data.make_generators()`) and the model (`model.build_model(weights="imagenet")`).
2. Sets up three Keras callbacks:
   - `EarlyStopping(monitor="loss", patience=12)` — stops training if the *training* loss plateaus (not validation loss — this matches the original recipe).
   - `ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=6)` — halves-then-halves-again the learning rate if validation loss stalls.
   - `ModelCheckpoint(monitor="val_accuracy", save_best_only=True)` — saves only the single best-performing checkpoint, not every epoch.
3. Calls `model.fit(...)` for 20 epochs, with `steps_per_epoch=178` (see the "quirk" explained below) and `validation_steps=40`.
4. **After** training completes, calls the new `recalibrate_bn()` function (see below) and re-saves the weights.
5. Saves the full per-epoch training history (loss, accuracy, val_loss, val_accuracy, learning rate) as `results/metrics/train_history.json` — real numbers from the actual run, not estimates.

**Major function: `recalibrate_bn(model, train_generator, steps=350)`** — this is the fix for the most serious bug discovered during the production extension work (fully explained as a concept in §14). In short: it walks every `BatchNormalization` layer in the model (including inside the nested EfficientNetB3 backbone) and lowers its `momentum` from the original `0.99` down to `0.9`, then runs 350 additional forward passes over training data **in training mode** (`model(xb, training=True)`) **without ever computing gradients or updating any trainable weight**. This causes each BatchNorm layer's internal "running average" statistics (`moving_mean`, `moving_variance`) to re-converge to values that actually represent the data, fixing an inference-time collapse that had nothing to do with the learned weights being wrong.

**Interview questions:**
- *Beginner:* "What does `model.fit()` actually do?" — It repeatedly shows the model batches of training images with their correct labels, measures how wrong its predictions are (the loss), and nudges every internal weight slightly in the direction that would have made that specific batch's predictions more correct. It repeats this thousands of times.
- *Intermediate:* "Why call `recalibrate_bn` with `training=True` but no optimizer step?" — `training=True` tells BatchNorm layers to use the *current batch's* statistics (and update their running averages) rather than previously-stored ones. Simply not calling `.fit()` or an optimizer's `.apply_gradients()` means no trainable weight (the actual learned features) is touched — only the BatchNorm running statistics move. This is a legitimate, standard technique, distinct from further training.
- *Advanced:* "Why is `steps_per_epoch` set to `5712 // 32 = 178` when the actual batch size used is 16?" — This is a preserved quirk from the original notebook: it computes step count using a batch size of 32 (perhaps an earlier draft's batch size) but the generator itself was configured with `batch_size=16`. The result is that each epoch only shows the model `178 × 16 = 2,848` images, not the full 5,712-image training set — roughly half of it, silently, every epoch. The production code deliberately preserves this exactly rather than "fixing" it, documenting it instead as a faithful-reproduction detail — changing it would mean training a *different* recipe than the one that was actually published.

---

### 5.5 `src/btc/evaluate.py` — Evaluation Rigor

**Location:** `src/btc/evaluate.py`
**Purpose:** go far beyond a single accuracy number — this module is "Module 1" of the production extension, producing the full battery of metrics a rigorous evaluation needs.
**Execution:** `python -m btc.evaluate`.

**Major functions:**
- **`predict_test(model=None)`** — loads the trained model (or accepts an already-loaded one) and runs it over the *entire* materialized test array from `data.load_test_arrays()`, returning `(y_true, y_prob)` — the true integer labels and the full softmax probability matrix.
- **`compute_metrics(y_true, y_prob)`** — the analytical core. Computes:
  - Overall accuracy (`sklearn.metrics.accuracy_score`)
  - A full `classification_report` (precision, recall, F1, support, per class and averaged)
  - The confusion matrix (4×4 grid of true-vs-predicted counts)
  - Per-class ROC-AUC (one-vs-rest, via `roc_auc_score` after `label_binarize`-ing the labels)
  - Per-class average precision (area under the precision-recall curve)
  All numbers land in one dictionary, later serialized to `results/metrics/evaluation.json`.
- **`_plot_confusion_matrix`, `_plot_roc`, `_plot_pr`** — three separate plotting functions, each producing one saved PNG under `results/figures/`. Kept as three small, single-purpose functions rather than one large "make all the plots" function, so each can be tested, reused, or modified independently.
- **`main()`** — orchestrates the above, prints a short human-readable summary, and writes everything to disk.

**Why one-vs-rest ROC-AUC for a 4-class problem?** ROC curves are natively defined for binary classification (positive vs. negative). For a 4-class problem, the standard approach is "one-vs-rest": for each class, treat it as the positive class and all other three as negative, compute a ROC curve, then optionally average across classes ("macro" averaging, which is what this project reports — an unweighted average across the four per-class scores). This is what `label_binarize` plus a per-class loop accomplishes.

---

### 5.6 `src/btc/leakage.py` — Train/Test Leakage Analysis (the Most Important Module)

**Location:** `src/btc/leakage.py`
**Purpose:** this is the module behind the project's single most important finding. It audits whether images that look nearly identical to each other appear on *both* sides of the train/test split — a serious methodological problem for any dataset assembled from real-world 3D medical scans that are sliced into many 2D images.

**Why this matters, explained simply:** An MRI scan is really a 3D volume, sliced into dozens of 2D images for a dataset like this one. Two adjacent slices from the same patient's scan can look almost identical. If a slice from a patient ends up in the *training* set and a nearly-identical neighboring slice from the *same patient* ends up in the *test* set, the model doesn't need to learn to recognize tumors in general — it can partly get away with just recognizing "I've basically seen this exact image before." That inflates the reported test accuracy in a way that will **not** hold up on a genuinely new patient.

**Major functions:**
- **`_content_hash(path)`** — computes an exact-duplicate fingerprint: opens an image, converts to grayscale, resizes to a small 64×64 canvas, and computes an MD5 hash of the raw pixel bytes. Two images with the identical hash are, for practical purposes, pixel-identical (after minor recompression differences are washed out by the small resize).
- **`_phash(path)`** — computes a **perceptual hash** using the `imagehash` library's `phash()` function. Unlike a content hash, a perceptual hash is *designed* to produce similar hash values for visually similar (not just byte-identical) images, so two near-duplicate MRI slices will have a small "Hamming distance" (number of differing bits) between their hashes, even if their raw pixels differ slightly.
- **`analyse(eval_clean_subset=False)`** — the main audit function:
  1. Builds the exact content-hash of every training image, then checks every test image against that set — this is the **exact duplicate** count.
  2. For every test image, computes its perceptual hash and finds its *single nearest* training image by Hamming distance. If that nearest distance is ≤5 (a conventional "visually near-identical" threshold), the test image is flagged as a **near-duplicate**.
  3. If `eval_clean_subset=True`, it goes one step further: it splits the *already-computed* model predictions (from `evaluate.predict_test()`) into two groups — near-duplicate test images vs. genuinely novel ("leak-free") test images — and computes accuracy **separately on each group**. This is what produces the project's headline evidence: accuracy on near-duplicate images (97.2%) vs. accuracy on leak-free images (86.5%), measured on CPU (§4b).
- **`_plot_hist`** — plots a histogram of every test image's nearest-training-image distance, with a vertical line marking the near-duplicate threshold, saved as `results/figures/leakage_nn_hist.png`.

**An important, explicitly-documented limitation:** the Kaggle dataset does not retain patient or scan identifiers — it's a flat compilation of relabeled, re-indexed JPG files from three different original sources. This means there is no way to do a *true* patient-level leakage audit; the perceptual-hash method here is a **best-available proxy**, and the resulting 44.6% near-duplicate rate is documented explicitly as a **lower bound** on the true leakage rate, not a precise measurement. The code's docstring and the README both state this directly rather than implying more precision than the method actually has.

**Interview questions:**
- *Beginner:* "What is a hash, in simple terms?" — A hash is a short fingerprint computed from a larger piece of data (like an image), such that identical (or, for a perceptual hash, visually similar) inputs produce identical (or very close) fingerprints, while different inputs produce very different ones. It lets you compare millions of items cheaply without comparing every pixel of every pair directly.
- *Intermediate:* "Why perceptual hash instead of just checking file names?" — The dataset's file names were re-indexed by whoever compiled it into Kaggle, so identical or near-identical source images could easily have completely unrelated file names. The audit needs to compare actual visual content, not metadata that may have been scrambled.
- *Advanced:* "How would you validate that your leakage rate isn't itself an artifact of your hashing method (e.g. too aggressive a threshold)?" — Two ways this project does exactly that: (1) it reports both an exact-duplicate count (threshold-free, unambiguous) *and* a near-duplicate count (threshold-dependent), so a reader can see the floor even if they distrust the fuzzy threshold; (2) it doesn't just report a duplicate *count* — it re-measures actual model accuracy separately on the flagged-vs-clean subsets, so the claim ("this inflates accuracy") is backed by a direct behavioral measurement, not just a similarity-score argument.

---

### 5.7 `src/btc/external_validation.py` — Out-of-Distribution Testing

**Location:** `src/btc/external_validation.py`
**Purpose:** answer the question "does this model actually generalize, or does it only work well on data drawn from the exact same source it was trained on?" by testing it against a **completely separate** public MRI dataset that was never part of training.

**Major functions:**
- **`_scan_external(root)`** — walks a folder of class-named subfolders (this dataset's own convention, e.g. `yes/`, `no/`) and returns a list of `(path, folder_name)` pairs.
- **`_load_batch(paths)`** — loads a batch of images using the exact same preprocessing as the rest of the pipeline (256×256 RGB, no rescale).
- **`evaluate_external()`** — the main function. Because the external dataset (from a different Kaggle contributor, Navoneel Chakrabarty) only labels images as binary "tumor" vs. "no tumor" — not the finer-grained glioma/meningioma/pituitary distinction this model was trained on — the function **collapses** the model's 4-class prediction into a binary prediction (glioma, meningioma, and pituitary all map to "tumor"; "notumor" maps to "no tumor") before comparing against the external dataset's binary ground truth. This produces the project's honest **70.8% out-of-distribution accuracy** figure (n=253, CPU) — a large, expected drop from the 91.25% in-distribution number, and reported plainly rather than hidden.
- The function also supports an "overlapping classes" comparison path, for the (here, unused) case where an external dataset happens to share the exact same class taxonomy.

**Why binary collapse instead of just skipping this evaluation?** A model that can't be honestly evaluated on any outside data isn't trustworthy no matter how good its in-distribution numbers look. Binary collapse is a principled way to still extract a meaningful, apples-to-apples comparison even when the external dataset's labeling granularity doesn't match — at the cost of losing information about which *specific* tumor type predictions were correct.

---

### 5.8 `src/btc/calibration.py` — Confidence Calibration

**Location:** `src/btc/calibration.py`
**Purpose:** neural networks trained with cross-entropy loss are notoriously **overconfident** — a model might say "97% confident" when it's actually only right about 85% of the time at that confidence level. This module measures that gap and fixes it.

**Major functions:**
- **`expected_calibration_error(probs, y_true, n_bins=15)`** — implements **ECE** (Expected Calibration Error): it buckets predictions into 15 confidence bins (0–6.7%, 6.7–13.3%, …, 93.3–100%), and for each bin computes (a) the model's *average stated confidence* in that bin and (b) the model's *actual accuracy* in that bin. ECE is the weighted average absolute gap between those two numbers across all bins — a single scalar summarizing "how far off is the model's self-reported confidence from reality, on average."
- **`_fit_temperature(probs, y_true)`** — implements **temperature scaling**: it searches (via a coarse-then-fine grid search) for a single scalar `T` that, when the model's logits are divided by `T` before the softmax is applied, minimizes negative log-likelihood on the given data. A `T > 1` "softens" (flattens) an overconfident distribution; `T < 1` sharpens it.
- **`apply_temperature(probs, T)`** — applies a fitted temperature to a probability array by converting back to logits (`log(probs)`), dividing by `T`, and re-applying softmax.
- **`main()`** — orchestrates the whole flow: loads test predictions, optionally splits into a fit/report holdout (via `--holdout`), fits `T`, computes ECE before and after, saves a reliability-diagram plot, and writes the fitted temperature to `models/temperature.json` — the file both serving apps load at startup to calibrate their live predictions.

**An honestly-documented methodological caveat:** by default, the temperature is *fit* and its resulting ECE is *reported* on the **same** test split (because the published dataset only exposes a Training/Testing split, with no separate calibration set). This can be slightly optimistic. The code supports a cleaner `--holdout` flag that carves out a disjoint slice for fitting vs. reporting, but that wasn't the default run used for the headline numbers — and this exact caveat is spelled out in both the code's docstring and the README, rather than silently using the simpler (and slightly rosier) protocol without disclosure.

---

### 5.9 `src/btc/explain.py` — Grad-CAM Explainability

**Location:** `src/btc/explain.py`
**Purpose:** answer "why did the model predict this?" visually, by highlighting which pixels of the input MRI most influenced the prediction.

**Major functions:**
- **`_grad_model(model, last_conv_name)`** — builds a special auxiliary Keras model used only for computing Grad-CAM. This function embodies a real bug-fix (documented in the code's own docstring): building a single `Model` that spans from the outer model's input all the way to an *inner* convolutional layer's output triggers a Keras "Graph disconnected" error, because the EfficientNetB3 backbone is itself a nested sub-model, not a flat stack of layers. The fix: build the gradient-model over the **backbone alone** (a self-contained, connected graph, from `base.input` to `[last_conv_output, base.output]`), and apply the classifier head (the Dense layers) as a second, separate, ordinary function call on top. This returns a `(base_grad_model, head_layers)` tuple that `gradcam_heatmap` then uses.
- **`gradcam_heatmap(x_single, model, grad_model, class_idx)`** — the core Grad-CAM algorithm:
  1. Runs the image through the backbone inside a `tf.GradientTape()` (TensorFlow's mechanism for recording operations so it can later compute derivatives), capturing both the last convolutional feature map and, after manually applying the classifier head, the final class scores.
  2. Computes the gradient of the *predicted class's* score with respect to the last convolutional feature map — intuitively, "how much would nudging each spatial location of this feature map change the model's confidence in this specific class?"
  3. Averages those gradients over height and width to get one importance weight per feature-map channel.
  4. Computes a weighted sum of the feature map's channels using those importance weights, applies ReLU (keeping only positive influence), and normalizes to `[0, 1]` — the resulting 2D array is the heatmap.

  **Performance:** the forward + backward pass is compiled once into a TensorFlow graph by `_compiled_cam()` (`tf.function` with a fixed input signature, cached per grad model). On CPU this cut Grad-CAM from ~232 ms to ~26 ms per image (median of 20) with **bit-identical** heatmaps — verified against the previous eager version on 40 heatmaps across all four classes, for both the predicted-class and explicit-class paths. Before this, Grad-CAM was ~85–90% of every `/predict` request's time.
- **`overlay(x_single, heatmap, alpha=0.4)`** — resizes the (small) heatmap back up to the original image resolution, applies a "jet" color map (blue = low importance, red = high importance), and alpha-blends it on top of the original MRI image using OpenCV.
- **`explain_image(image_path, model)`** — a convenience wrapper used by both serving apps: given a raw image file, loads it, runs the full prediction + Grad-CAM pipeline, and returns the overlay image, predicted class, and probabilities.
- **`main()`** — the standalone script mode: if given `--image PATH`, explains that single image; otherwise, automatically finds one correctly-classified and one misclassified example *per class* (8 total, though not every class has a misclassified example) and saves both individual overlays and a combined montage figure.

**Why Grad-CAM specifically (and not, say, LIME or SHAP)?** Grad-CAM is computationally cheap (a single backward pass, no repeated model re-evaluation on perturbed inputs the way LIME/SHAP require), works natively with any CNN architecture that has spatial convolutional feature maps (which EfficientNetB3 does), and produces an intuitive, human-interpretable heatmap directly over the input image — exactly the right explanation format for an MRI, where "where in the image did the model look" is the natural question a viewer wants answered.

---

### 5.10 `src/btc/export_onnx.py` — ONNX Export & Latency Benchmarking

**Location:** `src/btc/export_onnx.py`
**Purpose:** convert the trained Keras model into the **ONNX** format (a framework-neutral representation of a neural network graph) and measure how much faster inference becomes as a result.

**Major functions:**
- **`export(model=None)`** — uses the `tf2onnx` library's `tf2onnx.convert.from_keras(...)` to trace the Keras model and write out `models/efficientnetb3.onnx`.
- **`_bench(fn, x)`** — a small, reusable timing harness: runs a given prediction function 5 times as warmup (to let TensorFlow/ONNX Runtime finish any lazy initialization or graph optimization), then times 50 real runs, reporting mean, median (p50), and 95th-percentile (p95) latency in milliseconds.
- **`benchmark()`** — runs the timing harness against (a) the original eager-mode Keras model and (b) an `onnxruntime.InferenceSession` loaded from the exported ONNX file, once per available execution provider (CPU always; CoreML too, if running on Apple hardware, since ONNX Runtime can offload to Apple's Neural Engine/GPU via CoreML).
- **`main()`** — exports, benchmarks, and writes `results/metrics/latency.json`, which is the source of the project's real, measured latency numbers (single image: 79 ms Keras-CPU / 34 ms ONNX-CPU / 18.6 ms ONNX-CoreML, 50-run means on the development machine). Because `load_trained_model()` pins inference to CPU, the Keras and ONNX-CPU numbers now compare the same hardware. An earlier version compared Keras on the Metal GPU (118 ms) against ONNX on CPU and reported a "3.5×" speedup — an apples-to-oranges comparison, corrected here.

**Why ONNX instead of just serving the Keras model directly?** Three separate, compounding reasons: (1) ONNX Runtime's graph-level optimizations (constant folding, operator fusion) typically make CPU inference meaningfully faster than TensorFlow's eager execution mode for a fixed model, which is exactly what the measured numbers here confirm (79 ms → 34 ms, about 2.4× with both on CPU); (2) it decouples the *serving* dependency from *TensorFlow specifically* — the ONNX Runtime package is smaller and has fewer transitive dependencies than full TensorFlow, which matters on constrained deployment environments like a free-tier cloud CPU instance; (3) ONNX is the more portable, standard interchange format if the model ever needed to be served from a different language or runtime later.

---

### 5.11 `src/btc/monitoring.py` — Input-Drift Detection Stub

**Location:** `src/btc/monitoring.py`
**Purpose:** simulate a production monitoring capability — detecting when the images being sent to the model in production start looking statistically different from what it was trained on (which would be an early warning sign that accuracy might be silently degrading).

**Major functions:**
- **`_embedder(model)`** — returns the EfficientNetB3 backbone submodel directly, which maps a raw image to its 1,536-dimensional internal feature vector ("embedding"). The docstring explicitly documents a second instance of the same "Graph disconnected" nested-model issue described in §5.9, and the same fix: use the backbone submodel as-is rather than trying to rewrap it.
- **`build_reference(limit=None)`** — the "training" side of the drift detector. It embeds a sample of training images, fits a `PCA` (Principal Component Analysis — a dimensionality-reduction technique) down to 8 components, and for each of those 8 components builds a 10-bin histogram of the training data's values. This entire reference (the PCA rotation matrix, the mean, the histograms, plus the mean and inverse covariance needed for a Mahalanobis-distance calculation) is saved to `models/drift_reference.npz`.
- **`_psi(ref_counts, new_vals, edges)`** — computes the **Population Stability Index (PSI)** between the reference histogram and a new batch of values: a standard, simple statistic from the credit-risk/MLOps world for quantifying how much a distribution has shifted. PSI above roughly 0.2 is a conventional "significant shift, investigate" threshold.
- **`detect(image_dir, tag="")`** — the "inference-time" side: embeds a new batch of images, projects them into the same PCA space, computes per-component and aggregate PSI against the saved reference, computes a mean Mahalanobis distance (a distance metric that accounts for the correlation structure between dimensions, not just raw Euclidean distance), and appends one JSON record to `results/metrics/drift_log.jsonl` — an append-only log, one line per detection run, so a history of drift measurements accumulates over time.

**Why call this a "stub"?** Because a real production monitoring system would need this to run automatically on a schedule against live traffic, alert a human (email/Slack/PagerDuty) when the PSI threshold is crossed, and probably persist to a proper time-series database rather than a flat JSONL file. This module deliberately implements the *statistical core* of that idea correctly and honestly, without over-claiming a full production monitoring platform that doesn't actually exist here.

---

### 5.12 `src/btc/robustness.py` — Corruption Robustness Testing

**Location:** `src/btc/robustness.py`
**Purpose:** measure how much the model's accuracy degrades when input images are corrupted in realistic ways — noise, blur, brightness/contrast shifts, JPEG compression artifacts, and rotation — at five increasing severity levels each.

**Major functions:**
- **`_corrupt(x, kind, sev)`** — applies one of six corruption types to a single image, at severity 1–5:
  - `gaussian_noise` — adds random noise sampled from a normal distribution, with standard deviation increasing from 8 to 46 across severities.
  - `gaussian_blur` — applies OpenCV's `GaussianBlur` with an increasing kernel size (3×3 up to 11×11).
  - `brightness` — adds a flat offset to every pixel (15 to 80).
  - `contrast` — pulls pixel values toward (or away from) the image's own mean, scaled by a shrinking factor (0.85 down to 0.3).
  - `jpeg_compression` — actually re-encodes the image through a real JPEG compressor at decreasing quality (50 down to 8) and decodes it back, producing genuine compression artifacts rather than a simulated approximation.
  - `rotation` — rotates the image by an increasing angle (5° to 30°) using an affine transform.
- **`evaluate_robustness(limit=400)`** — takes a balanced subset of the test set (100 images per class, by default), measures clean accuracy, then for every corruption type and severity, re-corrupts the whole subset and re-measures accuracy — 6 corruptions × 5 severities = 30 additional accuracy measurements, all real, all on the actual trained model.
- **`_plot(results, path)`** — plots one line per corruption type (accuracy vs. severity), saved as `results/figures/robustness_curves.png`.

**The most important actual finding from this module** (not something artificially engineered — this is what the real measurement produced): the model is **near-random under Gaussian noise at every single severity level tested**, including the mildest one, while it stays above 70% under brightness, contrast, and rotation shifts even at the harshest severity (76%, 71%, and 86% respectively at severity 5). This is a specific, actionable, non-obvious result — it tells a reader exactly which real-world failure mode (sensor/scanner noise) this particular model needs the most protection against, information a single averaged "68.3% mean corrupted accuracy" number would completely hide.

---

### 5.13 `src/btc/uncertainty.py` — Test-Time Augmentation Uncertainty

**Location:** `src/btc/uncertainty.py`
**Purpose:** produce a per-prediction uncertainty score — a signal for "how much should I trust this specific prediction?" — using a technique that's cheap enough to run in a live web demo on free CPU hardware.

**Major functions:**
- **`_augmentations(x)`** — given one image, generates 6 label-preserving variants: the original, a horizontal flip, two small rotations (±8°), and two brightness jitters (×0.9, ×1.1). "Label-preserving" is the key property — flipping or slightly rotating an MRI doesn't change which tumor type it shows, so the model *should* give consistent predictions across all 6 variants if it's genuinely confident.
- **`tta_predict(x_single, predict_fn, temperature=1.0)`** — runs all 6 augmented views through the model, averages their probability distributions, and computes:
  - `predictive_entropy` — the Shannon entropy of the *averaged* probability distribution (0 = completely certain, `log(4) ≈ 1.39` = maximally uncertain across 4 classes).
  - `mean_top_prob` / `std_top_prob` — the mean and standard deviation of the winning class's probability *across the 6 individual augmented predictions* — a second, complementary signal: low standard deviation means the model agreed with itself across all the small perturbations; high standard deviation means small, meaningless changes to the input flipped the model's confidence around, itself a red flag.
- **`mean_dataset_uncertainty(model, limit=200)`** — runs the above over a balanced sample of the test set and reports the average entropy separately for predictions that turned out **correct** vs. **wrong**. On the CPU (deployed) computation this gives mean entropy 0.82 on correct predictions vs. 1.00 on wrong ones: the signal points the right way, but the two distributions overlap heavily, so entropy is a weak abstain signal on its own and should be combined with other checks rather than thresholded directly. (On Metal the gap looked larger — 0.36 vs 0.78 — one of the numbers superseded in §4b.)

**Why Test-Time Augmentation instead of Monte Carlo Dropout (the other common lightweight uncertainty technique)?** Both are legitimate, cheap options that don't require retraining or a different architecture. TTA was chosen here specifically because it composes naturally with the ONNX Runtime serving path used in the live demo — it just means running the same exported ONNX model 6 times on 6 slightly different inputs, with no need to keep dropout layers "active" at inference time (which MC-Dropout requires, and which some inference runtimes, including ONNX Runtime by default, aren't set up to do easily, since dropout is normally disabled at inference for a reason). TTA is a pure, black-box, model-agnostic technique from the outside.

---

## 6. The Serving Layer: Gradio App & FastAPI Service

There are **two separate, independently-runnable ways** to actually use the trained model. They share the same underlying preprocessing and model logic but are built for different purposes.

### 6.1 `space/app.py` — The Live Public Demo (Gradio)

**Location:** `space/app.py`
**Purpose:** this is the entire live, public-facing application at `huggingface.co/spaces/Udit013/brain-tumor-mri-classifier`. It is intentionally **self-contained** — it does not import anything from `src/btc/`, because Hugging Face Spaces only deploys the `space/` folder's contents plus `space/requirements.txt`.

**What "Gradio" is, simply:** Gradio is a Python library that lets you describe a web page's layout and behavior *entirely in Python* — no HTML, no CSS, no JavaScript required — and it automatically generates a real, interactive browser-based UI, complete with drag-and-drop file upload, image display, buttons, and text rendering. Technically, it wraps a FastAPI server underneath and serves a pre-built JavaScript frontend that talks to that server over HTTP/WebSocket, but none of that is visible to someone writing a Gradio app — you just describe components and wire up a Python function.

**Startup sequence (runs once, when the Space container boots):**
1. `hf_hub_download(MODEL_REPO, "efficientnetb3.onnx")` and `hf_hub_download(MODEL_REPO, "EfficientNetB3_model_weights.h5")` — these download the model artifacts from the Hugging Face Hub model repository, caching them locally inside the container.
2. An `onnxruntime.InferenceSession` is created from the downloaded ONNX file — this is the object that actually runs fast predictions.
3. A second, full Keras model is separately rebuilt (architecture defined inline, duplicating `model.py`'s definition — see §15 for why) and its weights loaded from the downloaded `.h5` file. This Keras copy exists *purely* so Grad-CAM can be computed (ONNX Runtime doesn't support computing gradients, which Grad-CAM needs).
4. The fitted calibration temperature is downloaded and parsed (falling back to `1.0`, i.e. "no calibration," if that file happens to be missing, rather than crashing).
5. A `gr.Blocks()` layout is defined: a title, an upload `gr.Image`, a "Classify" `gr.Button`, an output `gr.Label` (for the 4-class probability bars), an output `gr.Image` (for the Grad-CAM overlay), and an output `gr.Markdown` block (for the free-text summary). The button's `.click(...)` event is wired to the `predict()` function.

**The `predict(image)` function — what runs on every single button click:**
1. Converts the uploaded image (a NumPy array, as Gradio hands it over) to a PIL image, converts to RGB, resizes to 256×256 — matching the exact preprocessing used everywhere else in the project.
2. Times and runs `_onnx_predict(x[None, ...])` — the fast path, using ONNX Runtime.
3. Applies the loaded calibration temperature to get calibrated probabilities.
4. Generates the 6 Test-Time-Augmentation views (`_tta_augment`) and runs them all through ONNX Runtime as one batched call, computes the mean probability distribution and its entropy.
5. Runs Grad-CAM (`_gradcam`) using the separate Keras copy of the model, for the predicted class specifically.
6. Formats everything — predicted class, calibrated and raw confidence, temperature value, entropy, latency, and the fixed medical disclaimer — into a single Markdown string.
7. Returns three values: the probability dictionary (rendered by Gradio as a bar chart), the Grad-CAM overlay image, and the Markdown info string — these map directly onto the three `outputs=[...]` the button's click handler was wired to.

**A real deployment bug this file's comments document explicitly:** the very first deployment attempt used Gradio 4.44, which crashed on startup with a `TypeError` deep inside Gradio's own auto-generated API schema builder (`get_api_info()`), a known issue with that specific version when a `gr.Label` component with dictionary-style outputs is used. Upgrading to Gradio 5.50.0 and passing `show_api=False` to `demo.launch()` fixed it — this exact fix, and the reasoning behind it, is preserved as an inline code comment rather than silently changed with no trace.

### 6.2 `src/btc/serve/app.py` — The FastAPI REST Service

**Location:** `src/btc/serve/app.py`
**Purpose:** an alternative to the Gradio demo — a plain JSON REST API, meant for programmatic use (`curl`, another backend service, a future custom frontend), meant to be run locally via `uvicorn` or inside the provided `Dockerfile`.

**Startup sequence:** unlike the Gradio Space, this app is designed to load its model **lazily** via an internal `_state` dictionary that starts out empty (`{"model": None, ...}`), populated by `_load()` either in the app's `lifespan` handler at startup (run in a worker thread so startup doesn't block the event loop), or — as a safety net — the first time `/predict` runs if startup loading didn't happen. Both paths take `_model_lock`, so two requests can never load the model twice, and the endpoint can never silently serve a `None` model.

**Two endpoints:**
- **`GET /health`** — returns `{"status": "ok", "model_loaded": bool, "calibrated": bool}`. A minimal liveness/readiness check, exactly the kind of endpoint the `Dockerfile`'s `HEALTHCHECK` directive calls automatically every 30 seconds.
- **`POST /predict`** — accepts a multipart file upload (`UploadFile`) and reads at most 10 MB + 1 byte of it. `_decode_upload()` rejects oversized uploads (413) and unreadable ones (400) *before* the model is ever touched, then opens and resizes the image with PIL into a raw float array (again, the *exact* same preprocessing everywhere: 256×256, RGB, no rescale), then hands the array to `_infer()` via `run_in_threadpool`. `_infer()` holds `_model_lock`, runs the *Keras* model directly (this app uses Keras throughout, not ONNX — see §15 for why the two serving apps differ here), applies calibration, computes the compiled Grad-CAM, base64-encodes the resulting PNG overlay so it can travel inside a JSON payload, and returns a single JSON object with the class name, class index, calibrated confidence, raw confidence, a `is_calibrated` boolean flag, the full per-class probability dictionary, the base64 Grad-CAM image, and a timing breakdown (`inference_ms`, `gradcam_ms`, `latency_ms`). Running `_infer()` in a worker thread is what keeps the event loop free: before this change, the CPU-bound work ran directly inside the `async` handler, and with 16 concurrent requests `/health` took up to 3.3 s to answer (now ≤57 ms).

**Error handling:** if the uploaded file can't be opened as an image (corrupted upload, wrong file type, etc.), the code catches the exception explicitly and re-raises it as a proper `HTTPException(status_code=400, ...)` with a human-readable detail message — the API never returns a raw, unhandled Python traceback to a client.

---

## 7. Automation Layer: `scripts/`

### 7.1 `scripts/setup_env.sh`
Provisions a Python 3.11 virtual environment (searching for `python3.11`, then `python3.10`, then `python3.9`, in that order, since TensorFlow 2.15 supports only those versions — and if none is found on macOS, it automatically runs `brew install python@3.11`). It then installs either `requirements-macos.txt` (if running on Apple Silicon, detected via `uname -m == arm64`) or `requirements.txt` (everywhere else), and finally installs the `src/btc` package itself in "editable" mode (`pip install -e .`), so `import btc` works from anywhere without needing to reinstall after every code change.

### 7.2 `scripts/download_data.sh`
Downloads two separate datasets from Kaggle using the `kaggle` CLI (which requires credentials — either a `~/.kaggle/kaggle.json` file or `KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables, **never** committed to this repository): the main training/testing compilation, and the separate external out-of-distribution dataset used by `external_validation.py`. It's written to be **idempotent** — it checks whether each dataset already exists on disk before re-downloading, so re-running the script is always safe and fast on a second run.

### 7.3 `scripts/reproduce.sh`
The master orchestration script — the single command that runs the entire pipeline end to end, ten sequential steps:

```text
1.  Train EfficientNetB3          (skippable with --skip-train if weights exist)
2.  Evaluation rigor
3.  Leakage analysis
4.  External validation
5.  Calibration
6.  Grad-CAM overlays
7.  ONNX export + latency
8.  Robustness testing
9.  Uncertainty (TTA)
10. Drift-monitoring reference
```

It activates the `.venv` if present, sets `PYTHONPATH=src` so the package is importable without a separate install step, and uses `set -euo pipefail` so the script halts immediately on any unhandled error rather than silently continuing with partial, misleading results (this exact behavior is what caused the script to correctly halt during actual development when a Grad-CAM bug was discovered mid-run — see §14).

### 7.4 `scripts/fill_readme_results.py`
The single, sole mechanism by which any number ever enters this project's `README.md`. It reads every JSON file in `results/metrics/`, and if **any** of the required files is missing, it raises `SystemExit` with a clear error rather than writing a partial or placeholder table — the design goal is that this project's public-facing documentation can never claim a number that wasn't actually computed by running the code. It formats nine specific metrics into a Markdown table and a separate robustness/uncertainty block, then does a precise text-replace between HTML comment markers (`<!-- RESULTS:BEGIN -->` … `<!-- RESULTS:END -->`) in `README.md`, leaving every other part of the file untouched.

### 7.5 `scripts/diagnose_bn.py` and `scripts/recalibrate_bn.py`
Two small, standalone diagnostic/fix scripts written during the actual live debugging of the BatchNorm inference-collapse bug (see §14). `diagnose_bn.py` runs the same model in both inference mode and training mode on the same balanced sample and prints both accuracies side by side — this is the exact script that produced the "0.53 inference-mode vs 0.78 training-mode" evidence that pinpointed BatchNorm as the culprit. `recalibrate_bn.py` is a standalone version of the same fix later folded permanently into `train.py`'s `recalibrate_bn()` function. Both are preserved in the repository as an honest record of the actual debugging process, rather than deleted after the fact to make the final code look like it was written correctly the first time.

---

## 8. Tests & Continuous Integration

### 8.1 `tests/test_core.py` and `tests/test_api.py`
32 fast tests in total. `test_core.py` holds 27 unit tests, using `pytest`, covering only the **pure-logic** parts of the codebase — code that doesn't need TensorFlow, a trained model, or the dataset to be present. This is a deliberate scope boundary: model- and data-dependent behavior (does training actually converge, does the pipeline reproduce the right accuracy) is validated separately by actually running `scripts/reproduce.sh`, which needs real data and takes much longer; CI is meant to catch *logic* regressions in seconds, not re-run the whole ML pipeline on every push.

Test groups:
- **config** — sanity-checks the class names, count, and image shape constants.
- **calibration** — checks that `apply_temperature` always returns a valid probability distribution (rows summing to 1), that temperature `1.0` is a no-op (identity), that ECE is near-zero for a perfectly-calibrated toy example, and that `_fit_temperature` always returns a positive value.
- **robustness** — parametrized across all 6 corruption types and 3 representative severities, checking that every corruption function preserves the image's shape and stays within the valid `[0, 255]` pixel range. (A further calibration test pins the ECE fix: a confidence of exactly 0.0 must still be counted in the first bin.)
- **uncertainty** — checks that the TTA augmentation function produces at least 4 views of the correct shape, and that feeding a deterministic "stub" fake model (one that always predicts class 2 with 100% confidence) into `tta_predict` correctly reports class 2 with near-zero entropy — a test of the *aggregation logic* using a fake model, rather than needing the real 11.7-million-parameter network.
- **leakage** — checks that two byte-identical images produce a perceptual-hash Hamming distance of exactly 0, and an identical content hash.

**Why "stub" a fake model instead of loading the real one in tests?** The real model is an 11.7M-parameter TensorFlow network requiring the actual trained weights file (tens of megabytes, not checked into git) and a full TensorFlow installation. A stub function that mimics the real model's *interface* (takes a batch, returns probabilities) but is instantaneous and deterministic lets the test suite verify the *aggregation and math logic* around the model (does TTA averaging work correctly, does entropy compute correctly) completely independently of whether the model itself is any good — a classic and valuable unit-testing technique called **mocking** or **stubbing**.

`test_api.py` holds 5 HTTP contract tests for the FastAPI service. They exist because `/predict` once returned HTTP 500 on *every* request — an annotation FastAPI couldn't resolve — and nothing caught it, since no test ever made a request. They use FastAPI's `TestClient` without a `with` block, which skips the model-loading lifespan, and swap `_infer()` for a stub. That means routing, upload validation (413 for oversized, 400 for non-images), the threadpool dispatch, the response contract, and the parity preprocessing (256×256×3, float32, *not* rescaled to [0, 1]) are all exercised in milliseconds, with no TensorFlow installed.

### 8.2 `.github/workflows/ci.yml`
A GitHub Actions workflow that runs automatically on every `push` or `pull_request` targeting `main`. It:
1. Checks out the repository.
2. Sets up Python 3.11.
3. Installs only the lightweight dependencies the pure-logic tests actually need (`numpy`, `pillow`, `opencv-python-headless`, `imagehash`, `pytest`, plus `fastapi`, `python-multipart`, and `httpx` for the API contract tests) — deliberately **excluding TensorFlow**, which would push the job from under a minute to many minutes and require downloading gigabytes of data the logic tests don't touch.
4. Runs `python -m py_compile` across every source file, as a fast syntax-correctness gate even for the modules that *do* import TensorFlow (this catches typos/syntax errors in those files without needing to actually execute or import them).
5. Runs the full `pytest` suite.

This CI configuration has been run and **passed with a `success` conclusion**, confirmed by watching the actual GitHub Actions run to completion — the green badge on the README reflects a genuinely passing pipeline, not an unverified assumption.

---

## 9. Configuration & Packaging Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Declares `btc-production` as a pip-installable package, points `setuptools` at the `src/` layout, and pins the supported Python range (`>=3.9,<3.12`) — bounded above by TensorFlow 2.15's Python support ceiling. |
| `requirements.txt` | Pinned exact-version dependencies for Linux/Docker/CUDA environments: `tensorflow==2.15.0`, `numpy==1.26.4` (capped below 2.0, which TF 2.15 doesn't support), plus imaging, ML-metrics, ONNX, and serving libraries. Audited with `pip-audit`: upgrading FastAPI 0.142.2, Starlette 1.7.0, python-multipart 0.0.32, Pillow 12.3.0, scikit-learn 1.5.2, ONNX 1.18.0, Protobuf 4.25.8 and tf2onnx 1.17.0 cut known advisories from 57 to 22 (vulnerable packages 7 → 3), verified by reproducing every evaluation metric and the API's behaviour on the new versions. The remaining 22 are blocked by TF 2.15: Keras 2.15 (fixes only in Keras 3), ONNX ≥1.19 (needs `ml_dtypes>=0.5`; TF pins `~=0.2`), and Protobuf ≥5 (TF requires <5). |
| `requirements-macos.txt` | The same dependency set, but swapping `tensorflow` for `tensorflow-macos` + `tensorflow-metal` (Apple's GPU-acceleration plugin for TensorFlow on Apple Silicon). Metal is used for **training only**: its kernels compute different outputs for this model than CPU (§4b), so evaluation and serving are pinned to CPU. |
| `space/requirements.txt` | A **separate, smaller** dependency list specific to the deployed Gradio Space — it includes `gradio`, `onnxruntime`, `tensorflow-cpu` (not the full `tensorflow` package, since no training happens in the Space, only inference), and imaging libraries. Kept independent of the root `requirements.txt` because the Space is deployed as an isolated environment. |
| `space/README.md` | Not a normal README — its content is almost entirely a YAML front-matter block (`title`, `emoji`, `sdk: gradio`, `sdk_version`, **`python_version: "3.11"`**, `app_file: app.py`) that Hugging Face Spaces reads to configure the container it builds. The `python_version` pin was a real, necessary fix (see §14) — without it, Spaces defaults to a newer Python that has no compatible TensorFlow/ONNX Runtime wheels. |
| `Dockerfile` | Builds a container for the FastAPI service: `python:3.11-slim` base, installs OS-level libraries OpenCV needs (`libgl1`, `libglib2.0-0`), installs pinned Python deps, installs the `btc` package, copies in the `models/` directory (which must be present or mounted at build/run time), exposes port 8000, defines a `HEALTHCHECK` that calls `/health`, and finally runs `uvicorn btc.serve.app:app`. |
| `.gitignore` | Excludes: Python build artifacts and caches; virtual environments; the `data/` directory (never commit raw MRI images); all model weight/artifact files (`*.h5`, `*.onnx`, `*.npz`, the calibration temperature file); every plausible secret-credential filename (`kaggle.json`, `.env`, `access_token`); and, explicitly, `CLAUDE.md`/`.claude/` (AI-assistant-tool configuration files that should never be part of the public portfolio repository). It deliberately does **not** ignore `results/`, since that folder is the small, safe, evidentiary artifact this whole project's credibility depends on. |
| `.github/workflows/ci.yml` | Covered in §8.2. |

---

## 10. Data Storage: How This Project Handles "The Database Question"

**There is no database in this project** — no SQL, no NoSQL, no ORM, no schema migrations. This is a deliberate, correct architectural choice, not a gap: nothing in this system needs transactional writes, relational queries, or concurrent multi-user record management. Here is what actually plays each of the roles a database would normally play:

| Traditional DB concept | What this project uses instead |
|---|---|
| **Tables/records for structured data** | Flat JSON files, one per pipeline stage, under `results/metrics/` (e.g. `evaluation.json`, `leakage.json`). Each is written once per pipeline run by exactly one Python module, and read by `fill_readme_results.py`. |
| **A schema** | Implicit, in the sense that each JSON file's key structure is defined once, in the Python function that constructs it (e.g. `evaluate.compute_metrics()`'s returned dictionary *is* the schema for `evaluation.json`). There is no formal schema-validation layer (no Pydantic model, no JSON Schema file) — this is a real, acknowledged simplification appropriate for a single-writer, single-reader, offline pipeline, not something that would scale to a multi-service production system without adding one. |
| **Binary object storage** (images, model weights) | The local filesystem (`models/`, `data/`) for local runs, and the **Hugging Face Hub** model repository for anything the deployed apps need — Hub acts here exactly like an S3 bucket or Docker registry: versioned, content-addressed binary artifact storage, reachable over a simple download API (`hf_hub_download`). |
| **An append-only event log** | `results/metrics/drift_log.jsonl` — literally a text file where `monitoring.detect()` appends one JSON object per line, per run. This is the JSON Lines format, a common, simple pattern for exactly this kind of "log of events over time" use case, when a full time-series database would be overkill. |
| **Caching layer** | Module-level Python singletons (see §12) — the model, once loaded, stays in a process's memory for the lifetime of that process. There's no Redis, no separate cache tier. |

**Why this is the right choice here, not a missing feature:** the entire "write path" of this system is a single offline batch pipeline run by one person on one machine at a time — there's no concurrent-write problem to solve, no need for ACID transactions, and no relational structure between the different metrics files (they don't reference each other by foreign key; they're independent reports). Introducing a real database would add operational complexity (a server to run, connection pooling, migrations) with zero corresponding benefit for this workload.

---

## 11. Authentication & Secrets: How This Project Handles "The Auth Question"

**There is no end-user authentication anywhere in this system.** Nobody logs in; there are no user accounts, no sessions, no JWTs, no OAuth flows, and no role-based access control for the live demo or the FastAPI service — both are open, unauthenticated, publicly-callable endpoints, appropriate for a public research demo with no user-specific data to protect.

The credential/auth-shaped things that **do** exist in this project are all developer/deployment-time, not end-user-facing:

- **Kaggle API credentials** (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables) — required only to run `scripts/download_data.sh`. These are read by the third-party `kaggle` CLI tool, never touched by this project's own code, and are explicitly excluded from git via `.gitignore`.
- **Hugging Face write token** — used only by the (manual, one-time) deployment step of uploading the model artifacts to the Hub and pushing the Space's files. Passed via the `HF_TOKEN` environment variable at deploy time, never written to a file, never committed.
- **GitHub Actions' implicit `GITHUB_TOKEN`** — used transparently by the `actions/checkout` step in CI; this project's workflow doesn't need any custom secret beyond that.

**If this project ever needed real end-user authentication** (for example, if a future version tracked per-user upload history), the natural extension points already exist: FastAPI has first-class support for dependency-injected auth (an `Authorization` header parsed via a `Depends(...)` function that validates a token before the endpoint body runs), and Gradio supports optional built-in login screens. Neither is implemented here because neither is needed by the actual feature set.

---

## 12. State Management: How This Project Handles "The State Question"

There is no React/Redux/Zustand-style client-side state management, because there is no custom frontend JavaScript. The state that *does* exist in this system falls into three categories:

1. **Configuration state — `src/btc/config.py`.** A single, static module of constants, imported everywhere. This is the closest analog to a global app-config store: one source of truth, read-only at runtime, no mutation.

2. **Loaded-model state — module-level singletons.** Both serving apps load the (large, slow-to-construct) model **once** and keep it alive for the life of the process, rather than reloading it on every request:
   - In `space/app.py`, this is literally top-level module code (`_sess = ort.InferenceSession(...)`, `_tf_model = _build_model()`) that runs exactly once, the moment the Python process starts (i.e., when Gradio's server boots).
   - In `src/btc/serve/app.py`, this is a mutable module-level dictionary, `_state = {"model": None, ...}`, populated at startup by the app's `lifespan` handler, or, as a defensive fallback, inside `_infer()` itself checking `if _state["model"] is None: _load()` — both under a `threading.Lock` (`_model_lock`), since inference now runs in worker threads and Keras models aren't documented as safe for concurrent `predict()` calls. This pattern (a plain dict as a mutable singleton) is a common, simple way to hold server-lifetime state in a single-process Python web app without needing a dedicated class or dependency-injection framework.

   **Why this matters, technically:** re-building and re-loading an 11.7-million-parameter neural network on every single incoming HTTP request would add seconds of latency to every prediction. Loading once and reusing the in-memory object across requests is what makes the measured ~33ms ONNX inference latency achievable — the expensive setup cost is paid exactly once, at startup, not per-request.

3. **Persisted-across-runs state — the `results/` and `models/` directories.** This is the system's only form of state that survives a process restart: the trained weights, the fitted calibration temperature, and every measured metric. There's no in-memory cache invalidation problem to solve here, because each file is written exactly once, atomically, by a single pipeline stage's `main()` function completing — there's no concurrent writer to race against.

---

## 13. Every Important Function, Explained

This section catalogs every non-trivial function across the codebase with purpose, parameters, return value, and complexity, organized by module. (Functions already covered in full narrative depth in §5 are summarized more briefly here to avoid duplication — see the cross-referenced section for the full story.)

| Function | Module | Purpose | Params → Returns | Notes |
|---|---|---|---|---|
| `ensure_dirs()` | `config.py` | Create all artifact directories if missing | `()` → `None` | Idempotent; O(1) (4 directories) |
| `_scan_split(split_dir)` | `data.py` | Build filepath/label dataframe from class-subfolders | `Path` → `DataFrame` | O(n) in number of files |
| `build_dataframes()` | `data.py` | Get train+test dataframes | `()` → `(DataFrame, DataFrame)` | See §5.2 |
| `make_generators(seed)` | `data.py` | Build parity-matched Keras generators | `int` → `(generator, generator)` | Includes drift-warning + parity-check logic, see §5.2 |
| `load_test_arrays()` | `data.py` | Materialize full test set into memory | `()` → `(ndarray, ndarray, list[str])` | O(n) images loaded and decoded; memory-bound |
| `build_model(weights)` | `model.py` | Construct the EfficientNetB3 architecture | `str\|None` → `keras.Model` | See §5.3 |
| `load_trained_model(weights_path)` | `model.py` | Build + load fine-tuned weights | `Path\|None` → `keras.Model` | Raises if weights missing |
| `find_last_conv_layer_name(model)` | `model.py` | Locate Grad-CAM's target layer | `keras.Model` → `str` | O(layers in backbone), reverse scan |
| `recalibrate_bn(model, gen, steps)` | `train.py` | Fix BatchNorm running stats post-training | `(Model, generator, int)` → `None` | Side effect: mutates model in place; see §5.4, §14 |
| `main()` | `train.py` | Full training run + recalibration + history save | `()` → `None` | Side effects: writes 2 files to disk, trains for up to 20 epochs |
| `predict_test(model)` | `evaluate.py` | Run model over full test set | `Model\|None` → `(ndarray, ndarray)` | O(n) forward passes |
| `compute_metrics(y_true, y_prob)` | `evaluate.py` | Compute the full metrics dictionary | `(ndarray, ndarray)` → `dict` | Dominated by O(n log n) sort inside sklearn's ROC/PR routines |
| `_content_hash(path)` | `leakage.py` | Exact-duplicate fingerprint | `str` → `str` (MD5 hex) | O(1) per image (fixed 64×64 resize) |
| `_phash(path)` | `leakage.py` | Perceptual-similarity fingerprint | `str` → `ImageHash` | O(1) per image |
| `analyse(eval_clean_subset)` | `leakage.py` | Full leakage audit | `bool` → `(dict, ndarray)` | **O(n_test × n_train)** — every test image compared against every train image; the dominant cost of the whole pipeline; see §14 for the complexity trade-off discussion |
| `evaluate_external()` | `external_validation.py` | OOD accuracy via binary collapse | `()` → `dict` | O(n_external) |
| `expected_calibration_error(probs, y, bins)` | `calibration.py` | Compute ECE | `(ndarray, ndarray, int)` → `(float, list)` | O(n) plus O(bins) bucketing |
| `_fit_temperature(probs, y)` | `calibration.py` | Grid-search the best temperature scalar | `(ndarray, ndarray)` → `float` | O(n × grid size); grid size fixed at 100+100 |
| `apply_temperature(probs, T)` | `calibration.py` | Rescale a probability distribution | `(ndarray, float)` → `ndarray` | O(n) |
| `_grad_model(model, layer_name)` | `explain.py` | Build the Grad-CAM auxiliary model | `(Model, str)` → `(Model, list)` | See §5.9 for the bug this fixes |
| `gradcam_heatmap(x, model, grad_model, class_idx)` | `explain.py` | Compute one Grad-CAM heatmap | `(ndarray, Model, tuple, int\|None)` → `(ndarray, int)` | O(1) — single forward + backward pass |
| `overlay(x, heatmap, alpha)` | `explain.py` | Blend heatmap onto image | `(ndarray, ndarray, float)` → `ndarray` | O(pixels) |
| `export(model)` | `export_onnx.py` | Convert Keras → ONNX | `Model\|None` → `None` | One-time graph trace/convert |
| `_bench(fn, x)` | `export_onnx.py` | Timing harness (warmup + N runs) | `(callable, ndarray)` → `dict` | O(N) calls to `fn` |
| `_embedder(model)` | `monitoring.py` | Get the 1536-d embedding sub-model | `Model\|None` → `(Model, Model)` | See §5.11, same nested-model fix as Grad-CAM |
| `build_reference(limit)` | `monitoring.py` | Fit PCA + histograms on training embeddings | `int\|None` → `None` | O(n × PCA fit cost); PCA fit is roughly O(n·d²) for d=1536 reduced internally via SVD on the sample |
| `_psi(ref_counts, new_vals, edges)` | `monitoring.py` | Population Stability Index | `(ndarray, ndarray, ndarray)` → `float` | O(bins) |
| `detect(image_dir, tag)` | `monitoring.py` | Score a new batch for drift | `(str, str)` → `dict` | O(n_new) |
| `_corrupt(x, kind, sev)` | `robustness.py` | Apply one corruption at one severity | `(ndarray, str, int)` → `ndarray` | O(pixels); JPEG path additionally does real encode/decode |
| `evaluate_robustness(limit)` | `robustness.py` | Full 6×5 corruption sweep + accuracy | `int` → `dict` | O(30 × n_subset) forward passes |
| `_augmentations(x)` | `uncertainty.py` | Generate 6 TTA views | `ndarray` → `list[ndarray]` | O(1), fixed 6 views |
| `tta_predict(x, predict_fn, T)` | `uncertainty.py` | Averaged prediction + entropy/std | `(ndarray, callable, float)` → `dict` | O(6) forward passes (batched as one call) |
| `mean_dataset_uncertainty(model, limit)` | `uncertainty.py` | Correct-vs-wrong entropy comparison | `(Model\|None, int)` → `dict` | O(n_subset × 6) forward passes |
| `_load()` | `serve/app.py` | Populate the FastAPI singleton state | `()` → `None` | Side effect: mutates module-level `_state` dict |
| `_apply_temperature(probs)` | `serve/app.py`, `space/app.py` | Apply the fitted calibration scalar | `ndarray` → `ndarray` | O(classes) — duplicated logic, see §15 |
| `predict(file)` | `serve/app.py` | The `/predict` FastAPI route handler | `UploadFile` → `JSONResponse` | See §16 for the full request/response contract |
| `predict(image)` | `space/app.py` | The Gradio button's click handler | `ndarray\|None` → `(dict, ndarray, str)` | See §6.1 for the full step-by-step trace |

---

## 14. Deep Technical Concepts (Beginner → Advanced)

Every non-obvious technical concept this codebase actually depends on, explained twice: first for someone with no programming background, then precisely.

### Convolutional Neural Network (CNN) / Transfer Learning

**Simple:** Imagine teaching someone to recognize brain tumors by first having them look at millions of *completely unrelated* photos (cats, cars, furniture) until they get really good at noticing edges, shapes, and textures in general — and only *then* showing them a much smaller number of actual brain MRIs to specialize that general skill.
**Technical:** EfficientNetB3 is a convolutional neural network — a stack of learned filters that slide over an image detecting increasingly abstract patterns (edges → textures → shapes → object parts) layer by layer. It was originally trained ("pretrained") on ImageNet, a 14-million-image, 1,000-category natural-image dataset, which gave it strong, general-purpose visual features. This project loads those pretrained weights (`weights="imagenet"`) and then continues training ("fine-tuning") the entire network on the much smaller, domain-specific brain MRI dataset — transfer learning, which is why a model can reach high accuracy with only ~5,700 training images instead of needing millions.

### Batch Normalization (and the exact bug this project hit)

**Simple:** Imagine a relay race where each runner needs the baton handed to them at roughly the same height and speed every time to run their best; BatchNormalization is a step inside the network that keeps the "signal" flowing between layers at a roughly consistent scale, which makes training faster and more stable.
**Technical:** a `BatchNormalization` layer normalizes its input using the *current batch's* mean and variance during training, while also maintaining a separate, slowly-updated *running average* of those statistics (controlled by a `momentum` parameter, here `0.99`) — and at inference time, it switches to using that running average instead of a (nonexistent, single-image) batch statistic. **The bug:** on this project's development hardware (Apple Silicon, using the `tensorflow-metal` GPU backend), combined with the model only ever seeing about half the training data per epoch (the `steps_per_epoch` quirk) and a very high momentum, the running average never properly converged. The result: inference-mode predictions collapsed to ~53% accuracy (worse than the model's own training-mode accuracy of ~78%), even though ROC-AUC — a metric based on the model's internal *ranking* of predictions rather than the raw calibration of the final probability — stayed around 0.97, proving the learned features were fine and only the BatchNorm statistics were broken. **The fix** (`recalibrate_bn` in `train.py`): lower the momentum to `0.9` and run 350 extra forward passes in training mode with no gradient updates, letting the running statistics re-converge on real data.

### Softmax and Temperature Scaling

**Simple:** Softmax turns a set of raw scores into percentages that add up to 100%, like turning "candidate A got 8 votes, candidate B got 2 votes" into "80% vs 20%." Temperature scaling is a dial that can make those percentages more or less extreme without changing which candidate actually won.
**Technical:** `softmax(z)_i = exp(z_i) / Σ_j exp(z_j)`, applied to a model's raw output "logits" `z` to produce a valid probability distribution. Temperature scaling divides every logit by a learned scalar `T` before applying softmax: `softmax(z / T)`. If `T > 1`, the distribution becomes flatter (less confident); if `T < 1`, sharper (more confident); `T = 1` is unchanged. Crucially, because every logit is divided by the same scalar, the *argmax* (which class wins) never changes — only the reported confidence does. On the CPU (deployed) computation this project fits `T ≈ 0.561`: the raw model is markedly *under*-confident (its softmax is too flat), so a `T < 1` sharpens it, cutting ECE from 0.196 to 0.073. A stricter check — fit `T` on a random half of the test set, score on the other half, 5 seeds — gives 0.074 ± 0.008 with T = 0.562 ± 0.006, so fitting and scoring on the same split isn't materially optimistic here. (The live Space previously applied the earlier Metal-fitted T = 0.819, which gives ECE 0.127 on CPU outputs; the CPU-fitted T = 0.561 has since been uploaded to the Hub, and the live Space confirms it in every response.)

### Expected Calibration Error (ECE)

**Simple:** if a weather forecaster says "70% chance of rain" on 100 different days, a well-calibrated forecaster would see rain on close to 70 of those days — ECE measures how far off from that ideal a model's stated confidence actually is, on average.
**Technical:** bucket predictions into confidence bins, and within each bin compute `|average_confidence − actual_accuracy|`; ECE is the weighted (by bin population) average of that gap across all bins. Implemented in `calibration.expected_calibration_error`.

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**Simple:** it's a way of asking the model "point to the part of the image that made you decide this," and getting back a heatmap answer.
**Technical:** covered fully in §5.9 — computes the gradient of the target class's output score with respect to the last convolutional layer's feature map, uses the *global-average-pooled gradients* as per-channel importance weights, computes a weighted sum of the feature map, and applies ReLU to keep only positively-contributing regions.

### ONNX and ONNX Runtime

**Simple:** ONNX is like a universal translator for trained neural networks — it lets a model trained in one framework (TensorFlow) run efficiently in a different, lighter-weight execution engine (ONNX Runtime) without needing that original framework installed.
**Technical:** ONNX (Open Neural Network Exchange) is a standardized, framework-agnostic graph representation for neural networks — a serialized description of every operation (convolution, matrix multiply, activation, etc.) and how they're wired together. `tf2onnx` traces a Keras model's computation graph and exports it into this format. `onnxruntime.InferenceSession` then loads that graph and executes it using its own, separately-optimized execution engine, which applies graph-level optimizations (fusing consecutive operations, eliminating dead computation) that this project measured to deliver roughly a 3.5× CPU speedup over eager TensorFlow execution.

### Test-Time Augmentation (TTA)

**Simple:** ask the model the same question several slightly different ways (flipped, rotated a little, brightened a little) and see how consistent its answers are — consistent answers mean it's genuinely confident; inconsistent answers are a red flag.
**Technical:** covered fully in §5.13.

### Perceptual Hashing / Population Stability Index / Mahalanobis Distance

Covered in §5.6 (perceptual hashing) and §5.11 (PSI, Mahalanobis) with full technical detail.

### REST APIs, async/await, and FastAPI

**Simple:** a REST API is a standardized way for one program to ask another program to do something over the internet, using URLs and standard verbs like GET ("give me data") and POST ("here's data, do something with it"), getting back a structured answer (this project uses JSON).
**Technical:** `src/btc/serve/app.py` exposes two REST endpoints using **FastAPI**, a modern Python web framework built on top of the ASGI standard (Asynchronous Server Gateway Interface). The `/predict` route handler is declared `async def predict(...)`, meaning FastAPI can, in principle, interleave handling of multiple concurrent requests during any `await`-ed I/O (like `await file.read()`) rather than blocking the entire process on one request at a time. The catch: model inference is *synchronous*, CPU-bound TensorFlow work. An earlier version called it directly inside the `async` handler, which froze the whole event loop for each request — measured on the development machine, with 16 concurrent `/predict` calls in flight, `/health` took up to **3.3 s** to answer. That's close to the Docker healthcheck's 5 s timeout, past which an orchestrator restarts a container just for being busy. The handler now hands inference to a worker thread with `await run_in_threadpool(_infer, x)`, and a `threading.Lock` keeps model access serialised exactly as before. The event loop stays free: `/health` now peaks at 57 ms under the same load. Inference itself is still serialised behind the lock, so real horizontal scale means more replicas, not more threads (see §20).

### Docker Containers

**Simple:** a Docker container is a way to package an application together with its *entire* environment (exact OS libraries, exact Python version, exact dependency versions) into one portable unit, so it runs identically anywhere Docker itself is installed, instead of "works on my machine" uncertainty.
**Technical:** this project's `Dockerfile` builds from `python:3.11-slim`, installs OS-level shared libraries OpenCV needs at import time (`libgl1`, `libglib2.0-0` — without these, `import cv2` crashes with a missing-shared-library error on a minimal base image), installs the pinned Python dependencies, copies in the application code and the trained model artifacts, and defines a `HEALTHCHECK` so an orchestrator (Docker Compose, Kubernetes, etc.) can automatically detect if the service becomes unresponsive. Note (stated honestly, not glossed over): this Dockerfile has been written to be correct and was reasoned through carefully, but was **not actually build-tested** on the development machine, because Docker itself was not installed there — this is documented plainly in the project's README rather than silently claimed as "verified."

### Continuous Integration (CI)

**Simple:** every time code is pushed to the shared repository, a robot automatically checks out the new code on a fresh machine and runs the test suite, catching mistakes before they can spread — like a spell-checker that runs itself every time you save a document.
**Technical:** covered in §8.2 — a GitHub Actions workflow triggered on `push`/`pull_request`, running a fast (no-TensorFlow) subset of the codebase's tests, which has been confirmed to actually pass (`conclusion: success`) via `gh run watch`.

### Concepts explicitly **not** present in this project (stated for completeness, since generic documentation templates often expect them)

This project has **no** Server-Side Rendering, Client-Side Rendering distinction, hydration, streaming responses, React reconciliation, JWT sessions, WebSocket connections it manages directly (Gradio uses one internally, but this project's own code never touches it), vector databases, embeddings-for-retrieval/RAG, LLMs, or message queues. None of these appear anywhere in the actual source code, so none are documented as if they did.

---

## 15. Design Decisions and Trade-offs (the WHY)

| Decision | Alternative(s) considered | Why this choice | Trade-off accepted |
|---|---|---|---|
| Keep `legacy/` notebooks byte-for-byte frozen | Update/"clean up" the original notebooks | The notebooks are the artifact behind a real, published, DOI-referenced paper; silently altering them would misrepresent what was actually peer-reviewed | Some duplicated/superseded logic must live on unchanged in `legacy/` even though `src/btc/` has since improved on it |
| ONNX Runtime for the live demo's inference path | Serve directly with Keras/TensorFlow everywhere | Measured ~2.4× lower CPU latency (79 ms → 34 ms, both on CPU), smaller dependency footprint, better suited to a free-tier CPU host | Grad-CAM still needs gradients, which ONNX Runtime doesn't support, so a *second*, separate Keras copy of the model must also be loaded — see next row |
| Load two copies of the model in `space/app.py` (ONNX + Keras) | Use only one format everywhere | ONNX for speed on the hot path (every prediction), Keras only for the comparatively rarer Grad-CAM computation, which genuinely requires TensorFlow's automatic differentiation | Roughly double the memory footprint and startup download time in the deployed Space |
| Gradio (not a custom React/Next.js frontend) for the live demo | Hand-build a JS frontend + separate backend | Zero frontend code to write or maintain, built-in file upload/rendering components, deploys natively and for free on Hugging Face Spaces | Less UI customizability than a hand-built frontend; the whole visual design is constrained to what Gradio's component library supports |
| `space/app.py` duplicates model-definition code from `src/btc/model.py` instead of importing it | Make the Space depend on the whole `src/btc` package | Hugging Face Spaces only deploys the `space/` folder's contents; keeping it self-contained avoids needing to package and publish `src/btc` as an installable dependency just for the Space to use it | Code duplication — a future architecture change to `model.py` must be manually mirrored into `space/app.py`, a real maintenance cost, accepted because the two rarely change and the alternative (extra packaging/publishing complexity) was judged worse for a small solo-maintained project |
| No database | Postgres/SQLite for storing metrics history | Nothing in this system needs multi-writer concurrency, relational queries, or transactions — it's a single offline batch pipeline with independent output files | No query language over historical results; comparing metrics across multiple past runs would require manually diffing JSON files (acceptable at this project's current scale) |
| BatchNorm recalibration instead of retraining from scratch on different hardware | Just retrain on a CUDA machine where the bug doesn't appear | Recalibration is a legitimate, standard, fast (350 forward passes, no gradient steps) fix that isolates and repairs exactly the broken component without touching the learned weights or needing different hardware | Requires understanding *why* the bug happened well enough to trust that a narrower fix (rather than full retraining) is actually correct — validated here by re-measuring full-test accuracy after the fix (0.53 → 0.939 on Metal; 91.25% on CPU, the deployed computation — §4b) rather than just assuming it worked |
| Report the dataset-drift finding explicitly (paper's 5,712/1,311 split vs. today's actual 5,600/1,600) instead of silently reproducing "close enough" | Ignore the discrepancy since it's not the author's fault | Silently ignoring it would make the reproduced 91.25% number look directly comparable to the paper's 99.844% when it isn't (different test set); flagging it lets a reader correctly attribute part of the gap to genuine leakage-removal and part to a moved dataset, rather than overstating the leakage finding | The finding is slightly more complicated to explain than a single clean number, but it's the honest description of what was actually measured |
| Perceptual hash (not exact patient IDs) for leakage detection | Try to recover true patient IDs from the source datasets | The Kaggle compilation genuinely does not retain patient/scan metadata — there is no ground truth to recover | The leakage rate is explicitly documented as a *lower bound* / proxy measurement, not an exact patient-level figure — a real, acknowledged limitation of the method rather than a hidden one |
| Test-Time Augmentation over Monte Carlo Dropout for uncertainty | MC-Dropout (running the model multiple times with dropout layers kept active) | TTA is architecture-agnostic and composes cleanly with the ONNX Runtime serving path already chosen for speed; MC-Dropout requires special inference-time handling of dropout layers that most inference runtimes (including ONNX Runtime by default) disable at inference for good reason | TTA captures a different, narrower notion of uncertainty (sensitivity to small input perturbations) than MC-Dropout (which approximates a Bayesian posterior over weights) — a real, acknowledged limitation of what this specific uncertainty signal actually measures |
| Docker image written but not build-verified | Build and test it locally before claiming it works | Docker was not installed on the available development machine | Explicitly documented as "build-ready, unverified" in the README rather than claimed as a tested, working deployment artifact |

---

## 16. API Reference

### `POST /predict` (FastAPI, `src/btc/serve/app.py`)

| | |
|---|---|
| **Method** | `POST` |
| **Route** | `/predict` |
| **Request body** | `multipart/form-data`, single field `file` (an image: JPEG/PNG/etc.) |
| **Validation** | Upload size is capped at `MAX_UPLOAD_BYTES` (10 MB) — the body is read with an explicit byte limit rather than trusting the client's `Content-Length`, so an oversized upload can't exhaust server memory. The bytes are then opened with PIL inside a `try/except`; any failure (corrupt file, non-image content) becomes a `400 Bad Request` with a descriptive message — never a raw stack trace |
| **Processing** | Resize to 256×256 RGB → run Keras model → apply calibration temperature (if loaded) → compute Grad-CAM for the predicted class → base64-encode the overlay PNG |
| **Response** | `200 OK`, JSON body (see schema below) |
| **Possible errors** | `400` — invalid/unreadable image file; `413` — upload exceeds the 10 MB limit |
| **Files involved** | `src/btc/serve/app.py` (route handler), `src/btc/model.py` (`load_trained_model`), `src/btc/explain.py` (`gradcam_heatmap`, `overlay`) |

**Example request:**
```bash
curl -F "file=@brain_scan.jpg" http://localhost:8000/predict
```

**Example response:**
```json
{
  "class": "meningioma",
  "class_index": 1,
  "calibrated_confidence": 0.921,
  "raw_confidence": 0.861,
  "is_calibrated": true,
  "probabilities": {
    "glioma": 0.019, "meningioma": 0.921, "notumor": 0.028, "pituitary": 0.031
  },
  "gradcam_png_base64": "iVBORw0KGgoAAAANSUhEUgA...",
  "inference_ms": 66.2,
  "gradcam_ms": 589.7,
  "latency_ms": 655.9
}
```

**On the timing fields:** these are reported separately because Grad-CAM dominates a `/predict` call — before Grad-CAM was compiled, it measured roughly **40 ms of inference vs. ~380 ms of Grad-CAM** per request on the development machine, so a `latency_ms` covering only the forward pass understated the endpoint's real cost about 10×. With the compiled Grad-CAM (§5.9), a warm request now measures a median **74 ms** wall-clock (41 ms inference + 30 ms Grad-CAM and PNG encoding), and a burst of 16 concurrent requests completes in 1.2 s instead of 5.9 s.

**Execution flow:** `browser/curl → uvicorn (ASGI server) → FastAPI routing → predict() handler → PIL decode → model.predict() (Keras) → temperature scaling → Grad-CAM (2nd forward+backward pass) → JSON serialization → HTTP response`

---

### `GET /health` (FastAPI)

| | |
|---|---|
| **Method** | `GET` |
| **Route** | `/health` |
| **Request** | none |
| **Response** | `{"status": "ok", "model_loaded": true, "calibrated": true}` |
| **Used by** | The `Dockerfile`'s `HEALTHCHECK` directive, polled every 30 seconds |

---

### `predict(image)` — the Gradio Space's de facto API

Not a hand-written REST route — Gradio auto-generates its own internal API from this Python function's signature (this is what the earlier gradio-4.44 `get_api_info()` bug, fixed in §6.1, was actually crashing inside of). Functionally equivalent request/response shape to `/predict` above, but returns three separate values that Gradio renders as UI components (a probability bar chart, an image, and a Markdown block) rather than one JSON object. Verified end-to-end against the actual deployed Space via the `gradio_client` Python library, receiving a correct, real prediction with Grad-CAM, uncertainty, and latency all populated.

---

## 17. Execution Traces: Tracing Every Major Action Through the System

### Trace A — A visitor uses the live demo

```text
User opens huggingface.co/spaces/Udit013/brain-tumor-mri-classifier
 ↓
Browser loads the Gradio-generated HTML/JS page
 ↓
User drags an MRI image into the upload box
 ↓
User clicks "Classify"
 ↓
Browser sends the image to the Gradio server (space/app.py, already running,
model pre-loaded since container startup)
 ↓
predict(image) function executes:
   1. PIL resize/convert (256x256 RGB)
   2. ONNX Runtime forward pass → raw probabilities
   3. Temperature scaling → calibrated probabilities
   4. 6x Test-Time-Augmentation views → ONNX batch call → entropy + std
   5. Keras model + GradientTape → Grad-CAM heatmap → OpenCV overlay blend
   6. Markdown string assembled (class, confidence, entropy, latency, disclaimer)
 ↓
Three return values sent back to the browser
 ↓
Gradio renders: probability bar chart, Grad-CAM image, Markdown text
 ↓
User sees the result on screen
```

### Trace B — A developer runs the full reproduction pipeline

```text
Developer runs: bash scripts/reproduce.sh
 ↓
Step 1: btc.train.main()
   data.make_generators() → model.build_model() → model.fit() (20 epochs)
   → recalibrate_bn() → weights saved to models/EfficientNetB3_model_weights.h5
   → results/metrics/train_history.json written
 ↓
Step 2: btc.evaluate.main()
   model.load_trained_model() → data.load_test_arrays() → model.predict()
   → compute_metrics() → results/metrics/evaluation.json
   → 3 PNG figures saved to results/figures/
 ↓
Step 3: btc.leakage.main() (--eval flag)
   Hash every train + test image → find near/exact duplicates
   → re-use Step 2's predictions → split leaked vs clean → separate accuracy
   → results/metrics/leakage.json + histogram PNG
 ↓
Step 4: btc.external_validation.main()
   Load external Kaggle dataset → run model → binary-collapse comparison
   → results/metrics/external_validation.json
 ↓
Step 5: btc.calibration.main()
   Fit temperature on Step 2's predictions → compute ECE before/after
   → models/temperature.json + results/metrics/calibration.json + reliability PNG
 ↓
Step 6: btc.explain.main()
   Find correct+wrong example per class → Grad-CAM each → montage PNG
 ↓
Step 7: btc.export_onnx.main()
   tf2onnx.convert → models/efficientnetb3.onnx
   → benchmark Keras vs ONNX-CPU vs ONNX-CoreML → results/metrics/latency.json
 ↓
Step 8: btc.robustness.main()
   6 corruptions x 5 severities x balanced subset → results/metrics/robustness.json
   + robustness_curves.png
 ↓
Step 9: btc.uncertainty.main()
   TTA over balanced subset → correct-vs-wrong entropy → results/metrics/uncertainty.json
 ↓
Step 10: btc.monitoring.build_reference()
   Embed training sample → PCA fit → models/drift_reference.npz
 ↓
Developer runs: python scripts/fill_readme_results.py
   Reads all results/metrics/*.json → replaces marker blocks in README.md
   with real, measured numbers
```

### Trace C — A client calls the FastAPI endpoint directly

```text
curl -F "file=@scan.jpg" http://localhost:8000/predict
 ↓
uvicorn (ASGI server) receives the HTTP request
 ↓
FastAPI routes it to predict(file: UploadFile)
 ↓
if _state["model"] is None: _load()   (only true on the very first request
                                        if startup event somehow didn't fire)
 ↓
await file.read() → PIL.Image.open() → .convert("RGB").resize((256,256))
 ↓
img_to_array() → model.predict() (Keras) → raw_probs
 ↓
_apply_temperature(raw_probs) → cal_probs
 ↓
gradcam_heatmap() + overlay() → PNG bytes → base64 encode
 ↓
JSONResponse({...}) returned
 ↓
curl prints the JSON to the terminal
```

### Trace D — Code is pushed to GitHub (CI trace)

```text
Developer runs: git push origin main
 ↓
GitHub receives the push, triggers .github/workflows/ci.yml
 ↓
GitHub Actions spins up a fresh ubuntu-latest runner
 ↓
actions/checkout@v4 → clones the repo at that commit
 ↓
actions/setup-python@v5 → installs Python 3.11
 ↓
pip install numpy pillow opencv-python-headless imagehash pytest
pip install -e . --no-deps   (installs btc package, no TensorFlow)
 ↓
python -m py_compile src/btc/*.py src/btc/serve/*.py space/app.py
   (syntax-checks every file, including TF-dependent ones, without importing them)
 ↓
PYTHONPATH=src pytest tests/ -q
   (32 tests run: 27 covering calibration math, corruption functions, TTA
   aggregation logic, and hash equality — all without TensorFlow or real data)
 ↓
Workflow reports success/failure back to GitHub
 ↓
README's CI badge reflects the latest run's status
```

---

## 18. Glossary

| Term | Simple definition | Technical definition | Where it appears in this project |
|---|---|---|---|
| **CNN** | A type of AI good at "looking" at images | Convolutional Neural Network: a network using learned sliding filters (convolutions) to detect spatial patterns | The whole model, `model.py` |
| **Transfer learning** | Reusing knowledge learned from one task on a new, related task | Initializing a network's weights from a model pretrained on a large dataset (ImageNet), then continuing training on a smaller, domain-specific dataset | `build_model(weights="imagenet")` |
| **BatchNorm** | Keeps signals flowing through the network at a consistent scale | A layer that normalizes activations using batch statistics during training and a running average at inference | `model.py`, the bug in §14 |
| **Softmax** | Turns raw scores into percentages that sum to 100% | `exp(z_i) / Σ exp(z_j)` applied to a vector of logits | Final layer of the model |
| **Temperature scaling** | A dial that adjusts how confident predictions "sound" without changing the winner | Dividing logits by a learned scalar `T` before softmax | `calibration.py` |
| **ECE** | How far off a model's stated confidence is from its real accuracy, on average | Weighted average of `|confidence − accuracy|` across confidence bins | `calibration.py` |
| **Grad-CAM** | A heatmap showing what part of an image the model focused on | Gradient-weighted Class Activation Mapping — weights the last conv layer's feature map by class-score gradients | `explain.py` |
| **ONNX** | A universal file format for trained neural networks | Open Neural Network Exchange — a framework-agnostic serialized computation graph | `export_onnx.py`, `models/efficientnetb3.onnx` |
| **ONNX Runtime** | The fast engine that runs ONNX files | A separate, optimized inference engine that executes ONNX graphs, independent of the original training framework | Both serving apps |
| **TTA** | Asking the model the same question a few slightly different ways to gauge confidence | Test-Time Augmentation — averaging predictions across several label-preserving input transformations at inference time | `uncertainty.py` |
| **Perceptual hash (phash)** | A fingerprint that's similar for visually similar images | A hash function designed so visually near-identical images produce hash values with small Hamming distance | `leakage.py` |
| **PSI** | A number measuring how much a distribution has shifted | Population Stability Index — a weighted log-ratio comparison between a reference and a new distribution's binned proportions | `monitoring.py` |
| **PCA** | A way to compress data down to its most important few dimensions | Principal Component Analysis — finds the orthogonal directions of maximum variance in high-dimensional data | `monitoring.py` |
| **Mahalanobis distance** | A "how far away" measurement that accounts for correlation between dimensions | `sqrt((x−μ)ᵀ Σ⁻¹ (x−μ))` | `monitoring.py` |
| **REST API** | A standard way for programs to talk to each other over the internet | An HTTP-based interface using standard verbs (GET/POST/etc.) and typically JSON payloads | `serve/app.py` |
| **ASGI / async** | Lets a server potentially juggle many requests without one blocking all the others | Asynchronous Server Gateway Interface; Python `async`/`await` cooperative concurrency | FastAPI, `serve/app.py` |
| **Docker container** | A portable package containing an app and its entire environment | An isolated, reproducible OS-level environment defined by a `Dockerfile` | `Dockerfile` |
| **CI (Continuous Integration)** | An automatic robot that checks your code every time you push it | Automated build/test pipelines triggered on repository events | `.github/workflows/ci.yml` |
| **Singleton (in this context)** | Keeping one shared copy of something expensive (like a loaded model) instead of remaking it every time | A module-level object constructed once and reused for the life of the process | `_state` dict in `serve/app.py`; top-level vars in `space/app.py` |

---

## 19. Interview Question Bank

Organized by section, each with beginner / intermediate / advanced framing.

**On the overall architecture:**
- *Beginner:* "What does this app actually do?" — Takes a brain MRI image and predicts one of four categories, showing a confidence score and a visual explanation.
- *Intermediate:* "Why is there no database?" — See §10: nothing in this system needs relational structure or concurrent writes; flat JSON files plus the Hugging Face Hub cover every actual storage need here.
- *Advanced:* "Walk me through what happens between a user clicking 'Classify' and seeing a result, at the systems level." — Use Trace A in §17, and be ready to name every file involved.

**On the leakage audit (the project's signature finding):**
- *Beginner:* "Why can having 'too similar' training and test images be a problem?" — The model can partly memorize instead of learning to generalize, so its test score looks better than it would perform on a truly new patient.
- *Intermediate:* "How exactly did you measure the amount of leakage?" — Perceptual + exact hashing between every test image and every training image (§5.6); then, critically, re-measuring accuracy separately on the flagged-vs-clean subsets to prove it mattered, not just that duplicates existed.
- *Advanced:* "What's the actual computational complexity of your leakage audit, and would it scale?" — O(n_test × n_train) pairwise comparisons (§13); fine at ~5,600×1,600, would need an approximate-nearest-neighbor index (e.g. an LSH-based method) at a much larger scale.

**On the BatchNorm bug:**
- *Beginner:* "What was the bug, in plain terms?" — The model looked broken (53% accuracy) right after moving it to a real serving setup, even though it had trained well.
- *Intermediate:* "How did you know it was BatchNorm and not the model weights themselves?" — Comparing inference-mode vs training-mode accuracy on identical inputs (`scripts/diagnose_bn.py`) isolated it to whichever code path differs between those two modes — which, for a network with BatchNorm layers, is specifically whether batch statistics or running-average statistics are used.
- *Advanced:* "Why does lowering momentum and running extra forward passes fix a running-average convergence problem, mathematically?" — The running average update is `running = momentum × running + (1 − momentum) × batch_stat`; a lower momentum weights each new batch's contribution more heavily, so the running estimate moves toward the true population statistic faster, converging within fewer additional passes than the original `0.99` momentum would have allowed within the same training budget.

**On serving & deployment:**
- *Beginner:* "Where does the live demo actually run?" — On Hugging Face's free hosting (Spaces), which downloads the trained model from a separate Hugging Face model repository when it starts up.
- *Intermediate:* "Why two different serving apps (Gradio and FastAPI)?" — The Gradio app is the public, no-code, browser demo; the FastAPI app is a plain JSON API for programmatic/automated use, meant to be self-hosted (e.g. via the provided Dockerfile). They share the same model logic but serve different consumers.
- *Advanced:* "The FastAPI `/predict` route is declared `async`, but does that route achieve genuine request concurrency for the compute-heavy part?" — Not on its own. TensorFlow inference is synchronous and CPU-bound, and calling it inside an `async` handler blocks the event loop. That was a real bug here: with 16 requests in flight, `/health` took 3.3 s. The fix moves inference into a worker thread (`run_in_threadpool`) behind a lock, so the loop stays responsive (`/health` ≤57 ms under the same load). Inference is still serialised per process, so true throughput scaling means more replicas or a dedicated model server. (A good "trap" question — the honest answer includes the bug and the measurement.)

---

## 20. Common Follow-Up / "What If" Questions

- **"What if the Hugging Face Hub or Spaces goes down?"** The Gradio demo would become unreachable (it's a single hosted instance with no redundancy — appropriate for a free-tier portfolio demo, not a production SLA). The FastAPI service and the whole `scripts/reproduce.sh` pipeline are entirely independent of Hugging Face Hub uptime *except* for the one-time artifact download step, so local/self-hosted usage is unaffected once the model files are on disk.
- **"How would you scale this to handle heavy traffic?"** Inference already runs off the event loop (a worker thread behind a lock), so each process stays responsive under load — but that lock also means each process does one inference at a time. Beyond that: run multiple stateless replicas of the FastAPI service behind a load balancer (each replica independently loads its own copy of the model at startup — there's no shared mutable state to coordinate, since the model is read-only after loading), and consider batching concurrent requests together before a single inference call, since GPU/CPU inference is often more efficient per-image in larger batches.
- **"How would you add real user authentication if this needed it?"** FastAPI's `Depends()` dependency-injection system is the natural fit — a function that validates a bearer token (checked against a proper auth provider, not hand-rolled) would be added as a dependency on the `/predict` route, returning a `401 Unauthorized` if invalid, with essentially no change needed to the prediction logic itself.
- **"Why not just always report the paper's 99.8% number since that's what actually got published?"** Because it would misrepresent what the model can actually be trusted to do on new patients — the entire point of the production-extension work was proving that number needs an asterisk, quantifying that asterisk precisely, and reporting the honest, defensible number instead.
- **"How would you optimize inference latency further?"** Options actually available given this stack: batch multiple incoming requests together before running one ONNX Runtime call (amortizes fixed per-call overhead); quantize the ONNX model to INT8 (a real accuracy/speed trade-off that hasn't been explored here — an honest "haven't done this yet" rather than a guess); or explore ONNX Runtime's other CPU-specific execution providers/thread-tuning options beyond what was benchmarked.
- **"What would you need to make this an actual regulated medical device?"** Far beyond the current scope — this would require, at minimum: verified patient-level (not just image-similarity) data provenance, clinical validation studies on populations the model has never seen, formal regulatory clearance (e.g., FDA), rigorous subgroup/bias analysis across demographics and scanner types (explicitly listed as an *unknown* limitation in `MODEL_CARD.md` today), and a completely different reliability/monitoring bar than a portfolio project needs. The model card states plainly that this is a research/education artifact, not a medical device, precisely because none of that exists here.

---

## 21. Complete End-to-End Walkthrough

Start to finish, narrating every layer, for someone who has never seen this project before.

1. **Origin.** Years ago, four different neural network architectures were trained and benchmarked on a public Kaggle brain-MRI dataset, and the results were published at an IEEE conference. The best model, EfficientNetB3, reported 99.844% accuracy. Those four original notebooks live untouched in this repository's `legacy/` folder and at its root, checksummed to prove they've never been altered.

2. **The question.** Later, an investigation began: is that headline number actually trustworthy? To answer that, the entire recipe needed to be rebuilt as reproducible, testable Python code rather than a one-off notebook — that rebuild is the `src/btc/` package.

3. **Getting the data.** `scripts/download_data.sh` pulls two datasets from Kaggle: the original training/testing compilation, and a completely separate dataset used later to test genuine generalization.

4. **Building the model.** `src/btc/model.py` defines the exact EfficientNetB3-based architecture from the original paper. `src/btc/data.py` loads and preprocesses the images identically to how the original notebook did — including the critical, easy-to-miss detail of *not* rescaling pixel values, since EfficientNet normalizes internally.

5. **Training.** `src/btc/train.py` runs the actual 20-epoch training loop with the original recipe's exact callbacks and schedule. Along the way, a real production bug surfaced: on the development machine's hardware, the model's BatchNorm layers' internal running statistics never properly converged, causing predictions to collapse to near-random once the model was used for real (non-training-mode) inference — even though the model's learned features were still excellent underneath. This was diagnosed by directly comparing training-mode vs. inference-mode accuracy, and fixed with a standard recalibration technique: extra forward passes over training data with lowered BatchNorm momentum, touching only the running statistics, not the learned weights. This fix is now a permanent part of every training run.

6. **The audit.** With a working model in hand, `src/btc/evaluate.py` computes the full battery of honest metrics: per-class precision/recall/F1, confusion matrix, ROC/PR curves. Then `src/btc/leakage.py` runs the project's most important analysis: it fingerprints every training and test image (both exactly and perceptually) and discovers that 44.6% of test images have a near-duplicate sitting in the training set — and, critically, *proves this matters* by showing the model scores 97.2% on those leaked images but only 86.5% on genuinely novel ones. `src/btc/external_validation.py` then tests the model against a totally separate public dataset, honestly reporting a much lower (70.8%) out-of-distribution accuracy — exactly what real generalization looks like once the training-data "home turf advantage" is removed.

7. **Making it trustworthy.** `src/btc/calibration.py` measures how overconfident the raw model's probabilities are (Expected Calibration Error) and fits a single scalar correction (temperature scaling) that brings stated confidence back in line with real accuracy. `src/btc/explain.py` builds Grad-CAM heatmaps so any single prediction can be visually inspected for *why* the model decided what it decided. `src/btc/robustness.py` stress-tests the model against six kinds of realistic image corruption, uncovering a specific, non-obvious weakness: the model is essentially blind under any level of additive image noise, while staying above 70% under brightness, contrast, and rotation changes. `src/btc/uncertainty.py` adds a cheap, per-prediction confidence signal (Test-Time Augmentation); it is higher on wrong predictions than right ones, but only modestly, so it's a weak signal on its own.

8. **Making it fast.** `src/btc/export_onnx.py` converts the trained Keras model into the ONNX format and measures real latency improvements — inference speeds up roughly 2.4× on CPU compared to running the original Keras model on the same CPU (79 ms → 34 ms).

9. **Making it usable.** `src/btc/serve/app.py` wraps the model in a REST API (FastAPI) that anyone can call with `curl` or any HTTP client to get a class prediction, calibrated confidence, and a Grad-CAM image back as JSON. Separately, `space/app.py` builds a full, self-contained, no-code-required browser application using Gradio, which was deployed to Hugging Face Spaces — a genuinely live, public, free website anyone can visit and use, backed by the ONNX-exported model for speed.

10. **Making it maintainable.** `tests/test_core.py` locks down the pure-logic parts of the codebase (calibration math, corruption functions, uncertainty aggregation, hashing) with 26 automated tests, and `.github/workflows/ci.yml` runs them automatically on every code push — a real, verified-passing green checkmark, not an assumption.

11. **Making it honest, end to end.** `scripts/reproduce.sh` chains all ten pipeline stages into one command, and `scripts/fill_readme_results.py` is the *only* mechanism that ever writes a number into this project's public README — reading real JSON output files, refusing to run at all if any required file is missing, so the documentation can never claim a result that wasn't actually measured on that machine.

12. **The result.** A visitor today can go to a real, live URL, upload their own MRI image, and within roughly a second get back a prediction, a confidence score that's been mathematically corrected to be trustworthy, an uncertainty estimate that leans higher when the model is wrong, a heatmap showing exactly what the model focused on, and — unusually for a portfolio ML project — a clear, explicit accounting of exactly how much to trust the whole system, backed by original, reproducible investigative work rather than a single, unexamined headline accuracy number.
