# ADR 0001 — From MRI classifier to research platform

Status: accepted (2026-11-17). Supersedes nothing; the research package and
its decisions stay in force.

## Context (audit, 2026-11-17)

What exists and works (verified this session, not assumed):

| Asset | State |
|---|---|
| IEEE ICC-ROBINS 2024 notebooks (`legacy/`, root copies) | byte-for-byte originals; reported 99.844% = 639/640 test images |
| `src/btc` research package | parity reproduction, leakage audit, leakage-safe split, two-stage fine-tune, evaluation, calibration, robustness, uncertainty, ONNX export; 65 tests, CI green |
| Deployed model | fine-tuned EfficientNetB3, Hub `Udit013/brain-tumor-efficientnetb3@d655e25`; 93.44% test, 89.4% novel subset, 72.7% external |
| Gradio Space | live, free CPU; previous model at tag `parity-2024` |
| FastAPI (`btc.serve`) | single `/predict`, TensorFlow + Keras for Grad-CAM, Docker image 3.29 GB |

Limits found in the audit:

- **Serving depends on TensorFlow only for Grad-CAM.** That drives the image to
  3.3 GB and rules out small hosts.
- **No persistence, auth, job model, history or provenance** beyond JSON files.
- **No literature grounding.** The Space explains itself with fixed text.
- **Preprocessing parity rests on an unpinned decoder.** Pillow 13 decodes 8 of
  200 test JPEGs differently from Pillow 10.3/12.3 (up to 7 grey levels); the
  model was evaluated with 10.3. Nothing would have caught a silent upgrade.
- **No LLM credentials exist in the development environment**, and a public
  demo cannot expect visitors to bring keys.
- **HF now requires a PRO plan to create new Gradio/Docker Spaces** (HTTP 402),
  so new hosted services need another free/cheap host.

## Decisions

### 1. Keep the research package; add a separate platform package
`src/btc` (research, TensorFlow) is untouched in role. A new `src/neuroclass`
package holds the platform (API, agent, RAG, persistence). It imports only the
TensorFlow-free pieces of `btc` (`config`, `preprocess`, `serving_math`).
Separate environments: `.venv` (research, TF 2.15) and `.venv-app` (platform).

### 2. TensorFlow-free inference with analytic Grad-CAM
`btc.export_serving` exports one ONNX graph (probabilities + the 8×8×1536
`top_activation` map) and the head weights. Grad-CAM's gradient through the
small head (max-pool → BN → 3 dense layers) is closed-form
(`btc.serving_math`). Verified on export, and the export refuses to write a
manifest otherwise: probabilities within 5e-6 of TensorFlow on all 1,600 test
images (0 argmax changes, 93.44%), heatmaps within 9e-6 on 64 images.
*Rejected:* exporting `GradientTape` gradients to ONNX (fragile op coverage);
keeping TensorFlow (image size, memory, cost).

### 3. Input-shift warning = Mahalanobis heuristic, measured and labelled
Pooled backbone features of the training split define a Gaussian reference
(Ledoit-Wolf shrinkage); the threshold is the 99th percentile on clean
validation images. It is reported as a heuristic with its measured AUROC on
near-OOD (external MRI) and far-OOD (non-MRI) inputs — never as OOD detection.

### 4. Persistence: SQLAlchemy 2 + Alembic; PostgreSQL + pgvector in deployment
One relational store for analyses, jobs, reviews, traces, users and the RAG
index. SQLite is supported for local runs and unit tests; PostgreSQL with
pgvector is the deployment target and is exercised by integration tests in CI
(service container). Vector search goes through a small interface with an exact
in-process backend and a pgvector backend; they are benchmarked against each
other rather than assumed.

### 5. Retrieval: local ONNX embeddings, BM25, hybrid, optional rerank — chosen by measurement
Embeddings and reranking run locally through fastembed (ONNX, CPU): no
per-query cost, no third-party data flow. Candidate embedders and the retrieval
variants (lexical, dense, hybrid with reciprocal-rank fusion, + metadata
filters, + cross-encoder rerank) are compared on a labelled question set; the
default is whatever the measurements support at acceptable latency.

### 6. LLM: Claude through the official Anthropic SDK — optional by design
Explanation and research answers use `messages.parse` with Pydantic schemas:
statements carry citation IDs, so a deterministic verifier can check them.
Default model `claude-opus-5-5` (configurable). Without credentials, the
platform runs an **extractive** mode that quotes retrieved passages, labelled as
such — the workbench, benchmarks and reports never depend on an LLM.

### 7. Orchestration: one LangGraph graph with deterministic gates
validate → infer → assess reliability (deterministic) → retrieve → explain
(LLM or extractive) → verify citations (deterministic) → human review
(`interrupt`, only when reliability rules require it) → report. Bounded
retries and per-node timeouts; checkpoints in the same SQL database
(SQLite/Postgres checkpointer). No multi-agent swarm: each node is a function
with a typed contract and its own tests.

### 8. Jobs: in-process worker + database state + SSE; no Redis/Celery
Analyses take ~0.2 s (vision) to ~20 s (with an LLM). A thread pool executes
graph runs; state and step events are persisted, so progress streams over SSE
and survives page reloads; a restart resumes from the graph checkpoint.
*Rejected for now:* Redis + Celery — a second always-on service for a
single-instance demo. Revisit when more than one API replica is needed.

### 9. MLOps: file registry + Hub tags + regression gate; no MLflow server
Each model version has a manifest with checksums, metrics and provenance; a
promotion script refuses a candidate that regresses on defined metrics. Hub
tags remain the rollback mechanism. *Rejected:* an MLflow tracking server —
one more stateful service for a single model line.

### 10. Observability: structured logs + persisted trace events
Every graph step, retrieval, LLM call (tokens, cost) and inference is a trace
event in the database, visible in the UI. No image bytes or file names are
logged. OpenTelemetry export is a documented extension point, not a dependency.

### 11. MCP: a narrow, read-only server
Benchmark metrics, model metadata, reproducibility facts and literature search
are exposed as MCP tools (stdio). No uploads, analyses or file access.

### 12. Frontend: Next.js (App Router) + TypeScript + Tailwind
Separate deployable (Vercel-class hosting), typed client generated from the
API's OpenAPI schema.

## Costs and risks

- Running costs are dominated by the API host (needs ~1 GB RAM without
  TensorFlow) and optional LLM calls; quantified in `docs/COST.md`.
- Corpus coverage is limited to openly licensed articles; answers can only be
  as good as that subset, and the UI says so.
- Single-seed model results remain a stated limitation.
