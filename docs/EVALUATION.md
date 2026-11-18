# Platform evaluation

Everything here was measured on 2026-11-17 by the scripts named in each
section; the JSON outputs are tracked in `results/`. The model's own evaluation
(accuracy, leakage, calibration, robustness, external validation) is in the
[model card](../MODEL_CARD.md). The IEEE 2024 paper's reported 99.844% is the
paper's own claim (scored on 640 of 1,311 test images) and is never mixed with
these numbers.

## 1. Reliability levels — do the rules separate good results from bad?

`scripts/eval_reliability.py` → `results/metrics/reliability_eval.json`. The
deterministic rules (`reliability.py`, version `2026.11-r1`) were applied to
every validation and test image.

| Split | Level | Images | Share | Accuracy |
|---|---|---|---|---|
| Test (n=1,600) | standard | 1,516 | 94.8% | **96.0%** |
| | caution | 78 | 4.9% | **47.4%** |
| | unreliable | 6 | 0.4% | 50.0% |
| Validation (n=841) | standard | 809 | 96.2% | 97.5% |
| | caution | 27 | 3.2% | 70.4% |
| | unreliable | 5 | 0.6% | 40.0% |

The caution/unreliable levels catch a small share of images that are wrong
about half the time, so routing them to human review is worthwhile. The
"unreliable" groups are tiny (6 and 5 images); their accuracies are not
meaningful on their own. On the external MRI set (n=253): 208 standard,
34 caution, 11 unreliable.

**Input-shift heuristic** (Mahalanobis distance on pooled backbone features,
threshold = validation 99th percentile = 71.4): AUROC **0.95** for test images
vs synthetic non-MRI inputs, but **0.52** (chance) for test vs a different MRI
dataset. Flag rates: validation 1.1%, test 1.4%, external MRI 4.0%, synthetic
80% (tinted MRI 0%). It catches "this is not an MRI", not "this MRI comes from
another scanner" — the UI calls it a heuristic, not OOD detection.

## 2. Retrieval — which embedder and which mode?

`scripts/eval_rag.py --rerank` → `results/rag/retrieval_eval.json`. 50 labelled
questions (`evals/rag_questions.jsonl`): 36 known-item questions written from a
specific article and 14 questions about NeuroClass itself. Relevance is at the
document level; recall is a lower bound (other documents can be relevant).
Latency: single-threaded query on an Apple M3 Pro, warm.

| Configuration | R@5 | R@10 | MRR@10 | nDCG@10 | p50 | p95 |
|---|---|---|---|---|---|---|
| BM25 | 0.960 | 0.960 | 0.920 | 0.918 | 6 ms | 12 ms |
| bge-small dense | 0.907 | 0.920 | 0.910 | 0.900 | 20 ms | 32 ms |
| bge-small hybrid (RRF) | 0.927 | 0.973 | 0.933 | 0.931 | 27 ms | 42 ms |
| **bge-small routed** (deployed) | **0.980** | **1.000** | 0.953 | 0.966 | 27 ms | 41 ms |
| bge-small routed + cross-encoder rerank | 0.980 | 0.980 | 0.970 | 0.971 | 1,029 ms | 1,430 ms |
| arctic-embed-s routed | 1.000 | 1.000 | 0.870 | 0.902 | 27 ms | 41 ms |
| bge-base routed | 1.000 | 1.000 | 0.967 | 0.976 | 75 ms | 117 ms |

"Routed" = hybrid retrieval plus a metadata filter to project documents when
the question is about NeuroClass, and a cap of two passages per document.

The index (`20261117-bge-small-en-v1.5-bddad395`) includes the project documents
as they were on 2026-11-17, before the documentation was expanded; the gold
facts for project questions were checked against that indexed text.

Decisions from these numbers:
- **Routing matters more than the embedder.** Project questions go from 0.786
  (dense) / 0.905 (hybrid) recall@10 to 1.00 when routed.
- **bge-small stays.** bge-base ranks slightly better (MRR 0.967 vs 0.953 —
  about one rank position across 50 questions) at 2.8× the query latency, 3× the
  indexing time (2,155 s vs 711 s) and twice the vector storage.
- **Reranking stays off.** It adds ~1 s per query on CPU and does not improve
  recall here.
- **BM25 is a strong baseline** (0.96 R@10), partly because known-item questions
  share vocabulary with their source article; dense retrieval earns its place on
  the paraphrased questions BM25 misses.

## 3. Generation — what does the user actually see?

`scripts/eval_generation.py` → `results/rag/generation_eval.json`, per-question
outputs in `results/rag/generation_runs/`. Same 50 questions. The 14 project
questions carry gold facts (regexes, each checked to occur in the indexed
documents), e.g. P01 needs "99.844" and "640".

| Variant | Fact recall (project) | Cites a relevant doc | Citation precision | Invalid citations | Unsupported statements shown | p50 |
|---|---|---|---|---|---|---|
| extractive, naive (dense top-5, no verifier) | 0.536 | 94% | 0.776 | 0 | 0 | 13 ms |
| **extractive, full** (routed top-6 + verifier) | **0.661** | **98%** | 0.528 | 0 | 0 | 18 ms |
| Gemini closed-book (LLM only) | not run | | | | | |
| Gemini + naive RAG | not run | | | | | |
| Gemini + full pipeline | not run | | | | | |

Reading this honestly:
- Better retrieval raises **fact recall** (0.54 → 0.66) and the share of
  answers citing a right source (94% → 98%).
- **Citation precision drops** (0.78 → 0.53): the two-per-document cap spreads
  citations over more documents, more of which are not in the (incomplete)
  relevance labels. It is a lower bound, but the trade-off is real.
- Extractive answers quote retrieved sentences verbatim, so "unsupported
  shown" and "invalid citations" are 0 by construction. They are the metrics
  that matter for LLM answers, which the verifier is designed to catch.
- **The LLM variants were not run**: no Gemini key was available, and no paid
  provider is used. The harness is ready, quota-guarded and resumable; with a
  free key it produces the missing rows (see `docs/DEPLOYMENT.md` §3). The
  verifier and the fallback paths are covered by tests with scripted provider
  responses (`tests/app_tests/test_llm_providers.py`, `test_agent.py`).
- The verifier is lexical (token overlap ≥ 0.5 plus exact numbers), so it is a
  guard against invented citations and numbers, not a semantic judge.

## 4. Serving — TensorFlow vs the TF-free path

`scripts/bench_serving.py` → `results/metrics/serving_benchmark.json`. One
process per backend, warm (5 warm-up calls), 48 test images (12 per class),
single-image requests including Grad-CAM, Apple M3 Pro CPU.

| Backend | p50 | p95 | Peak RSS | Load time |
|---|---|---|---|---|
| TensorFlow 2.15: Keras predict + compiled Grad-CAM | 89 ms | 94 ms | 803 MB | 7.0 s |
| **ONNX Runtime, 1 view + NumPy Grad-CAM** | **41 ms** | 42 ms | **262 MB** | **0.24 s** |
| ONNX Runtime, 6 TTA views + NumPy Grad-CAM (platform default) | 175 ms | 238 ms | 571 MB | 0.20 s |

The TF-free path is 2.2× faster for the same work, uses a third of the memory
and loads 30× faster; the platform spends part of that on six test-time
augmentation views for the stability signal. Parity: probabilities within
5e-6 of TensorFlow on all 1,600 test images (0 changed predictions), Grad-CAM
within 9e-6 on 64 images (`models/serving/manifest.json`). Container images:
API 1.49 GB (the TensorFlow API image was 3.29 GB).

On the free HF Space (2 vCPU) a warm analysis took 179 ms of model time in
one observed run; the first request after start-up is slower (500 ms observed)
while ONNX Runtime warms up.

## 5. Tests that were run

| Suite | Result (2026-11-17) |
|---|---|
| Platform, SQLite (`tests/app_tests`) | 81 passed, 3 skipped (Postgres-only tests) |
| Platform, PostgreSQL 17 + pgvector 0.8.1 | 84 passed — incl. history and a paused review surviving an app restart, pgvector store selection |
| Platform on the HF Space dependency set (pydantic 2.12.5, no MCP/Claude SDK) | 72 passed, 7 skipped (SQLite); 75 passed, 4 skipped (PostgreSQL) — skips are the Claude-SDK and Postgres-only tests |
| Research (`tests/`, no TensorFlow) | 65 passed |
| Web unit/component (Vitest) | 8 passed |
| Browser journeys (Playwright, desktop + Pixel 7) | 12 / 12 against the dev stack, the Docker Compose stack and the live deployment |
| Dependency audit | pip-audit: no known vulnerabilities in `requirements-app.txt`; npm audit (prod): 0 |

## 6. Live verification on production (Neon + Gemini)

| Check | Result |
|---|---|
| `/api/ready` | PostgreSQL via Neon's pooled endpoint; `retrieval.backend: pgvector` (7,355 chunks synced); Gemini `gemini-3.5-flash-lite` enabled; `paid_allowed: false` |
| Research answer through Gemini | "What share of test images have a near-duplicate in training?" → 44.6%, cited to the project README, verified, 0 statements removed; 2 calls, 7,740 input / 141 output tokens, $0 |
| Reviewer privilege | signing up with a listed email but no code → role `user`; claiming with the code → `reviewer` |
| Restart persistence | analysis with `review_mode=always` → `awaiting_review`; Space restarted; still in history and the review queue; approved → `completed`, explanation drafted by Gemini |
| Neon plan | Free, $0/month, no payment method; 0.05 CU-hours and 33 MB used after these tests |

Test accounts `student-check@neuroclass.test` and `review-check@neuroclass.test`
and their analyses remain in the database; the temporary reviewer settings used
for this check were removed from the Space afterwards.
