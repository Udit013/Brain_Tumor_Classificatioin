# Case study: from a published benchmark to a $0 research platform

## 1. Starting point

In 2024 the project's IEEE ICC-ROBINS paper compared four CNNs for four-class
brain-tumour MRI classification and reported 99.844% accuracy for
EfficientNetB3. Two years later the question was: what would it take to put
that work in front of people honestly — and keep it running with no budget?

## 2. The research audit (what the number actually meant)

Re-running the original notebooks byte-for-byte and auditing the data changed
the story before any product work began:

- **The headline was scored on half the test set.** The test generator was not
  shuffled and evaluation stopped early: 99.844% = 639/640 images, 49% of the
  1,311-image test split.
- **The benchmark leaks.** A perceptual-hash audit found that 44.6% of test
  images have a near-duplicate in training (plus 114 exact duplicates). The
  paper-recipe model scored 95.9% on leaked images and 85.7% on novel ones.
- **The hardware changed the function.** Apple's Metal backend computed a
  different function in graph mode (93.81% vs 90.25% on CPU); the root cause was
  traced to a graph optimiser, and every metric since is measured on CPU, the
  computation the deployed model runs.

The paper's number stays in the repository as the paper's claim; every other
number is measured and labelled.

## 3. A better model, selected without peeking

A leakage-safe validation split (near-duplicate clusters kept on one side), a
two-stage fine-tune with noise and blur augmentation, and one final test
evaluation moved accuracy from 90.25% to 93.44%, novel-image accuracy from
85.7% to 89.4%, glioma recall from 64.3% to 79.5% and mean accuracy under 30
corruptions from 67.3% to 90.7% (paired exact McNemar test significant). A
promotion gate now encodes these as rules — written after the fact and labelled
as such — and refuses any candidate that misses more gliomas.

## 4. From classifier to research platform

The product question was not "how do we show a label" but "how do we show a
label *with its limits*":

- **Reliability as rules, validated.** Calibrated confidence, test-time
  augmentation agreement, an input-shift distance and the model's measured
  per-class weaknesses feed deterministic rules. On test, "standard" results
  (95% of images) are right 96.0% of the time; "caution" results 47.4% — so
  those go to human review.
- **An agent where an agent helps.** A LangGraph workflow (validate → infer →
  assess → retrieve → explain → verify → human review → report) with typed
  state, bounded retries and revisions, a database checkpointer and
  `interrupt()` for review. It is one graph of testable functions, not a swarm.
- **Grounded explanations.** Retrieval over 241 openly licensed PMC articles plus
  the project's own documents; a verifier deletes any statement whose citation
  wasn't retrieved, whose numbers don't appear in its sources, or whose support
  is weak — and shows what it deleted and why.

## 5. The constraint that shaped everything: $0

The owner's budget is zero, permanently. That ruled out the obvious stack
(a paid LLM, a container host, a managed queue) and forced better engineering:

| Problem | Constraint-driven answer | Result |
|---|---|---|
| TensorFlow only needed for Grad-CAM, but it made the image 3.3 GB | export one ONNX graph with the last conv map; compute Grad-CAM in closed form through the dense head; refuse to export unless it matches TensorFlow | 41 ms vs 89 ms per image, 262 MB vs 803 MB RSS, 0.24 s vs 7 s load; probabilities within 5e-6 of TF on all 1,600 test images |
| No paid LLM, no key | provider abstraction; extractive mode as a first-class product; Gemini free tier as an option behind a quota guard; paid provider refused unless explicitly allowed | the platform is fully usable with no LLM; an exhausted quota degrades, never bills |
| No paid host; HF requires PRO for new Spaces | reuse the existing free Space as an API-only backend (FastAPI at `/api`, a small landing page at `/`) | live, free; Gradio 5 → 6 upgrade also cleared dependency advisories |
| No managed queue or Redis | in-process worker + database job state + SSE; checkpoints make restarts resumable | a review paused before a restart resumes after it (verified on Postgres) |
| Embedding APIs cost money | local ONNX embeddings, chosen by evaluation | routed hybrid search reaches recall@10 = 1.00 on the labelled set at 27 ms |

## 6. What went wrong (and what it taught)

- **The production build broke while dev worked.** Next 16's cache components
  refuse to prerender a layout that reads the URL; moving the read into a
  Suspense-wrapped leaf fixed it. Lesson: CI now builds production on every push.
- **The first Space deploy failed twice.** HF installs `gradio[oauth,mcp]`,
  whose extras pin `mcp<2` and `pydantic<=2.12.5`; then Gradio's SSR server
  took the port uvicorn needed. Fixes: a Space-specific requirements set and
  `ssr_mode=False` — and a CI job that installs exactly what HF installs.
- **A free-tier field broke a schema.** Recording `"tier": "free"` in a
  numbers-only usage map failed validation; the test that exercised the full
  Gemini path caught it before deploy.
- **Retrieval surprised us.** BM25 alone scored 0.96 recall@10; the real win was
  routing project questions to project documents (0.79 → 1.00), not a bigger
  embedder. Better retrieval also lowered citation precision (0.78 → 0.53) —
  reported, not hidden.
- **Some things could not be measured.** The LLM-vs-RAG generation comparison
  needs a free Gemini key; the harness is ready and the result is reported as
  "not run" instead of estimated.

## 7. Trade-offs I would defend in review

- Free Neon Postgres (history, reviews, checkpoints, pgvector) behind a
  fallback: if Neon suspends at its free limit, retrieval drops to the
  in-memory index and the API answers 503 where data is needed — never a bill.
- Extractive answers as the default — less fluent, but every sentence is a
  verifiable quotation.
- Lexical verification — cheap, deterministic, explainable; not a semantic judge.
- No reranker — the measured gain was zero recall for ~1 s per query on CPU.
- A single process — right for a demo with modest traffic; the database-backed
  job state is what would let it scale out later.

## 8. Research-to-product, in one paragraph

The paper asked which architecture classifies brain MRI best. The product asks
what a researcher can trust about one prediction. Getting there meant
correcting the paper's evaluation (partial test set, leakage, device effects),
improving the model on honest terms, then wrapping it in calibrated confidence,
validated reliability rules, explanations that cannot cite what they didn't
read, human review where the rules say so, and a deployment whose failure mode
under load is "slower" or "less fluent" — never "wrong without saying so" and
never "billed".
