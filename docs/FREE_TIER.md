# Running NeuroClass for $0

NeuroClass is designed so that **nothing in the default deployment can charge
money**, and so that every external dependency is optional: if a free quota
runs out, the affected feature degrades to a free fallback and says so. It
never escalates to a paid request.

Limits below were read from each provider's own pricing/documentation pages on
**2026-11-17**. Free tiers change; re-check the linked pages before relying on
a number, and treat the safeguards in this repository (not the numbers) as the
guarantee.

## The deployment at a glance

| Layer | Choice | Plan | Required? | Fallback when absent or exhausted |
|---|---|---|---|---|
| Backend API + demo | Hugging Face Space (existing Gradio Space) | CPU Basic (free) | yes | — (the Space wakes on the next visit) |
| Web app | Vercel | Hobby (free) | no | the Space's own demo at `/` and API docs at `/api/docs` |
| Database | SQLite on the Space disk | — | default | — |
| Durable database | Neon Postgres | Free | optional | SQLite (history resets on restart) |
| LLM | Gemini API | Free tier, unbilled project | optional | extractive answers (quoted, cited passages) |
| Embeddings | fastembed, `BAAI/bge-small-en-v1.5`, local ONNX | — (open model) | yes | — (no API, no quota) |
| Auth | built in (scrypt + HS256 JWT) | — | yes | — (no auth provider) |
| Observability | structured JSON logs + trace events in the database | — | yes | — (no SaaS) |
| CI | GitHub Actions on a public repository | standard runners | yes | run the same commands locally |

Deliberately **not** used: Redis/Celery (in-process worker + database job
state instead), a paid vector database (pgvector or the in-process index), a
hosted auth provider, MLflow/W&B (file registry + Hub tags), paid embeddings,
Render's free Postgres (deleted after 30 days), and any provider that needs a
card on file to stay on a free plan.

## Provider by provider

### Hugging Face Spaces — backend host
Source: [huggingface.co/pricing](https://huggingface.co/pricing),
[Spaces overview](https://huggingface.co/docs/hub/spaces-overview),
[Spaces hardware](https://huggingface.co/docs/hub/spaces-gpus).

| | |
|---|---|
| Plan | CPU Basic — listed as **FREE**: 2 vCPU, 16 GB RAM, 50 GB disk |
| Payment method | not needed for CPU Basic |
| Overages | none: free hardware has no metered usage |
| Sleep | sleeps after ~48 h without traffic; the next visitor wakes it (cold start) |
| Storage | disk is **not persistent**: lost on restart, rebuild or sleep |
| Restriction | creating *new* Gradio/Docker Spaces requires PRO; NeuroClass reuses the owner's existing Space |
| At the limit | there is no usage limit to hit; heavy traffic queues (demo concurrency 2, API workers 2) |

Never enable: upgraded hardware, persistent storage add-ons, or "always on".

### Vercel — web app
Source: [vercel.com/docs/plans/hobby](https://vercel.com/docs/plans/hobby) (updated 2026-11-12),
[vercel.com/docs/limits](https://vercel.com/docs/limits).

| | |
|---|---|
| Plan | Hobby — personal, non-commercial projects |
| Payment method | not required for Hobby |
| Overages | none on Hobby ("there are no billing cycles"); over an allowance the feature pauses, typically ~30 days |
| Monthly allowances | 100 GB fast data transfer, 1M function invocations, 4 CPU-hours active CPU, 5,000 image transformations |
| Function limits | 300 s max duration, 2 GB / 1 vCPU |
| NeuroClass usage | all pages are static or partially prerendered; data is fetched from the Space by the browser, so functions are barely used |

Never: upgrade to Pro, add a team seat, or enable paid add-ons.

### Neon — optional durable Postgres
Source: [neon.com/docs/introduction/plans](https://neon.com/docs/introduction/plans) (updated 2026-11-12),
[free-plan limits FAQ](https://neon.com/faqs/free-plan-limits-and-quotas) (updated 2026-11-03).

| | |
|---|---|
| Plan | Free, $0/month |
| Compute | 100 CU-hours per project per month, autoscaling up to 2 CU |
| Storage | 1 GB per project (20 GB across projects) |
| Scale to zero | after 5 minutes idle; cannot be disabled on Free |
| Payment method | not stated as required for the Free plan on the pages read; do not add one |
| Overages | none on Free |
| At the limit | compute **suspended** until next month when CU-hours or transfer run out; writes fail above 1 GB; data is not deleted |
| Footprint | analyses, reviews, trace events, checkpoints; the optional pgvector copy of the index is ~20 MB |

Recommended setting: in the Neon console, cap autoscaling at 0.25 CU so the
100 CU-hours last ~400 active hours a month. If Neon suspends, set
`DATABASE_URL` empty (or leave it; `/api/ready` reports the database as down and
the API keeps serving benchmarks and research search). Wakes after scale-to-zero
add a few hundred milliseconds to the first query.

### Gemini API — optional LLM
Source: [pricing](https://ai.google.dev/gemini-api/docs/pricing),
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits),
[models](https://ai.google.dev/gemini-api/docs/models) (these redirect to
aistudio.google.com/docs/...).

| | |
|---|---|
| Plan | Free tier, on a project with **no billing account linked** |
| Models | free-tier models include `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite` (stable). Default: `gemini-3.5-flash-lite`; override with `NEUROCLASS_LLM_MODEL` |
| Rate limits | per project and per model, shown in AI Studio (RPM, TPM, RPD); daily quotas reset at midnight Pacific |
| Payment method | not needed. Moving to a paid tier requires setting up billing |
| Overages | not possible on an unbilled project: over-quota requests fail (429) |
| Auto-upgrade | Google upgrades a project's tier automatically **once billing is linked**. Never link billing to the project that issues this key |
| Data use | free-tier content may be used to improve Google's products. NeuroClass sends model numbers, reliability findings, public literature excerpts and the user's question/note — never images. The assistant page says so |
| At the limit | NeuroClass stops calling (see safeguards) and answers extractively, showing the reason |

The per-model RPM/RPD values are not published on the docs pages; they appear
in AI Studio for your project. Set `NEUROCLASS_LLM_RPM` and
`NEUROCLASS_LLM_DAILY_CAP` **below** them.

### Local models — embeddings and (optional) reranking
`BAAI/bge-small-en-v1.5` (MIT) and `Xenova/ms-marco-MiniLM-L-6-v2` run on CPU
through fastembed/ONNX Runtime. They are downloaded once from the Hugging Face
Hub (free) and cached. No API, no quota, no cost.

### GitHub Actions — CI
Public repositories use standard GitHub-hosted runners without per-minute
charges. CI never calls an LLM, never needs a GPU, and runs entirely on fakes
and the pgvector service container.

## Safeguards in the code

| Risk | Safeguard | Where |
|---|---|---|
| A paid LLM gets used by accident | `auto` selects Gemini or nothing; Anthropic requires `NEUROCLASS_ALLOW_PAID_LLM=true` explicitly | `settings.resolved_llm_provider` |
| Exceeding the free quota | per-minute and per-day budgets taken **before** a request; no slot → no request → extractive | `agent.llm.QuotaGuard` |
| Hammering a 429 | SDK retries disabled; a 429 pauses the LLM until the server's retry delay, or until midnight Pacific for per-day quotas | `agent.llm_gemini` |
| Runaway agent loops | at most `tool_rounds` + 1 calls per request (default 2); graph revisions capped at 2 | `llm_gemini`, `agent.graph.MAX_REVISIONS` |
| Long or hung calls | 45 s LLM timeout, 20 s inference timeout, bounded retries on inference only | settings |
| Abuse of the public API | per-client sliding-window limits (10 analyses/min, 60 research calls/min), 10 MB uploads, 25 MP images, JPEG/PNG only | `api.deps`, `vision.validation` |
| Disk filling on the Space | uploads deleted after 24 h (hourly job), images processed in memory by the demo | `runtime.purge_expired_uploads` |
| Tests touching a real provider | an autouse fixture removes every LLM key from the environment; the Gemini/Claude adapters are tested against fake clients | `tests/app_tests/conftest.py` |
| Silent fallback | every extractive explanation carries `fallback_reason`; `/api/ready` shows provider, model, budget use and pause state | `schemas.Explanation`, `api/routers/system.py` |

## What "free" does and does not cover

This is a free **public demo with modest traffic**, not a free production
service. Expect: a cold start after idle periods; two concurrent demo
predictions; a small daily LLM budget (when a key is configured at all); and,
without Neon, history that resets on restart. A burst of visitors makes
requests queue or hit the per-client rate limit — it never makes anyone pay.
Scaling beyond this needs paid infrastructure, which this project does not use.
