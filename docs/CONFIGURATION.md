# Configuration

All settings are environment variables read by `neuroclass.settings.Settings`.
Defaults are safe for local development and cost nothing.

## Core

| Variable | Default | Notes |
|---|---|---|
| `NEUROCLASS_ENV` | `development` | `production` requires `NEUROCLASS_JWT_SECRET` |
| `NEUROCLASS_DATA_DIR` | `./var` | SQLite file, uploads, LangGraph SQLite checkpoints |
| `DATABASE_URL` | empty (SQLite) | Postgres URL (`postgres://` and `postgresql://` accepted); enables the Postgres checkpointer |
| `NEUROCLASS_EPHEMERAL_STORAGE` | `false` | set `true` where the disk is wiped on restart; `/ready` reports `persistent: false` and the UI warns |
| `NEUROCLASS_MODEL_DIR` | `models/serving` | the verified ONNX bundle |
| `NEUROCLASS_INFERENCE_TIMEOUT_S` | `20` | per image |
| `NEUROCLASS_WORKERS` | `2` | analysis threads |
| `NEUROCLASS_CORS_ORIGINS` | `http://localhost:3000` | comma-separated |

## Uploads and limits

| Variable | Default |
|---|---|
| `NEUROCLASS_MAX_UPLOAD_BYTES` | 10 MB |
| `NEUROCLASS_MAX_IMAGE_PIXELS` | 25,000,000 |
| `NEUROCLASS_UPLOAD_RETENTION_HOURS` | 24 |
| `NEUROCLASS_ANALYSIS_RATE_LIMIT_PER_MIN` | 10 per client (also used for sign-up/login) |
| `NEUROCLASS_RATE_LIMIT_PER_MIN` | 60 per client (research) |

## Auth

| Variable | Default | Notes |
|---|---|---|
| `NEUROCLASS_JWT_SECRET` | empty | required in production; ≥ 32 random characters |
| `NEUROCLASS_JWT_TTL_MIN` | 720 | |
| `NEUROCLASS_ALLOW_SIGNUP` | `true` | |
| `NEUROCLASS_REVIEWER_EMAILS` | empty | emails that receive the reviewer role at sign-up. Emails are not verified, so list an address and register it yourself before sharing the site |

## LLM (optional, free by default)

| Variable | Default | Notes |
|---|---|---|
| `NEUROCLASS_LLM_PROVIDER` | `auto` | `auto` → `gemini` if `GEMINI_API_KEY` is set, else `none`. `gemini`, `none`, or `anthropic` (paid; requires the next flag) |
| `NEUROCLASS_ALLOW_PAID_LLM` | `false` | the only way to enable a paid provider. Leave it false |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | — | from an AI Studio project with no billing account |
| `NEUROCLASS_LLM_MODEL` | `gemini-3.5-flash-lite` (gemini) | any free-tier model listed on Google's pricing page |
| `NEUROCLASS_LLM_RPM` | 4 | calls per minute, enforced before sending |
| `NEUROCLASS_LLM_DAILY_CAP` | 100 | calls per day (Pacific-time day in-process; UTC day in the database ledger) |
| `NEUROCLASS_LLM_TOOL_ROUNDS` | 1 | tool rounds before the structured call; calls per request ≤ rounds + 1 |
| `NEUROCLASS_LLM_MAX_TOKENS` | 4000 | |
| `NEUROCLASS_LLM_TIMEOUT_S` | 45 | |
| `ANTHROPIC_API_KEY`, `NEUROCLASS_LLM_EFFORT`, `NEUROCLASS_LLM_*_USD_PER_MTOK` | — | only for the paid adapter |

## Retrieval

| Variable | Default | Notes |
|---|---|---|
| `NEUROCLASS_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | must match the index; chosen by `scripts/eval_rag.py` |
| `NEUROCLASS_INDEX_DIR` | `data/rag_index` | the newest index for the embedding model is loaded |
| `NEUROCLASS_EMBED_CACHE` | `data/fastembed_cache` | model download cache |
| `NEUROCLASS_RETRIEVAL_MODE` | `hybrid` | |
| `NEUROCLASS_RERANK` / `NEUROCLASS_RERANKER_MODEL` | `false` / MiniLM cross-encoder | off: +1 s/query on CPU, no recall gain measured |

## Other

| Variable | Notes |
|---|---|
| `NEUROCLASS_MCP_TOKEN` | JWT of the one account the local MCP server may read analyses for |
| `NEUROCLASS_WEB_URL` | Space only: link from the demo to the web app |
| `NEXT_PUBLIC_API_URL` | web build: API base URL (`.../api` on the Space) |
| `NEUROCLASS_E2E_WEB`, `NEUROCLASS_E2E_API` | Playwright targets (default localhost:3000 / :8000) |
| `DATABASE_URL_TEST` | runs the Postgres integration tests |
