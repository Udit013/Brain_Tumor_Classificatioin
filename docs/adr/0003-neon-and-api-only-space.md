# ADR 0003 — Neon Postgres, reviewer codes, and an API-only Space

Status: accepted (2026-11-17). Amends ADR 0002 (database default, Space surface).

## Context

The owner created a Neon project (`neuroclass`, branch `production`, Free plan)
and added `DATABASE_URL` and `GEMINI_API_KEY` as Space secrets. Neon's console
suggested `neon init --services postgres,auth` and `neon deploy`. The owner also
asked that the Space stop presenting a separate Gradio classifier, and that the
"listed email = reviewer" rule be fixed because emails are not verified.

## Decisions

1. **Neon as a plain Postgres database.** The existing SQLAlchemy/Alembic,
   LangGraph `PostgresSaver` and pgvector code is used unchanged in shape. The
   Neon CLI template is not run: Neon Auth would duplicate the platform's auth,
   and `neon deploy` provisions resources the platform doesn't need.
2. **Pooled or direct endpoints both supported.** When the host contains
   `-pooler` (PgBouncer, transaction mode), psycopg prepared statements are
   disabled for SQLAlchemy and the checkpointer. Pools pre-ping, recycle at
   240 s and check connections, because Neon closes them while scaled to zero.
3. **pgvector on Neon, memory as fallback.** The bundled index is synced into
   pgvector on first start (idempotent per version). `FallbackStore` searches
   pgvector and silently uses the exact in-memory search if the database is
   unreachable; results are identical (tested).
4. **Degrade, don't fail, at Neon's limits.** Database errors become 503
   "temporarily unavailable"; tracing and the LLM ledger never break a request;
   research answers, benchmarks and samples keep working.
5. **Reviewer role = listed email + operator code.** `NEUROCLASS_REVIEWER_CODE`
   (≥ 16 chars, constant-time compare) is required at sign-up or through
   `POST /auth/reviewer`. Without a code nobody becomes a reviewer. Neon Auth
   was not needed for this; email verification would need a mail provider and
   is left out to stay at $0.
6. **API-only Space.** The Space keeps the Gradio SDK as its free Python host
   (Docker Spaces can't be created on a free account; switching SDKs risks the
   same restriction) but serves no Gradio UI: `/` is a small landing page that
   links to the Vercel app and shows live status; `/api` is unchanged.

## Verification (2026-11-17, production)

`/api/ready`: PostgreSQL via the pooled endpoint, `retrieval.backend: pgvector`,
Gemini enabled, `paid_allowed: false`. A Gemini answer was cited and verified
(2 calls, $0). A review paused before a Space restart survived it and completed
after approval. Neon console: Free plan, $0/month, no payment method, one
project/branch/database/compute, no Neon Auth; 0.05 CU-hours and 33 MB used.
