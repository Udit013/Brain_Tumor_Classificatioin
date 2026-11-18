# ADR 0002 — Zero-cost deployment and a free-first LLM layer

Status: accepted (2026-11-17). Amends ADR 0001 decisions 6 (LLM), 4 (hosting of
the database) and the hosting assumptions in "Costs and risks".

## Context

The owner's budget is **$0, permanently**: no paid services, no pay-as-you-go
billing, no trials treated as free, nothing that can start charging after a
quota runs out. ADR 0001 made Claude the LLM (paid per token, and no key
exists) and assumed a small paid container host.

Facts checked against the providers' own pages on 2026-11-17 (details and links
in [`docs/FREE_TIER.md`](../FREE_TIER.md)):

- **Hugging Face**: CPU Basic (2 vCPU, 16 GB RAM, 50 GB non-persistent disk) is
  free, sleeps after ~48 h idle. Creating *new* Gradio/Docker Spaces now needs
  PRO; the owner already has a Gradio Space on CPU Basic.
- **Gemini API**: a free tier exists for Flash/Flash-Lite models; a project only
  moves to a paid tier after a billing account is linked; requests over a quota
  fail with a rate-limit error. Free-tier content may be used to improve
  Google's products.
- **Vercel Hobby**: no billing cycle; usage over an allowance pauses the feature.
- **Neon Free**: 100 CU-hours/project/month, 1 GB storage/project, scale to zero;
  compute is suspended (not billed) when the allowance is used.
- **Render Free**: services spin down after 15 min; free Postgres is deleted
  after 30 days (+14-day grace) — unsuitable as the system of record.

## Decisions

1. **Backend on the existing free Space.** The Space switches from Gradio 5.50 +
   TensorFlow to Gradio 6.32 + the TF-free platform: one process serves the
   FastAPI platform at `/api` and a small Gradio demo at `/`. Gradio 6.32's
   pins (`pillow<14`, `starlette>=1`, `pydantic<=3`) accept the platform's
   pins, which Gradio 5.50 did not (`pillow<12`, `starlette<1`) — the upgrade
   also clears the Space's remaining Pillow/Starlette advisories.
2. **Frontend on Vercel Hobby** (static pages + one partially prerendered
   route; all data fetched client-side from the Space API).
3. **Database: SQLite by default, Neon Postgres optional.** On the Space,
   SQLite on the temporary disk is honest about being temporary (`/api/ready`
   reports `persistent: false`, the UI says so). Setting `DATABASE_URL` to a
   free Neon project makes history, reviews and LangGraph checkpoints durable;
   pgvector there is optional (the in-process index is the default store).
4. **LLM: provider abstraction, Gemini free tier as default, no LLM required.**
   `agent.llm` holds the provider-neutral contract (prompts, tools, output
   schema, untrusted-data wrapping); `agent.llm_gemini` and `agent.llm_anthropic`
   are adapters. `NEUROCLASS_LLM_PROVIDER=auto` selects Gemini only when
   `GEMINI_API_KEY` is set, otherwise none (extractive). The Anthropic adapter
   is kept (it exercises the same contract) but is refused unless
   `NEUROCLASS_ALLOW_PAID_LLM=true` — `auto` can never select it.
5. **Quota guard before every call.** Per-minute and per-day budgets (defaults
   4/min, 100/day) are taken *before* a request is sent; a provider 429 pauses
   the LLM until the daily reset (per-day quota) or the server's retry delay;
   SDK retries are disabled so a 429 is never retried. Every failure path ends
   in the extractive composer, with the reason shown to the user.
6. **At most two LLM calls per request** (one optional tool round + one
   structured-output call), so the free daily quota covers more requests.
7. **Embeddings stay local** (fastembed/ONNX, bge-small): no API, no quota.

## Consequences

- A visitor after a quiet weekend waits for a cold start (measured on deploy).
- Without Neon, history and the review queue reset when the Space restarts.
- Without a Gemini key the platform is fully functional in extractive mode;
  the LLM-vs-RAG generation comparison cannot be run and is reported as "not
  run" rather than estimated.
- Free tiers change. The guard, the provider switch and the "LLM optional"
  design make a quota change degrade the service instead of billing anyone.
