# Deploying and maintaining NeuroClass for $0

Live deployment (verified 2026-11-17, Neon and Gemini connected):

| Surface | URL | Host / plan |
|---|---|---|
| Web app | https://neuroclass-lab.vercel.app | Vercel Hobby |
| API (docs at `/api/docs`) | https://udit013-brain-tumor-mri-classifier.hf.space/api | Hugging Face Space, CPU Basic |
| Database | Neon project `neuroclass`, branch `production` (Postgres + pgvector) | Neon Free |
| Space page | https://huggingface.co/spaces/Udit013/brain-tumor-mri-classifier | API landing page (no classifier UI) |

Read [`FREE_TIER.md`](FREE_TIER.md) first: it lists every limit and what happens
at it. The rule for every step below is **never add a payment method, never
upgrade a plan, never enable a paid add-on**. If a screen asks for one, stop:
the step is not needed.

```
Browser ──► Vercel (Next.js, static/PPR) ──fetch──► HF Space :7860
                                                     ├─ /      landing page (links to the web app)
                                                     └─ /api   FastAPI platform
                                                              ├─ ONNX model + NumPy Grad-CAM
                                                              ├─ LangGraph workflow (Postgres checkpoints)
                                                              ├─ retrieval: pgvector on Neon ─fallback─ in-memory index
                                                              ├─ Neon Postgres: history, reviews, users, traces
                                                              └─ Gemini free tier ─or─ extractive (quota guard)
```

## 1. Backend: the Hugging Face Space

Free accounts can no longer *create* Gradio or Docker Spaces (that needs PRO),
so NeuroClass runs on an existing Gradio Space. It is API-only: the Gradio SDK is
just the free Python host — `app.py` runs uvicorn itself and serves no Gradio
interface. Switching the existing Space to the Docker SDK was considered and
rejected: free accounts can't create Docker Spaces, and changing the SDK of an
existing one risks the same restriction for no functional gain. If you don't have one, run the
API locally or in Docker (section 5); there is no free substitute that keeps
the same behaviour without a card.

```bash
hf auth login                                   # once; token stays in your keychain
```
```bash
.venv-app/bin/python deploy/hf_space/build.py   # assembles build/hf_space (~85 MB)
```
```bash
hf upload Udit013/brain-tumor-mri-classifier build/hf_space . --repo-type space
```

`build.py` copies the platform package, the verified ONNX bundle (checksummed in
`models/serving/manifest.json`), measured metrics, the bge-small literature
index and the example images, and writes a `requirements.txt` that satisfies
the extras HF installs with Gradio (`gradio[oauth,mcp]` pins `pydantic<=2.12.5`
and `mcp<2`; the CI job `space` installs exactly that set and runs the tests).

Space settings → **Variables and secrets**:

| Name | Kind | Value |
|---|---|---|
| `NEUROCLASS_JWT_SECRET` | secret | random, ≥ 32 characters (required in production) |
| `DATABASE_URL` | secret | Neon connection string — see section 4 |
| `GEMINI_API_KEY` | secret | free-tier key — see section 3 |
| `NEUROCLASS_CORS_ORIGINS` | variable | your Vercel URL, e.g. `https://neuroclass-lab.vercel.app` |
| `NEUROCLASS_WEB_URL` | variable | same URL (linked from the landing page) |
| `NEUROCLASS_LLM_PROVIDER` | variable | `auto` |
| `NEUROCLASS_EPHEMERAL_STORAGE` | variable | `false` (only matters without `DATABASE_URL`) |
| `NEUROCLASS_REVIEWER_EMAILS` | variable, optional | emails eligible for the reviewer role |
| `NEUROCLASS_REVIEWER_CODE` | secret, optional | ≥ 16 random characters; a listed email must also present it (section 1a) |

Keep the hardware on **CPU Basic** and do not add persistent storage.

Checks after a deploy:
```bash
curl -s https://udit013-brain-tumor-mri-classifier.hf.space/api/ready
```
`status: ready`, `database.backend: postgresql`, `retrieval.backend: pgvector`,
`llm.provider: gemini`, `llm.paid_allowed: false`.

### 1a. Reviewer access

Emails aren't verified, so a listed email alone grants nothing. To make yourself
a reviewer: add your email to `NEUROCLASS_REVIEWER_EMAILS` and a random secret
(≥ 16 characters, e.g. from a password manager) to `NEUROCLASS_REVIEWER_CODE`
in the Space settings; then create your account on the web app and open "I have
a reviewer access code" during sign-up. An existing account can claim the role
with `POST /api/auth/reviewer {"code": ...}`. Share the code only with people
you add to the list. Without a code configured, nobody can become a reviewer.

Rollback: the Space repository keeps history and tags. `tf-gradio5-2026-10` is
the last TensorFlow/Gradio-5 version, `parity-2024` the previous model.
```bash
hf download Udit013/brain-tumor-mri-classifier --repo-type space --revision tf-gradio5-2026-10 --local-dir /tmp/space-rollback
```
then upload that folder as above.

## 2. Frontend: Vercel Hobby

The Vercel project `neuroclass` is connected to the GitHub repository: every
push to `main` builds and deploys production; other branches get preview URLs.
Project settings: Root Directory `web`, framework Next.js, Node 24.x, and the
production variable `NEXT_PUBLIC_API_URL=https://udit013-brain-tumor-mri-classifier.hf.space/api`.

Setting it up from scratch (CLI):
```bash
cd web && vercel link --project neuroclass
```
```bash
printf 'https://udit013-brain-tumor-mri-classifier.hf.space/api' | vercel env add NEXT_PUBLIC_API_URL production
```
then connect the repository in the Vercel dashboard (Settings → Git) and set the
Root Directory to `web`.

`web/vercel.json` pins the framework to Next.js (a project created from the CLI
otherwise gets the "Other" preset and serves a 404). `NEXT_PUBLIC_API_URL` is
read at build time, so change it with a redeploy. Every page is static or
partially prerendered and fetches data in the browser, so the Hobby function
allowance is barely touched.

## 3. Gemini free tier (configured)

1. Sign in at [aistudio.google.com](https://aistudio.google.com) and create an
   API key in a project **with no billing account**. Do not click "Set up
   billing" / "Upgrade": linking billing is what moves a project to a paid tier.
2. Note the project's free-tier limits for the model on AI Studio's rate-limit
   page (requests per minute and per day).
3. Add `GEMINI_API_KEY` as a Space secret. Set `NEUROCLASS_LLM_RPM` and
   `NEUROCLASS_LLM_DAILY_CAP` below the limits you saw (defaults 4 and 100;
   each request uses at most 2 calls). Optionally pick another free-tier
   model with `NEUROCLASS_LLM_MODEL` (default `gemini-3.5-flash-lite`).
4. `/api/ready` should show `"provider": "gemini"`. When the budget or the
   provider's quota is reached, explanations switch to extractive mode with the
   reason shown; the LLM resumes after the pause.

Privacy: free-tier content may be used by Google to improve its products.
NeuroClass never sends images; it sends model numbers, reliability findings,
public literature excerpts and the user's question or note.

To run the generation evaluation with the key (free, resumable across days):
```bash
GEMINI_API_KEY=... PYTHONPATH=src .venv-app/bin/python scripts/eval_generation.py --rpm 4 --max-calls 60
```

## 4. Neon Postgres + pgvector (configured)

NeuroClass uses Neon as a plain PostgreSQL database through its own SQLAlchemy,
Alembic and auth code. Neon's generated setup (`neon init --services
postgres,auth`, `neon deploy`) is **not** used: it would provision Neon Auth and
deployment resources the platform doesn't need. No Neon CLI login is required.

1. Neon account on the **Free** plan, no card, one project with one branch
   (`production`). In the compute settings, cap autoscaling at **0.25 CU**
   (stretches the 100 CU-hours/month) and keep scale-to-zero on.
2. Put the connection string (`postgresql://...?sslmode=require`) in the Space
   secret `DATABASE_URL` — never in a frontend variable or a committed file.
   Pooled (`-pooler` host) and direct endpoints both work: the engine and the
   LangGraph checkpointer detect the pooler and turn off server-side prepared
   statements (PgBouncer transaction mode), and both pools pre-ping and recycle
   connections that Neon closed while scaled to zero.
3. On start, the API runs the Alembic migrations (non-destructive, versioned:
   they create the `vector` extension and tables if missing and never drop
   data), LangGraph creates its checkpoint tables, and on first use the bundled
   literature index is copied into pgvector (`NEUROCLASS_SYNC_INDEX_TO_DB=true`,
   idempotent per index version; ~7,355 rows, ~20 MB). Retrieval then runs on
   pgvector with the in-memory index as fallback.

Verified on the live Space (2026-11-17): `/api/ready` reports PostgreSQL through
the pooled endpoint and `retrieval.backend: pgvector`; a review paused before a
Space restart was still in the history and review queue after it, was approved,
and the workflow completed.

**At Neon's free limits** (100 CU-hours/month or 1 GB): Neon suspends compute or
refuses writes — it never bills on the Free plan. NeuroClass then answers 503
"database temporarily unavailable" for history, accounts and new analyses;
literature search and research answers keep working from the in-memory index;
benchmarks and samples need no database; the LLM keeps its in-process quota
guard. Everything resumes when Neon does. Removing `DATABASE_URL` switches to
temporary SQLite.

## 5. Local and Docker

```bash
docker compose -f deploy/docker-compose.yml up --build
```
Web on http://localhost:3000, API docs on http://localhost:8000/docs, Postgres
with pgvector inside the stack. Verified 2026-11-17: images 1.49 GB (API) and
406 MB (web); all 12 browser journeys pass against it; a review paused before an
API restart resumes and completes after it.

Without Docker: see [README → Quickstart](../README.md#quickstart).

## 6. Maintenance (all free)

| Task | How often | How |
|---|---|---|
| Watch CI | every push | GitHub Actions: research, platform (Postgres), space (HF dependency set), web |
| Dependency advisories | monthly | `pip-audit -r requirements-app.txt`, `npm audit --omit=dev` (both run in CI) |
| Re-verify free-tier terms | quarterly | re-read the pages linked in `FREE_TIER.md`; update the dates there |
| Gemini model retirement | when Google announces it | change `NEUROCLASS_LLM_MODEL`; the deprecation page lists dates |
| Refresh the literature corpus | as needed | `python -m neuroclass.rag.corpus fetch && ... build`, rebuild the index, rerun `scripts/eval_rag.py`, redeploy the Space |
| New model | when a candidate exists | `python -m neuroclass.registry gate` must pass; export with `btc.export_serving`; redeploy |
| Space asleep | after ~48 h idle | nothing: the next visitor wakes it (about a minute) |

Nothing in this list requires a paid plan. If a provider changes its free tier
so that a payment method becomes mandatory, remove that dependency: every one
of them is optional except the Space.
