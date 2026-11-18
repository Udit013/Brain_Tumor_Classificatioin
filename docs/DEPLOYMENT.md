# Deploying and maintaining NeuroClass for $0

Live deployment (verified 2026-11-17):

| Surface | URL | Host / plan |
|---|---|---|
| Web app | https://neuroclass-gilt.vercel.app | Vercel Hobby |
| API (docs at `/api/docs`) | https://udit013-brain-tumor-mri-classifier.hf.space/api | Hugging Face Space, CPU Basic |
| Demo | https://huggingface.co/spaces/Udit013/brain-tumor-mri-classifier | same Space, path `/` |

Read [`FREE_TIER.md`](FREE_TIER.md) first: it lists every limit and what happens
at it. The rule for every step below is **never add a payment method, never
upgrade a plan, never enable a paid add-on**. If a screen asks for one, stop:
the step is not needed.

```
Browser ──► Vercel (Next.js, static/PPR) ──fetch──► HF Space :7860
                                                     ├─ /      Gradio demo
                                                     └─ /api   FastAPI platform
                                                              ├─ ONNX model + NumPy Grad-CAM
                                                              ├─ LangGraph workflow (checkpoints)
                                                              ├─ literature index (in-process)
                                                              ├─ SQLite on /tmp  ─or─  Neon Postgres (optional)
                                                              └─ Gemini free tier (optional) ─or─ extractive
```

## 1. Backend: the Hugging Face Space

Free accounts can no longer *create* Gradio or Docker Spaces (that needs PRO),
so NeuroClass runs on an existing Gradio Space. If you don't have one, run the
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
| `NEUROCLASS_EPHEMERAL_STORAGE` | variable | `true` (tells the UI history is temporary) |
| `NEUROCLASS_CORS_ORIGINS` | variable | your Vercel URL, e.g. `https://neuroclass-gilt.vercel.app` |
| `NEUROCLASS_WEB_URL` | variable | same URL (linked from the demo) |
| `NEUROCLASS_LLM_PROVIDER` | variable | `auto` |
| `GEMINI_API_KEY` | secret, optional | free-tier key — see section 3 |
| `DATABASE_URL` | secret, optional | Neon connection string — see section 4 |
| `NEUROCLASS_REVIEWER_EMAILS` | variable, optional | comma-separated emails that get the reviewer role on sign-up |

Keep the hardware on **CPU Basic** and do not add persistent storage.

Checks after a deploy:
```bash
curl -s https://udit013-brain-tumor-mri-classifier.hf.space/api/ready
```
`status: ready`, `vision.ok`, `retrieval.ok`, `database.persistent` (false
without Neon), `llm.provider` (`none` or `gemini`), `llm.paid_allowed: false`.

Rollback: the Space repository keeps history and tags. `tf-gradio5-2026-10` is
the last TensorFlow/Gradio-5 version, `parity-2024` the previous model.
```bash
hf download Udit013/brain-tumor-mri-classifier --repo-type space --revision tf-gradio5-2026-10 --local-dir /tmp/space-rollback
```
then upload that folder as above.

## 2. Frontend: Vercel Hobby

```bash
cd web && vercel link --project neuroclass
```
```bash
printf 'https://udit013-brain-tumor-mri-classifier.hf.space/api' | vercel env add NEXT_PUBLIC_API_URL production
```
```bash
vercel deploy --prod
```

`web/vercel.json` pins the framework to Next.js (a project created from the CLI
otherwise gets the "Other" preset and serves a 404). `NEXT_PUBLIC_API_URL` is
read at build time, so change it with a redeploy. Every page is static or
partially prerendered and fetches data in the browser, so the Hobby function
allowance is barely touched.

## 3. Optional: Gemini free tier

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

## 4. Optional: Neon Postgres (durable history and reviews)

1. Create a Neon account and project on the **Free** plan. Do not add a card.
2. In the compute settings, cap autoscaling at **0.25 CU** (stretches the
   100 CU-hours/month) and keep scale-to-zero on.
3. Copy the pooled connection string (`postgresql://...?sslmode=require`) into
   the Space secret `DATABASE_URL`. On start the API runs its migrations
   (creating the `vector` extension and tables) and LangGraph uses the Postgres
   checkpointer, so a paused review survives restarts.
4. Optional: load the literature index into pgvector (~20 MB) from your machine:
   ```bash
   DATABASE_URL='postgresql://...' PYTHONPATH=src .venv-app/bin/python -m neuroclass.rag.store load
   ```
   Without this the Space keeps using its in-process index, which is what the
   retrieval evaluation measured.
5. Set `NEUROCLASS_EPHEMERAL_STORAGE=false` (or remove it).

When Neon's monthly compute runs out, it suspends the database until the next
month; `/api/ready` reports the database as down, analyses fail with a clear
error, and benchmarks and literature search keep working. Remove
`DATABASE_URL` to fall back to temporary SQLite.

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
