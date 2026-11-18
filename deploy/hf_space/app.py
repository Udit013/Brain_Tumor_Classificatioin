"""NeuroClass API on a free Hugging Face Space (CPU Basic).

The Space is the platform's Python backend only:
  /api   the NeuroClass platform API (FastAPI): ONNX inference with analytic
         Grad-CAM, LangGraph analysis workflow with human review, literature
         retrieval, reports, benchmarks. The Next.js app on Vercel calls it.
  /      a minimal landing page pointing people to the web app and API docs.

The Space keeps the Gradio SDK only because free accounts can't create Docker
Spaces; nothing here imports or serves Gradio. Uvicorn binds port 7860 itself.

Configuration (Space settings; see docs/DEPLOYMENT.md):
  DATABASE_URL (secret)     Neon Postgres: history, reviews, checkpoints, pgvector
  GEMINI_API_KEY (secret)   Gemini free tier; without it explanations are extractive
  NEUROCLASS_JWT_SECRET, NEUROCLASS_REVIEWER_CODE (secrets), NEUROCLASS_* variables
"""

from __future__ import annotations

import html
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
os.environ.setdefault("NEUROCLASS_DATA_DIR", "/tmp/neuroclass")
os.environ.setdefault("NEUROCLASS_EMBED_CACHE", "/tmp/fastembed_cache")
os.environ.setdefault("NEUROCLASS_ENV", "production")
os.environ.setdefault("NEUROCLASS_WORKERS", "2")
os.environ.setdefault("NEUROCLASS_SYNC_INDEX_TO_DB", "true")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402

from neuroclass.api.app import create_app  # noqa: E402
from neuroclass.settings import get_settings  # noqa: E402

settings = get_settings()
api = create_app(settings)
WEB_URL = os.environ.get("NEUROCLASS_WEB_URL", "").rstrip("/")


def _warm():
    try:
        api.state.services.vision.ensure_loaded()
        api.state.services.retriever()          # syncs the index to pgvector on first start
    except Exception:  # noqa: BLE001 — /api/ready reports what failed
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Mounted sub-apps don't receive lifespan events; run the API's here.
    async with api.router.lifespan_context(api):
        threading.Thread(target=_warm, daemon=True).start()
        yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

LANDING = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroClass API</title>
<meta name="description" content="Backend API of the NeuroClass research platform.">
<style>
  :root { --bg:#fbfbfa; --fg:#16181d; --muted:#5d6370; --line:#e4e5e9; --card:#fff;
          --accent:#2a5bd7; --accent-fg:#fff; --warn-bg:#fdf6e7; --warn-line:#e9c46a; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0e1013; --fg:#eceef2; --muted:#9aa1ad; --line:#262a31; --card:#15181d;
            --accent:#7aa2ff; --accent-fg:#0b1020; --warn-bg:#211c10; --warn-line:#7a6326; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 560px; margin: 0 auto; padding: 56px 20px; }
  a { color: var(--accent); }
  .mark { display:inline-grid; place-items:center; width:32px; height:32px; border-radius:8px;
          background:var(--fg); color:var(--bg); font-weight:700; font-size:12px; letter-spacing:.02em; }
  h1 { font-size: 22px; margin: 18px 0 6px; letter-spacing: -0.01em; }
  p { margin: 0 0 14px; color: var(--muted); }
  .actions { display:flex; gap:10px; flex-wrap:wrap; margin: 22px 0 28px; }
  a.btn { display:inline-flex; align-items:center; height:38px; padding:0 16px; border-radius:8px;
          font-weight:600; text-decoration:none; border:1px solid var(--line); color:var(--fg);
          background:var(--card); }
  a.btn.primary { background:var(--accent); border-color:var(--accent); color:var(--accent-fg); }
  a.btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:6px 16px; }
  .row { display:flex; justify-content:space-between; gap:12px; padding:9px 0; font-size:14px; }
  .row + .row { border-top:1px solid var(--line); }
  .k { color:var(--muted); }
  .note { margin-top:20px; padding:12px 14px; border-left:3px solid var(--warn-line);
          background:var(--warn-bg); border-radius:6px; font-size:13.5px; color:var(--fg); }
</style>
</head>
<body>
<main>
  <span class="mark" aria-hidden="true">NC</span>
  <h1>NeuroClass API</h1>
  <p>This Space hosts the backend of the NeuroClass research platform: model inference with
     Grad-CAM, the analysis workflow, literature retrieval and reports. The product itself is
     the web app.</p>
  <div class="actions">
    __PRIMARY__
    <a class="btn" href="/api/docs">API reference</a>
  </div>
  <div class="card" aria-live="polite">
    <div class="row"><span class="k">Status</span><span id="status">Checking…</span></div>
    <div class="row"><span class="k">Model</span><span id="model">—</span></div>
    <div class="row"><span class="k">Database</span><span id="db">—</span></div>
    <div class="row"><span class="k">Explanations</span><span id="llm">—</span></div>
  </div>
  <p class="note">Research and education only. Outputs are an experimental model's predictions,
     not diagnoses, and must not inform anyone's care.</p>
  <p style="font-size:13px">Source: <a href="https://github.com/Udit013/Brain_Tumor_Classificatioin" target="_blank" rel="noopener">GitHub</a>
     · Paper: <a href="https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941" target="_blank" rel="noopener">IEEE ICC-ROBINS 2024</a></p>
</main>
<script>
fetch("/api/ready").then(r => r.json()).then(d => {
  const c = d.checks || {};
  document.getElementById("status").textContent = d.status === "ready" ? "Operational" : "Degraded";
  document.getElementById("model").textContent = (c.vision && c.vision.model_version) || "unavailable";
  document.getElementById("db").textContent = c.database && c.database.ok
    ? (c.database.backend === "postgresql" ? "PostgreSQL" : "SQLite (temporary)") : "unavailable";
  document.getElementById("llm").textContent = c.llm && c.llm.mode === "llm"
    ? "LLM-drafted, citation-verified" : "Quoted passages, citation-verified";
}).catch(() => { document.getElementById("status").textContent = "Starting up…"; });
</script>
</body>
</html>
"""


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def landing() -> str:
    primary = (f'<a class="btn primary" href="{html.escape(WEB_URL)}" target="_blank" rel="noopener">'
               'Open NeuroClass</a>'
               if WEB_URL else "")
    return LANDING.replace("__PRIMARY__", primary)


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/api", include_in_schema=False)
def api_root():
    return RedirectResponse("/api/docs")


app.mount("/api", api)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), proxy_headers=True,
                forwarded_allow_ips="*")
