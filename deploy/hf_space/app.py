"""NeuroClass on a free Hugging Face Space (CPU Basic, Gradio SDK).

One process, one port (7860), two surfaces:
  /      a small Gradio demo (upload -> class, calibrated confidence, Grad-CAM,
         reliability level), for people who arrive at the Space itself
  /api   the full NeuroClass platform API (FastAPI): agent workflow, research
         retrieval, reports, review queue. The Next.js app on Vercel calls this.

Both use the same TF-free vision service (ONNX Runtime + NumPy Grad-CAM) and
the same deterministic reliability rules, so they never disagree.

Zero-cost facts this file relies on (see docs/FREE_TIER.md):
- CPU Basic is free, sleeps after ~48 h without traffic, and its disk is not
  persistent. Without DATABASE_URL, history lives in SQLite under /tmp and
  resets when the Space restarts; /api/ready reports this as persistent=false.
- DATABASE_URL (optional secret) points at a free Neon Postgres project.
- GEMINI_API_KEY (optional secret) enables the free-tier LLM; without it every
  explanation is extractive. No paid provider is configured here.
"""

from __future__ import annotations

import base64
import io
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

import gradio as gr  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from PIL import Image  # noqa: E402

from neuroclass.api.app import create_app  # noqa: E402
from neuroclass.reliability import assess  # noqa: E402
from neuroclass.agent.compose import DISPLAY  # noqa: E402
from neuroclass.settings import get_settings  # noqa: E402
from neuroclass.vision.validation import InputRejected, assess_quality, decode_image  # noqa: E402

settings = get_settings()
api = create_app(settings)
WEB_URL = os.environ.get("NEUROCLASS_WEB_URL", "").rstrip("/")
EXAMPLES = sorted(str(p) for p in (HERE / "space" / "examples").glob("*.jpg"))

DISCLAIMER = ("**Research and education demo — not a medical device.** Do not use it for diagnosis, "
              "triage or any decision about a person's care. Accuracy on images unlike the "
              "training data is far lower than the benchmark score.")
LEVEL_ICON = {"standard": "●", "caution": "▲", "unreliable": "■"}


def _services():
    return api.state.services


def predict(path: str | None):
    if not path:
        raise gr.Error("Upload an image or pick an example first.")
    raw = Path(path).read_bytes()
    try:
        img = decode_image(raw, settings.max_upload_bytes, settings.max_image_pixels)
    except InputRejected as exc:
        raise gr.Error(str(exc), title="Can't use this image") from None
    try:
        result = _services().vision.predict(img, assess_quality(img))
    except Exception as exc:  # noqa: BLE001 — report to this user, keep the demo up
        raise gr.Error("The model could not score this image. Try another one.") from exc
    rel = assess(result)
    cls = DISPLAY[result.predicted_class].capitalize()
    summary = (f"## Model-predicted class: {cls}\n"
               f"Calibrated confidence **{result.calibrated_confidence:.1%}** · "
               f"reliability **{LEVEL_ICON[rel.level]} {rel.level}**"
               + (" · *would be sent to human review on the platform*" if rel.requires_review else ""))
    findings = "\n".join(f"- **{getattr(f.severity, 'value', f.severity)}** — {f.message}" for f in rel.findings) or \
        "- No reliability rule fired for this image."
    probs = {DISPLAY[p.label].capitalize(): p.calibrated for p in result.probabilities}
    cam = (Image.open(io.BytesIO(base64.b64decode(result.gradcam_png_base64)))
           if result.gradcam_png_base64 else None)
    tech = (f"- Model `{result.model_version}`, ONNX Runtime on CPU, analytic Grad-CAM\n"
            f"- Test-time augmentation: {result.tta_agreement}/{result.tta_views} views agree\n"
            f"- Input-shift distance {result.ood_distance:.1f} (flag above {result.ood_threshold:.1f})\n"
            f"- Latency {result.latency_ms.get('total', 0):.0f} ms\n"
            f"- Reliability rules `{rel.rules_version}`")
    return summary, probs, cam, findings, tech


with gr.Blocks(title="NeuroClass — Brain Tumor MRI Classifier") as demo:
    gr.Markdown("# NeuroClass\nClassifies a brain MRI slice as glioma, meningioma, pituitary tumor "
                "or no tumor, with a calibrated confidence, a Grad-CAM heatmap and a reliability "
                "level from measured rules.")
    gr.Markdown("⚠️ " + DISCLAIMER)
    links = ["[API docs](/api/docs)", "[Source code](https://github.com/Udit013/Brain_Tumor_Classificatioin)",
             "[IEEE 2024 paper](https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941)"]
    if WEB_URL:
        links.insert(0, f"**[Open the full research platform]({WEB_URL})** (agent workflow, "
                        "cited literature, reports, history)")
    gr.Markdown(" · ".join(links))
    with gr.Row(equal_height=False):
        with gr.Column(min_width=320):
            inp = gr.Image(type="filepath", label="Brain MRI slice (JPG or PNG)",
                           sources=["upload", "clipboard"], height=320)
            btn = gr.Button("Classify", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp, label="Examples (test split, CC BY 4.0)")
            gr.Markdown("Images are processed in memory for this result and not stored by the demo.")
        with gr.Column(min_width=320):
            out_summary = gr.Markdown("*Upload an image and press **Classify**.*")
            out_label = gr.Label(num_top_classes=4, label="Calibrated probabilities")
            out_cam = gr.Image(label="Grad-CAM: where the score came from (not a tumor outline)",
                               height=320, interactive=False)
            with gr.Accordion("Reliability findings", open=True):
                out_findings = gr.Markdown()
            with gr.Accordion("Technical details", open=False):
                out_tech = gr.Markdown()
    btn.click(predict, inputs=inp, outputs=[out_summary, out_label, out_cam, out_findings, out_tech],
              concurrency_limit=2)


def _warm():
    try:
        _services().vision.ensure_loaded()
        _services().retriever()                       # loads the index and embedding model
    except Exception:  # noqa: BLE001 — /api/ready reports what failed
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Mounted sub-apps don't get lifespan events; run the API's here.
    async with api.router.lifespan_context(api):
        threading.Thread(target=_warm, daemon=True).start()
        yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/api", include_in_schema=False)
def api_root():
    return RedirectResponse("/api/docs")


app.mount("/api", api)
# ssr_mode=False: HF sets GRADIO_SSR_MODE=true for Gradio Spaces, which would make
# Gradio start its Node SSR server on port 7860, the port uvicorn serves.
app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), proxy_headers=True,
                forwarded_allow_ips="*")
