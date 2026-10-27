"""Gradio web app — NeuroClass: brain tumor MRI classifier (EfficientNetB3).

Deployed on Hugging Face Spaces. For an uploaded MRI it returns:
  * predicted tumor class + temperature-calibrated confidence
  * full per-class probabilities
  * Grad-CAM attribution overlay
  * a plain-language stability level from 6 test-time-augmentation views
  * ONNX-Runtime CPU inference latency
  * an explicit not-a-medical-device disclaimer

Inference (class + confidence + uncertainty) runs on ONNX Runtime for efficient
CPU serving; Grad-CAM uses the Keras model. Weights are pulled from the HF Hub
model repo at startup. Preprocessing and all other model-free logic live in
logic.py (unit-tested; identical to btc.preprocess): 256x256 RGB, raw [0,255]
pixels (EfficientNet rescales internally).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download

import logic

MODEL_REPO = "Udit013/brain-tumor-efficientnetb3"
# Which model files to serve. Defaults are the deployed parity model; set these
# Space variables to switch models (and unset them to roll back) without a code
# change. All three must come from the same model.
ONNX_FILE = os.environ.get("NEUROCLASS_ONNX", "efficientnetb3.onnx")
WEIGHTS_FILE = os.environ.get("NEUROCLASS_WEIGHTS", "EfficientNetB3_model_weights.h5")
TEMPERATURE_FILE = os.environ.get("NEUROCLASS_TEMPERATURE", "temperature.json")
CLASS_NAMES = logic.CLASS_NAMES

# --------------------------------------------------------------------------- #
# Load artifacts once at startup
# --------------------------------------------------------------------------- #
print(f"Serving {ONNX_FILE} / {WEIGHTS_FILE} / {TEMPERATURE_FILE} from {MODEL_REPO}")
onnx_path = hf_hub_download(MODEL_REPO, ONNX_FILE)
weights_path = hf_hub_download(MODEL_REPO, WEIGHTS_FILE)
try:
    temp_path = hf_hub_download(MODEL_REPO, TEMPERATURE_FILE)
    with open(temp_path) as f:
        TEMPERATURE = float(json.load(f)["temperature"])
except Exception as exc:  # noqa: BLE001 — any failure here is non-fatal
    print(f"WARNING: could not load calibration temperature, defaulting to "
          f"uncalibrated (T=1.0): {exc}")
    TEMPERATURE = 1.0

import onnxruntime as ort  # noqa: E402 — after the downloads it needs

_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
_in_name = _sess.get_inputs()[0].name


def _onnx_predict(batch: np.ndarray) -> np.ndarray:
    return _sess.run(None, {_in_name: batch.astype("float32")})[0]


# TensorFlow model (Grad-CAM only) ------------------------------------------ #
import tensorflow as tf  # noqa: E402

# Spaces hardware here is CPU-only; pinning makes local runs compute the same
# thing (on Apple Silicon, tensorflow-metal's graph mode computes a different
# function — see btc.config.disable_metal_graph_optimizer).
tf.config.set_visible_devices([], "GPU")
from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # noqa: E402
from tensorflow.keras import regularizers  # noqa: E402


def _build_model():
    base = tf.keras.applications.efficientnet.EfficientNetB3(
        include_top=False, weights=None, input_shape=(256, 256, 3), pooling="max")
    model = Sequential([
        base,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(512, kernel_regularizer=regularizers.l2(0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation="relu"),
        Dropout(0.4, seed=75),
        Dense(256, kernel_regularizer=regularizers.l2(0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation="relu"),
        Dropout(0.2, seed=75),
        Dense(4, activation="softmax"),
    ])
    return model


_tf_model = _build_model()
_tf_model.load_weights(weights_path)
_base = _tf_model.get_layer("efficientnetb3")
_last_conv = next(layer.name for layer in reversed(_base.layers)
                  if len(getattr(layer, "output_shape", ())) == 4)
_base_grad = tf.keras.models.Model(
    inputs=_base.input, outputs=[_base.get_layer(_last_conv).output, _base.output])
_head_layers = _tf_model.layers[1:]


# Compiled once (fixed input signature). ~9x faster than running the same
# forward+backward pass eagerly (232 -> 26 ms per image on a laptop CPU) with
# bit-identical heatmaps — Grad-CAM was ~85% of request time.
@tf.function(input_signature=[tf.TensorSpec((1, 256, 256, 3), tf.float32),
                              tf.TensorSpec((), tf.int32)])
def _cam_graph(arr, class_idx):
    with tf.GradientTape() as tape:
        conv_out, base_out = _base_grad(arr, training=False)
        tape.watch(conv_out)
        h = base_out
        for layer in _head_layers:
            h = layer(h, training=False)
        loss = tf.gather(h, class_idx, axis=1)
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))
    return cam / (tf.reduce_max(cam) + 1e-9)


def _gradcam(x: np.ndarray, class_idx: int) -> np.ndarray:
    import cv2

    cam = _cam_graph(tf.convert_to_tensor(x[None, ...], dtype=tf.float32),
                     tf.constant(class_idx, tf.int32)).numpy()
    cam = cv2.resize(cam, (x.shape[1], x.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip(x, 0, 255) * 0.6 + heat * 0.4)



EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLES = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg"))

DISCLAIMER = (
    "**Research and education demo — not a medical device.** Do not use it for "
    "diagnosis, triage or any decision about a person's care. It was trained on "
    "a public Kaggle compilation with documented train/test overlap, and its "
    "accuracy on images unlike its training data is far lower than its "
    "benchmark score."
)

UPLOAD_HELP = """
**What to upload:** one 2-D slice of a brain MRI as a JPG or PNG — the kind
of image in the examples below. The model was trained only on such slices and
on four labels (glioma, meningioma, pituitary tumor, no tumor); any other image
(a photo, a CT scan, a different organ) still gets one of those four labels,
and that label is meaningless.

**Privacy:** images are processed in memory to compute the result and are not
saved by this app. Remove any patient-identifying information before uploading.
"""

ABOUT = """
**Model.** EfficientNetB3 fine-tuned on the
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(Masoud Nickparvar, CC BY 4.0), reproducing an
[IEEE ICC-ROBINS 2024 paper](https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941).
Inference runs on ONNX Runtime (CPU); Grad-CAM uses the Keras model.

**Known weaknesses (measured on CPU, see the
[model card](https://github.com/Udit013/Brain_Tumor_Classificatioin/blob/main/MODEL_CARD.md)).**
{weaknesses}
- The heatmap shows which pixels influenced the score, not where a tumor is.

**Example images** are from the dataset's test split (CC BY 4.0).
"""

# Measured test-set facts for each published model (MODEL_CARD.md); the About
# panel must describe the model actually being served.
WEAKNESSES = {
    "efficientnetb3.onnx": """**Model in use:** the original reproduction (parity model).
- It misses gliomas most often: about 1 in 3 glioma test images is labelled
  something else, and about 1 in 12 is labelled *no tumor*.
- Accuracy on test images with no near-copy in the training set is about 10
  points lower than on images that have one (86% vs 96%).
- On a different public MRI dataset it was right only about 70% of the time
  (tumor vs no tumor), finding only half of the tumors.""",
    "neuroclass_v2.onnx": """**Model in use:** NeuroClass v2 (retrained on a leakage-safe split).
- It still misses gliomas most often: about 1 in 5 glioma test images is
  labelled something else, and about 1 in 16 is labelled *no tumor*.
- Accuracy on test images with no near-copy in the training set is about 9
  points lower than on images that have one (89% vs 98%).
- On a different public MRI dataset it was right about 73% of the time
  (tumor vs no tumor), finding just over half of the tumors.""",
}
ABOUT = ABOUT.format(weaknesses=WEAKNESSES.get(
    ONNX_FILE, f"**Model in use:** `{ONNX_FILE}` — see the model card for its measured limitations."))

CSS = """
.nc-disclaimer { border-left: 4px solid #b45309; padding: 10px 14px;
                 border-radius: 6px; background: var(--block-background-fill); }
.nc-result h2 { margin-top: 0; }
"""


def predict(image):
    """Returns (summary_md, probabilities, gradcam_overlay, technical_md)."""
    try:
        x = logic.preprocess(image)
    except logic.InputError as exc:
        raise gr.Error(str(exc), title="Can't use this image") from None
    except Exception as exc:  # noqa: BLE001 — undecodable or unusual data
        raise gr.Error("This file could not be read as an image. Try a JPG or PNG.",
                       title="Can't use this image") from exc

    # A failure past this point is reported to this user, never raised out of
    # the handler, so one bad request cannot take the shared demo down.
    try:
        t0 = time.perf_counter()
        raw = _onnx_predict(x[None, ...])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        cal = logic.apply_temperature(raw, TEMPERATURE)
        cls = int(cal.argmax())
        tta_probs = _onnx_predict(logic.tta_views(x))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: inference failed: {exc!r}")
        raise gr.Error("The model failed on this image. Please try another one.",
                       title="Inference error") from exc

    try:
        overlay = _gradcam(x, cls)
    except Exception as exc:  # noqa: BLE001 — the prediction is still valid
        print(f"WARNING: Grad-CAM failed: {exc!r}")
        overlay = None

    level, reason = logic.stability(tta_probs, cls, float(cal[cls]))
    probs = {logic.DISPLAY_NAMES[c]: float(cal[i]) for i, c in enumerate(CLASS_NAMES)}
    return (logic.summary_markdown(cls, float(cal[cls]), level, reason),
            probs, overlay,
            logic.technical_markdown(raw, cal, TEMPERATURE, tta_probs, latency_ms,
                                     overlay is not None))


EMPTY_SUMMARY = "*Upload an image and press **Classify** to see the result here.*"

with gr.Blocks(title="NeuroClass — Brain Tumor MRI Classifier",
               theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
               css=CSS) as demo:
    gr.Markdown("# NeuroClass\nClassifies a brain MRI slice as glioma, meningioma, "
                "pituitary tumor or no tumor, and shows how stable and how "
                "confident that answer is.")
    gr.Markdown("⚠️ " + DISCLAIMER, elem_classes="nc-disclaimer")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            inp = gr.Image(type="numpy", label="Brain MRI slice (JPG or PNG)",
                           sources=["upload", "clipboard"], height=320)
            with gr.Row():
                clear = gr.ClearButton(value="Clear")
                btn = gr.Button("Classify", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp,
                            label="Or try an example — one per class, in order glioma, "
                                  "meningioma, no tumor, pituitary (picked as the first "
                                  "file of each, not for being classified correctly)")
            with gr.Accordion("What can I upload?", open=False):
                gr.Markdown(UPLOAD_HELP)
        with gr.Column(scale=1, min_width=320):
            out_summary = gr.Markdown(EMPTY_SUMMARY, elem_classes="nc-result")
            out_label = gr.Label(num_top_classes=4, label="Calibrated probabilities")
            out_cam = gr.Image(label="Grad-CAM: red = drove the score (not a tumor outline)",
                               height=320, interactive=False, show_download_button=False)
            with gr.Accordion("Technical details", open=False):
                out_tech = gr.Markdown()

    with gr.Accordion("About the model and its limitations", open=False):
        gr.Markdown(ABOUT)

    outputs = [out_summary, out_label, out_cam, out_tech]
    btn.click(predict, inputs=inp, outputs=outputs, show_progress="full",
              concurrency_limit=2)
    clear.add([inp, out_label, out_cam, out_tech])
    clear.click(lambda: EMPTY_SUMMARY, outputs=out_summary)

if __name__ == "__main__":
    # server_name=0.0.0.0 so the HF Spaces proxy can reach it; show_api=False
    # keeps the demo UI-only.
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
