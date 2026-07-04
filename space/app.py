"""Gradio web app — Brain Tumor MRI Classifier (EfficientNetB3).

Deployed on Hugging Face Spaces. For an uploaded MRI it returns:
  * predicted tumor class + temperature-calibrated confidence
  * full per-class probabilities
  * Grad-CAM attribution overlay
  * TTA-based uncertainty (predictive entropy)
  * ONNX-Runtime CPU inference latency
  * an explicit not-a-medical-device disclaimer

Inference (class + confidence + uncertainty) runs on ONNX Runtime for efficient
CPU serving; Grad-CAM uses the Keras model. Weights are pulled from the HF Hub
model repo at startup. Preprocessing matches the published pipeline exactly:
256x256 RGB, raw [0,255] pixels (EfficientNet rescales internally).
"""

from __future__ import annotations

import json
import time

import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download

MODEL_REPO = "Udit013/brain-tumor-efficientnetb3"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (256, 256)

# --------------------------------------------------------------------------- #
# Load artifacts once at startup
# --------------------------------------------------------------------------- #
onnx_path = hf_hub_download(MODEL_REPO, "efficientnetb3.onnx")
weights_path = hf_hub_download(MODEL_REPO, "EfficientNetB3_model_weights.h5")
try:
    temp_path = hf_hub_download(MODEL_REPO, "temperature.json")
    TEMPERATURE = float(json.load(open(temp_path))["temperature"])
except Exception:
    TEMPERATURE = 1.0

import onnxruntime as ort

_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
_in_name = _sess.get_inputs()[0].name


def _onnx_predict(batch: np.ndarray) -> np.ndarray:
    return _sess.run(None, {_in_name: batch.astype("float32")})[0]


# TensorFlow model (Grad-CAM only) ------------------------------------------ #
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers


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
_last_conv = next(l.name for l in reversed(_base.layers)
                  if len(getattr(l, "output_shape", ())) == 4)
_base_grad = tf.keras.models.Model(
    inputs=_base.input, outputs=[_base.get_layer(_last_conv).output, _base.output])
_head_layers = _tf_model.layers[1:]


def _apply_temperature(p: np.ndarray) -> np.ndarray:
    if TEMPERATURE == 1.0:
        return p
    logits = np.log(np.clip(p, 1e-12, 1.0)) / TEMPERATURE
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


def _gradcam(x: np.ndarray, class_idx: int) -> np.ndarray:
    import cv2

    arr = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, base_out = _base_grad(arr)
        tape.watch(conv_out)
        h = base_out
        for layer in _head_layers:
            h = layer(h, training=False)
        loss = h[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-9)
    cam = cv2.resize(cam, (x.shape[1], x.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip(x, 0, 255) * 0.6 + heat * 0.4)


def _tta_augment(x: np.ndarray) -> np.ndarray:
    import cv2

    h, w = x.shape[:2]
    views = [x, x[:, ::-1, :]]
    for angle in (-8, 8):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        views.append(cv2.warpAffine(x, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    for f in (0.9, 1.1):
        views.append(np.clip(x * f, 0, 255).astype("float32"))
    return np.stack(views).astype("float32")


DISCLAIMER = (
    "⚠️ **Research/education demo — NOT a medical device.** This model must not "
    "be used for diagnosis or clinical decisions. It is trained on a public "
    "Kaggle compilation with documented train/test leakage; real-world accuracy "
    "on novel patients is materially lower than benchmark numbers."
)


def predict(image):
    if image is None:
        return None, None, "Please upload an MRI image."
    from PIL import Image

    img = Image.fromarray(image).convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype="float32")  # [0,255], parity preprocessing

    t0 = time.perf_counter()
    raw = _onnx_predict(x[None, ...])[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    cal = _apply_temperature(raw)
    cls = int(cal.argmax())

    # TTA uncertainty (ONNX, cheap)
    views = _tta_augment(x)
    tta_probs = _onnx_predict(views)
    mean_p = tta_probs.mean(axis=0)
    entropy = float(-(mean_p * np.log(np.clip(mean_p, 1e-12, 1.0))).sum())
    std_top = float(tta_probs[:, cls].std())

    overlay = _gradcam(x, cls)
    probs_dict = {CLASS_NAMES[i]: float(cal[i]) for i in range(4)}

    info = (
        f"### Prediction: **{CLASS_NAMES[cls]}**\n"
        f"- **Calibrated confidence:** {cal[cls]*100:.1f}%  "
        f"(raw {raw[cls]*100:.1f}%, temperature T={TEMPERATURE:.3f})\n"
        f"- **Uncertainty:** predictive entropy {entropy:.3f} "
        f"(0 = certain, {np.log(4):.2f} = max); TTA std {std_top:.3f}\n"
        f"- **Inference latency (ONNX-CPU):** {latency_ms:.1f} ms\n\n"
        f"{DISCLAIMER}"
    )
    return probs_dict, overlay, info


with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
    gr.Markdown(
        "# 🧠 Brain Tumor MRI Classifier (EfficientNetB3)\n"
        "Upload a T1-weighted brain MRI to get the predicted tumor class, "
        "calibrated confidence, uncertainty, and a Grad-CAM attribution map. "
        "Extends an IEEE 2024 publication "
        "([DOI](https://doi.org/10.1109/ICC-ROBINS60238.2024.10533941)) with "
        "production evaluation, calibration, explainability and ONNX serving. "
        "[Code](https://github.com/Udit013/Brain_Tumor_Classificatioin)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="MRI image", height=300)
            btn = gr.Button("Classify", variant="primary")
            gr.Markdown("Classes: glioma · meningioma · notumor · pituitary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=4, label="Calibrated probabilities")
            out_cam = gr.Image(label="Grad-CAM attribution", height=300)
    out_info = gr.Markdown()
    btn.click(predict, inputs=inp, outputs=[out_label, out_cam, out_info])

if __name__ == "__main__":
    # server_name=0.0.0.0 so the HF Spaces proxy can reach it; show_api=False
    # avoids a gradio 4.44 client schema bug in api_info generation.
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
