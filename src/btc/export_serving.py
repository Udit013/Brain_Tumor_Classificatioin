"""Export a TensorFlow-free serving bundle for the deployed model.

The FastAPI service and the Space need TensorFlow only for Grad-CAM, which
makes the API image 3.3 GB. Grad-CAM needs the gradient of the class score
with respect to the last conv feature map, but everything after that map is a
tiny head: global max-pool -> BatchNorm -> Dense(512, relu) -> Dense(256, relu)
-> Dense(4, softmax). Its gradient has a closed form, so the bundle is:

  neuroclass_serving.onnx  inputs  image (N, 256, 256, 3) float32, raw [0, 255]
                           outputs probs (N, 4) and conv (N, 8, 8, 1536), the
                           `top_activation` map Grad-CAM is computed on
  head.npz                 the head's weights, for the analytic gradient
  ood_reference.npz        mean and shrunk precision matrix of training-set
                           pooled features (Mahalanobis input-shift heuristic)
  manifest.json            version, checksums, class order, preprocessing,
                           temperature, OOD threshold and verification results

`verify()` checks the bundle against the TensorFlow model this module exports
from: probabilities and Grad-CAM heatmaps must match within tolerance, or the
export is refused.

Usage (research environment, needs TensorFlow + tf2onnx):
    python -m btc.export_serving [--out models/serving]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import config

SERVING_DIR = config.MODELS_DIR / "serving"
BN_EPSILON = 0.001          # head BatchNormalization epsilon (model.build_model)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head_weights(model) -> dict:
    """Extract the classifier head (layers after the backbone) as arrays."""
    from tensorflow.keras.layers import BatchNormalization, Dense

    bn = next(layer for layer in model.layers if isinstance(layer, BatchNormalization))
    dense = [layer for layer in model.layers if isinstance(layer, Dense)]
    assert len(dense) == 3, "expected the 512 -> 256 -> 4 head"
    gamma, beta, mean, var = (w.astype("float64") for w in bn.get_weights())
    out = {"bn_gamma": gamma, "bn_beta": beta, "bn_mean": mean, "bn_var": var,
           "bn_epsilon": np.float64(bn.epsilon)}
    for i, layer in enumerate(dense, start=1):
        w, b = layer.get_weights()
        out[f"w{i}"], out[f"b{i}"] = w.astype("float64"), b.astype("float64")
    return out


def export_onnx(model, path: Path) -> None:
    """ONNX graph: image -> (softmax probs, top_activation feature map)."""
    import tensorflow as tf
    import tf2onnx

    from .model import find_last_conv_layer_name

    base = model.get_layer("efficientnetb3")
    conv_name = find_last_conv_layer_name(model)
    backbone = tf.keras.models.Model(base.input, [base.get_layer(conv_name).output,
                                                  base.output])
    # A functional model reusing the trained layers. (from_function would turn
    # the Normalization layer's statistics into graph inputs; from_keras keeps
    # them as constants, as in btc.export_onnx.)
    image = tf.keras.Input(config.IMG_SHAPE, name="image")
    conv, h = backbone(image, training=False)
    for layer in model.layers[1:]:
        h = layer(h, training=False)
    probs = tf.keras.layers.Activation("linear", name="probs")(h)
    conv = tf.keras.layers.Activation("linear", name="conv")(conv)
    serving = tf.keras.Model(image, [probs, conv], name="neuroclass_serving")
    spec = (tf.TensorSpec((None, *config.IMG_SHAPE), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(serving, input_signature=spec, opset=13,
                               output_path=str(path))


def _onnx_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_run(session, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = [o.name for o in session.get_outputs()]
    out = dict(zip(names, session.run(None, {session.get_inputs()[0].name: x})))
    probs = next(v for k, v in out.items() if "probs" in k)
    conv = next(v for k, v in out.items() if "conv" in k)
    return probs, conv


def fit_ood_reference(features: np.ndarray) -> dict:
    """Mean + Ledoit-Wolf-shrunk precision of training pooled features."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(features.astype("float64"))
    return {"mean": lw.location_.astype("float32"),
            "precision": lw.precision_.astype("float32"),
            "shrinkage": np.float64(lw.shrinkage_)}


def mahalanobis(features: np.ndarray, ref: dict) -> np.ndarray:
    d = features.astype("float64") - ref["mean"].astype("float64")
    return np.sqrt(np.einsum("ij,jk,ik->i", d, ref["precision"].astype("float64"), d))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the TF-free serving bundle")
    parser.add_argument("--out", type=Path, default=SERVING_DIR)
    parser.add_argument("--verify-n", type=int, default=64)
    args = parser.parse_args()

    from .data import load_test_arrays
    from .explain import _grad_model, gradcam_heatmap
    from .model import find_last_conv_layer_name, load_trained_model
    from .preprocess import load_images
    from .splits import load_split
    from .serving_math import gradcam_from_conv, head_forward

    args.out.mkdir(parents=True, exist_ok=True)
    model = load_trained_model()          # pins CPU
    onnx_path = args.out / "neuroclass_serving.onnx"
    t0 = time.time()
    export_onnx(model, onnx_path)
    head = head_weights(model)
    np.savez(args.out / "head.npz", **head)
    sess = _onnx_session(onnx_path)
    print(f"exported ONNX in {time.time() - t0:.0f}s")

    # --- verification against TensorFlow --------------------------------------
    x_test, y_test, _ = load_test_arrays()
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(x_test), size=min(args.verify_n, len(x_test)), replace=False)
    grad_model = _grad_model(model, find_last_conv_layer_name(model))
    p_tf = model.predict(x_test, batch_size=32, verbose=0)
    p_onnx = np.concatenate([onnx_run(sess, x_test[i:i + 64])[0]
                             for i in range(0, len(x_test), 64)])
    cam_diffs, head_diffs = [], []
    for i in idx:
        probs, conv = onnx_run(sess, x_test[i:i + 1])
        head_diffs.append(float(np.abs(head_forward(conv[0].max(axis=(0, 1)), head)[0]
                                       - probs[0]).max()))
        cam_np = gradcam_from_conv(conv[0], head, int(probs[0].argmax()))
        cam_tf, _ = gradcam_heatmap(x_test[i], model, grad_model)
        cam_diffs.append(float(np.abs(cam_np - cam_tf).max()))
    verification = {
        "n_test_images": int(len(x_test)),
        "probs_max_abs_diff_vs_tensorflow": float(np.abs(p_onnx - p_tf).max()),
        "argmax_disagreements_vs_tensorflow": int((p_onnx.argmax(1) != p_tf.argmax(1)).sum()),
        "test_accuracy_onnx": float((p_onnx.argmax(1) == y_test).mean()),
        "n_gradcam_checked": int(len(idx)),
        "gradcam_max_abs_diff_vs_tensorflow": max(cam_diffs),
        "head_probs_max_abs_diff_vs_onnx": max(head_diffs),
    }
    print(json.dumps(verification, indent=2))
    if (verification["probs_max_abs_diff_vs_tensorflow"] > 1e-4
            or verification["gradcam_max_abs_diff_vs_tensorflow"] > 1e-3
            or verification["head_probs_max_abs_diff_vs_onnx"] > 1e-4):
        raise SystemExit("serving bundle does not reproduce TensorFlow — refusing to write manifest")

    # --- input-shift reference from TRAINING features --------------------------
    split = load_split()

    def pooled(paths):
        feats = []
        for i in range(0, len(paths), 64):
            _, conv = onnx_run(sess, load_images(paths[i:i + 64]))
            feats.append(conv.max(axis=(1, 2)))
        return np.concatenate(feats)

    t1 = time.time()
    f_train = pooled(split["train"][0])
    f_val = pooled(split["val"][0])
    ref = fit_ood_reference(f_train)
    d_val = mahalanobis(f_val, ref)
    threshold = float(np.quantile(d_val, 0.99))   # 1% of clean validation images flagged
    np.savez(args.out / "ood_reference.npz", mean=ref["mean"], precision=ref["precision"])
    print(f"OOD reference from {len(f_train)} training images in {time.time() - t1:.0f}s; "
          f"threshold (val 99th pct) {threshold:.2f}")

    temperature = float(json.loads(config.TEMPERATURE_PATH.read_text())["temperature"])
    files = {p.name: _sha256(p) for p in sorted(args.out.glob("*")) if p.suffix in (".onnx", ".npz")}
    manifest = {
        "model_name": "neuroclass-efficientnetb3",
        "model_version": "2026.10-finetune",
        "source_weights": config.WEIGHTS_PATH.name,
        "source_weights_sha256": _sha256(config.WEIGHTS_PATH),
        "hub_repo": "Udit013/brain-tumor-efficientnetb3",
        "class_names": config.CLASS_NAMES,
        "input": {"shape": list(config.IMG_SHAPE), "dtype": "float32", "range": [0, 255],
                  "preprocessing": "PIL convert('RGB').resize((256, 256)) — btc.preprocess"},
        "temperature": temperature,
        "ood": {"method": "Mahalanobis distance of pooled backbone features to the "
                          "training distribution (Ledoit-Wolf shrinkage)",
                "threshold": threshold, "threshold_rule": "99th percentile on the "
                "leakage-safe validation split", "shrinkage": float(ref["shrinkage"]),
                "n_reference": int(len(f_train))},
        "files": files,
        "verification": verification,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
