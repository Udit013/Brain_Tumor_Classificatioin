"""Module 6 (part A) — ONNX export + inference-latency benchmark.

Converts the trained Keras EfficientNetB3 to ONNX (via tf2onnx) and measures
single-image inference latency for both the Keras model and the ONNX Runtime
session, on CPU and — if a GPU/Metal provider is available — on GPU.

Outputs:
  models/efficientnetb3.onnx
  results/metrics/latency.json

Usage:
    python -m btc.export_onnx
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import config

N_WARMUP = 5
N_RUNS = 50


def export(model=None) -> None:
    import tensorflow as tf
    import tf2onnx

    from .model import load_trained_model

    model = model or load_trained_model()
    spec = (tf.TensorSpec((None, *config.IMG_SHAPE), tf.float32, name="input"),)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13,
        output_path=str(config.ONNX_PATH),
    )
    print(f"Exported ONNX -> {config.ONNX_PATH}")


def _bench(fn, x):
    for _ in range(N_WARMUP):
        fn(x)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "n_runs": N_RUNS,
    }


def benchmark() -> dict:
    import onnxruntime as ort
    import tensorflow as tf

    from .model import load_trained_model

    x = np.random.rand(1, *config.IMG_SHAPE).astype("float32") * 255.0
    result = {"image_shape": list(config.IMG_SHAPE)}

    # Keras (eager) latency
    keras_model = load_trained_model()
    result["keras"] = _bench(lambda a: keras_model(a, training=False), x)

    # ONNX Runtime latency across available providers
    if not config.ONNX_PATH.exists():
        export(keras_model)
    available = ort.get_available_providers()
    result["onnx_available_providers"] = available
    result["onnx"] = {}
    # CPU always; add a GPU provider if present (CUDA or CoreML/Metal on mac).
    providers_to_try = ["CPUExecutionProvider"]
    for gpu_p in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
        if gpu_p in available:
            providers_to_try.append(gpu_p)
    for prov in providers_to_try:
        sess = ort.InferenceSession(str(config.ONNX_PATH), providers=[prov])
        in_name = sess.get_inputs()[0].name
        result["onnx"][prov] = _bench(
            lambda a: sess.run(None, {in_name: a}), x
        )
    return result


def main() -> None:
    config.ensure_dirs()
    export()
    result = benchmark()
    out = config.METRICS_DIR / "latency.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
