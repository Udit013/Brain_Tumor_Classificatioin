"""Serving latency and memory: TensorFlow path vs the TF-free ONNX path.

Each backend runs in its own process so peak memory is attributable:
  tf          Keras predict (1 view) + compiled TensorFlow Grad-CAM   (research env, btc.serve path)
  onnx_1view  ONNX probabilities + conv map, NumPy Grad-CAM           (platform env)
  onnx_tta    ONNX on the image + 5 TTA views, NumPy Grad-CAM         (platform default)

Single-image requests, warm (5 warm-up calls), 48 test images (12 per class), CPU, one
process at a time. Writes results/metrics/serving_benchmark.json when run with
--combine after the per-backend runs.

    .venv/bin/python scripts/bench_serving.py --backend tf --out /tmp/tf.json
    .venv-app/bin/python scripts/bench_serving.py --backend onnx_1view --out /tmp/o1.json
    .venv-app/bin/python scripts/bench_serving.py --backend onnx_tta --out /tmp/o6.json
    python scripts/bench_serving.py --combine /tmp/tf.json /tmp/o1.json /tmp/o6.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def images(n: int):
    from btc import config
    from btc.preprocess import load_image

    paths = [p for c in config.CLASS_NAMES for p in sorted((config.TEST_DIR / c).iterdir())[:n // 4]]
    return [load_image(p) for p in paths]


def run(backend: str, n: int) -> dict:
    import numpy as np

    t_load = time.perf_counter()
    if backend == "tf":
        from btc.explain import _grad_model, gradcam_heatmap
        from btc.model import find_last_conv_layer_name, load_trained_model

        model = load_trained_model()
        gm = _grad_model(model, find_last_conv_layer_name(model))

        def infer(x):
            p = model.predict(x[None], verbose=0)[0]
            gradcam_heatmap(x, model, gm, class_idx=int(p.argmax()))
    else:
        from neuroclass.vision.service import VisionService, tta_views
        from btc.serving_math import gradcam_from_conv

        svc = VisionService(ROOT / "models" / "serving")
        b = svc.ensure_loaded()

        def infer(x):
            batch = tta_views(x) if backend == "onnx_tta" else x[None]
            outs = dict(zip(b.output_names, b.session.run(None, {b.input_name: batch})))
            probs = next(v for k, v in outs.items() if "probs" in k)
            conv = next(v for k, v in outs.items() if "conv" in k)
            gradcam_from_conv(conv[0], b.head, int(probs[0].argmax()))
    load_s = time.perf_counter() - t_load
    xs = images(n)
    for x in xs[:5]:
        infer(x)
    lat = []
    for x in xs:
        t = time.perf_counter()
        infer(x)
        lat.append((time.perf_counter() - t) * 1000)
    return {"backend": backend, "n": len(xs), "load_seconds": round(load_s, 2),
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
            "peak_rss_mb": round(peak_rss_mb(), 0),
            "machine": f"{platform.machine()} {platform.system()}", "python": platform.python_version()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tf", "onnx_1view", "onnx_tta"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--combine", nargs="+", type=Path)
    args = ap.parse_args()
    if args.combine:
        rows = [json.loads(p.read_text()) for p in args.combine]
        import subprocess

        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True).stdout.strip() if sys.platform == "darwin" else platform.processor()
        out = {"note": "single-image requests incl. Grad-CAM, warm, CPU; one process per backend",
               "cpu": cpu, "measured": time.strftime("%Y-%m-%d"),
               "results": {r["backend"]: r for r in rows}}
        (ROOT / "results/metrics/serving_benchmark.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return
    r = run(args.backend, args.n)
    args.out.write_text(json.dumps(r))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
