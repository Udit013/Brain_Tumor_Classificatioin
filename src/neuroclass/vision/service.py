"""Vision inference on the TensorFlow-free serving bundle.

One ONNX call scores the image and its 5 test-time-augmentation views; Grad-CAM
is computed analytically from the identity view's conv map
(`btc.serving_math`), and the pooled features give the input-shift distance.
Preprocessing is `btc.preprocess.to_model_input`, the function every reported
metric was measured with.
"""

from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from btc.preprocess import to_model_input
from btc.serving_math import gradcam_from_conv

from ..schemas import CLASS_NAMES, ClassProbability, InferenceResult, InputQuality, ModelInfo


class ModelUnavailable(RuntimeError):
    """The bundle is missing, incomplete or fails its checksum."""


class InferenceTimeout(RuntimeError):
    pass


def tta_views(x: np.ndarray) -> np.ndarray:
    """identity, flip, +/-8 degree rotation, x0.9 / x1.1 brightness.

    Same views as space/logic.tta_views (asserted equal in tests).
    """
    import cv2

    h, w = x.shape[:2]
    views = [x, x[:, ::-1, :]]
    for angle in (-8, 8):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        views.append(cv2.warpAffine(x, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    for f in (0.9, 1.1):
        views.append(np.clip(x * f, 0, 255).astype("float32"))
    return np.stack(views).astype("float32")


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def overlay_png(x: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> bytes:
    """Same blend as btc.explain.overlay, encoded as PNG."""
    import cv2

    hm = cv2.resize(heatmap, (x.shape[1], x.shape[0]))
    hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    blended = np.uint8(np.clip(x, 0, 255) * (1 - alpha) + hm * alpha)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


@dataclass
class _Bundle:
    session: object
    head: dict
    ood_mean: np.ndarray
    ood_precision: np.ndarray
    manifest: dict
    input_name: str
    output_names: list[str]


class VisionService:
    """Thread-safe inference over a verified serving bundle."""

    def __init__(self, model_dir: Path, timeout_s: float = 20.0, threads: int = 2):
        self.model_dir = Path(model_dir)
        self.timeout_s = timeout_s
        self._bundle: _Bundle | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self._pool = cf.ThreadPoolExecutor(max_workers=threads, thread_name_prefix="vision")

    # ------------------------------------------------------------------ load
    def _load(self) -> _Bundle:
        import onnxruntime as ort

        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.exists():
            raise ModelUnavailable(f"no serving manifest in {self.model_dir}")
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["files"].items():
            path = self.model_dir / name
            if not path.exists():
                raise ModelUnavailable(f"missing bundle file {name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ModelUnavailable(f"checksum mismatch for {name}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(str(self.model_dir / "neuroclass_serving.onnx"), opts,
                                       providers=["CPUExecutionProvider"])
        head = {k: v for k, v in np.load(self.model_dir / "head.npz").items()}
        ood = np.load(self.model_dir / "ood_reference.npz")
        return _Bundle(session=session, head=head, ood_mean=ood["mean"].astype("float64"),
                       ood_precision=ood["precision"].astype("float64"), manifest=manifest,
                       input_name=session.get_inputs()[0].name,
                       output_names=[o.name for o in session.get_outputs()])

    def ensure_loaded(self) -> _Bundle:
        with self._lock:
            if self._bundle is None:
                try:
                    self._bundle = self._load()
                    self._load_error = None
                except Exception as exc:  # noqa: BLE001 — surfaced through health/503
                    self._load_error = f"{type(exc).__name__}: {exc}"
                    raise ModelUnavailable(self._load_error) from exc
            return self._bundle

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def model_info(self) -> ModelInfo:
        m = self.ensure_loaded().manifest
        return ModelInfo(model_name=m["model_name"], model_version=m["model_version"],
                         source_weights_sha256=m["source_weights_sha256"],
                         onnx_sha256=m["files"]["neuroclass_serving.onnx"],
                         hub_repo=m["hub_repo"], temperature=m["temperature"],
                         class_names=m["class_names"],
                         preprocessing=m["input"]["preprocessing"],
                         ood_threshold=m["ood"]["threshold"], ood_method=m["ood"]["method"])

    # ------------------------------------------------------------- inference
    def _run(self, x: np.ndarray, quality: InputQuality, with_gradcam: bool) -> InferenceResult:
        b = self.ensure_loaded()
        t0 = time.perf_counter()
        views = tta_views(x)
        outs = dict(zip(b.output_names, b.session.run(None, {b.input_name: views})))
        probs = next(v for k, v in outs.items() if "probs" in k)
        conv = next(v for k, v in outs.items() if "conv" in k)
        t_onnx = time.perf_counter()

        temperature = float(b.manifest["temperature"])
        raw = probs[0].astype("float64")
        cal = apply_temperature(raw, temperature)
        cls = int(cal.argmax())
        tta_cal = apply_temperature(probs.astype("float64"), temperature)
        mean_tta = tta_cal.mean(axis=0)
        entropy = float(-(mean_tta * np.log(np.clip(mean_tta, 1e-12, 1))).sum())
        agree = int((tta_cal.argmax(axis=1) == cls).sum())

        pooled = conv[0].max(axis=(0, 1)).astype("float64")
        d = pooled - b.ood_mean
        ood_distance = float(np.sqrt(d @ b.ood_precision @ d))
        threshold = float(b.manifest["ood"]["threshold"])

        gradcam = None
        t_cam0 = time.perf_counter()
        if with_gradcam:
            heatmap = gradcam_from_conv(conv[0], b.head, cls)
            gradcam = base64.b64encode(overlay_png(x, heatmap)).decode("ascii")
        t_end = time.perf_counter()

        return InferenceResult(
            model_version=b.manifest["model_version"],
            predicted_class=CLASS_NAMES[cls],
            calibrated_confidence=float(cal[cls]), raw_confidence=float(raw[cls]),
            probabilities=[ClassProbability(label=c, calibrated=float(cal[i]), raw=float(raw[i]))
                           for i, c in enumerate(CLASS_NAMES)],
            tta_agreement=agree, tta_views=len(views), tta_entropy=entropy,
            ood_distance=ood_distance, ood_threshold=threshold,
            ood_flag=ood_distance > threshold, input_quality=quality,
            gradcam_png_base64=gradcam,
            latency_ms={"model": round((t_onnx - t0) * 1000, 2),
                        "gradcam": round((t_end - t_cam0) * 1000, 2),
                        "total": round((t_end - t0) * 1000, 2)})

    def predict(self, img: Image.Image, quality: InputQuality,
                with_gradcam: bool = True) -> InferenceResult:
        """Score a decoded image, bounded by the configured timeout."""
        x = to_model_input(img)
        future = self._pool.submit(self._run, x, quality, with_gradcam)
        try:
            return future.result(timeout=self.timeout_s)
        except cf.TimeoutError as exc:
            future.cancel()
            raise InferenceTimeout(f"inference exceeded {self.timeout_s:.0f}s") from exc
