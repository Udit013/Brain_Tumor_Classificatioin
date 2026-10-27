"""Hugging Face Space logic (space/logic.py) — runs without TF, ONNX or Gradio."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from btc import config
from btc.preprocess import to_model_input

SPACE = Path(__file__).resolve().parents[1] / "space"
_spec = importlib.util.spec_from_file_location("space_logic", SPACE / "logic.py")
logic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(logic)

EXAMPLES = sorted((SPACE / "examples").glob("*.jpg"))


def test_class_order_matches_package():
    assert logic.CLASS_NAMES == config.CLASS_NAMES
    assert logic.IMG_SIZE == config.IMG_SIZE
    assert set(logic.DISPLAY_NAMES) == set(logic.CLASS_BLURB) == set(config.CLASS_NAMES)


def test_examples_one_per_class():
    assert [p.name.split("__")[0] for p in EXAMPLES] == config.CLASS_NAMES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_preprocess_matches_serving_function(path):
    # Gradio hands the handler an RGB numpy array (image_mode="RGB"); the
    # result must equal what FastAPI and evaluation compute from the file.
    with Image.open(path) as img:
        expected = to_model_input(img)
        gradio_array = np.asarray(img.convert("RGB"))
    got = logic.preprocess(gradio_array)
    assert got.dtype == np.float32 and got.shape == config.IMG_SHAPE
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("shape", [(300, 280), (300, 280, 1), (300, 280, 4)])
def test_preprocess_accepts_gray_and_alpha(shape):
    x = logic.preprocess(np.full(shape, 120, dtype=np.uint8))
    assert x.shape == (256, 256, 3) and 0 <= x.min() <= x.max() <= 255


@pytest.mark.parametrize("bad, msg", [
    (None, "upload"),
    (np.zeros((10, 300, 3), np.uint8), "too small"),
    (np.zeros((64, 64, 2), np.uint8), "Unsupported"),
    (np.zeros((4, 64, 64, 3), np.uint8), "Unsupported"),
])
def test_preprocess_rejects_unusable_input(bad, msg):
    with pytest.raises(logic.InputError, match=msg):
        logic.preprocess(bad)


def test_preprocess_clips_non_uint8():
    x = logic.preprocess(np.full((64, 64, 3), 400.0))
    assert x.max() == 255


def test_temperature_scaling():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    np.testing.assert_array_equal(logic.apply_temperature(p, 1.0), p)
    sharp = logic.apply_temperature(p, 0.5)
    soft = logic.apply_temperature(p, 2.0)
    for q in (sharp, soft):
        assert q.sum() == pytest.approx(1.0) and q.argmax() == 0
    assert sharp[0] > p[0] > soft[0]


def test_tta_views():
    x = np.random.default_rng(0).uniform(0, 255, (256, 256, 3)).astype("float32")
    v = logic.tta_views(x)
    assert v.shape == (logic.N_TTA_VIEWS, 256, 256, 3) and v.dtype == np.float32
    np.testing.assert_array_equal(v[0], x)
    np.testing.assert_array_equal(v[1], x[:, ::-1])
    assert v.min() >= 0 and v.max() <= 255


def _views(argmaxes):
    p = np.full((len(argmaxes), 4), 0.1)
    p[np.arange(len(argmaxes)), argmaxes] = 0.7
    return p


@pytest.mark.parametrize("argmaxes, conf, level", [
    ([0] * 6, 0.95, "High"),
    ([0] * 6, 0.80, "Moderate"),          # unanimous but not confident
    ([0] * 5 + [1], 0.95, "Moderate"),
    ([0] * 3 + [1] * 3, 0.95, "Low"),     # half the views disagree
    ([0] * 6, 0.55, "Low"),
])
def test_stability_levels(argmaxes, conf, level):
    got, reason = logic.stability(_views(argmaxes), 0, conf)
    assert got == level
    assert f"{argmaxes.count(0)} of 6" in reason


def test_summary_text():
    md = logic.summary_markdown(2, 0.87, "Moderate", "5 of 6 ...")
    assert "No tumor" in md and "87%" in md and "unreliable" in md
    assert "unreliable" not in logic.summary_markdown(0, 0.99, "High", "6 of 6")


def test_technical_text():
    raw = np.array([0.4, 0.3, 0.2, 0.1])
    md = logic.technical_markdown(raw, raw, 0.561, _views([0] * 6), 31.0, cam_ok=False)
    assert md.count("\n|") >= 5 and "0.561" in md and "Grad-CAM could not" in md
