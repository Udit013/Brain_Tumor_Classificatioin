"""Fast unit tests for pure-logic components (no TensorFlow, data, or weights).

These run in CI on CPU in seconds. Model/data-dependent behaviour is validated
separately by scripts/reproduce.sh (which needs the dataset + trained weights).
"""

import numpy as np
import pytest

from btc import config
from btc import calibration, robustness, uncertainty, leakage


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_class_names_and_shape():
    assert config.CLASS_NAMES == ["glioma", "meningioma", "notumor", "pituitary"]
    assert config.NUM_CLASSES == 4
    assert config.IMG_SHAPE == (256, 256, 3)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_apply_temperature_is_a_distribution():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=50)
    out = calibration.apply_temperature(probs, 0.8)
    assert out.shape == probs.shape
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_temperature_one_is_identity():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=20)
    np.testing.assert_allclose(calibration.apply_temperature(probs, 1.0), probs, atol=1e-6)


def test_ece_zero_for_perfect_calibration():
    # confidence == accuracy in every bin -> ECE ~ 0
    y = np.array([0, 1, 2, 3] * 25)
    probs = np.zeros((100, 4))
    probs[np.arange(100), y] = 1.0
    ece, _ = calibration.expected_calibration_error(probs, y)
    assert ece < 1e-6


def test_fit_temperature_positive():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, size=200)
    logits = rng.normal(size=(200, 4))
    logits[np.arange(200), y] += 2.0  # make correct class likely
    e = np.exp(logits)
    probs = e / e.sum(1, keepdims=True)
    T = calibration._fit_temperature(probs, y)
    assert T > 0


# --------------------------------------------------------------------------- #
# robustness corruptions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", robustness.CORRUPTIONS)
@pytest.mark.parametrize("sev", [1, 3, 5])
def test_corruption_preserves_shape_and_range(kind, sev):
    x = (np.random.rand(256, 256, 3) * 255).astype("float32")
    out = robustness._corrupt(x, kind, sev)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 255.5


# --------------------------------------------------------------------------- #
# uncertainty TTA
# --------------------------------------------------------------------------- #
def test_tta_augmentations():
    x = (np.random.rand(256, 256, 3) * 255).astype("float32")
    views = uncertainty._augmentations(x)
    assert len(views) >= 4
    assert all(v.shape == x.shape for v in views)


def test_tta_predict_with_stub():
    x = (np.random.rand(256, 256, 3) * 255).astype("float32")

    def stub(batch):  # deterministic fake model -> always class 2
        p = np.zeros((len(batch), 4)); p[:, 2] = 1.0
        return p

    r = uncertainty.tta_predict(x, stub)
    assert r["class_index"] == 2
    assert r["predictive_entropy"] < 1e-6
    assert r["n_augmentations"] >= 4


# --------------------------------------------------------------------------- #
# leakage hashing
# --------------------------------------------------------------------------- #
def test_phash_identical_images_zero_distance(tmp_path):
    from PIL import Image

    arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(arr).save(a)
    Image.fromarray(arr).save(b)
    assert leakage._phash(str(a)) - leakage._phash(str(b)) == 0
    assert leakage._content_hash(str(a)) == leakage._content_hash(str(b))
