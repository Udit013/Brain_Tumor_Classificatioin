"""Vision parity, retrieval mechanics and claim verification."""

import hashlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from btc.preprocess import load_image
from neuroclass.agent.verify import numbers, verify
from neuroclass.reports.render import render_html, render_markdown
from neuroclass.schemas import (AnalysisOut, AnalysisStatus, DraftExplanation, Explanation,
                                SearchFilters, Statement)
from neuroclass.vision.service import tta_views
from neuroclass.vision.validation import InputRejected, assess_quality, decode_image

from .conftest import EXAMPLES, REPO, SERVING, FakeVision

FIXTURE = REPO / "tests" / "fixtures" / "decoder_parity" / "Te-no_120.jpg"
# SHA-256 (first 16 hex) of btc.preprocess output under the pinned Pillow 13.0.0.
# Pillow 10.3/12.3 decode FIXTURE differently ("0741c5cc3328070a"); that change was
# measured to alter no prediction (results/metrics/decoder_parity.json). Any further
# decoder change must be re-measured the same way before these hashes are updated.
GOLDEN = {"glioma__Te-gl_1.jpg": "1c69d892c9216b97", "meningioma__Te-me_1.jpg": "f2cf780eab865646",
          "notumor__Te-no_1.jpg": "5db0a92b8434965b", "pituitary__Te-pi_1.jpg": "d8281f46f60b4198",
          "Te-no_120.jpg": "02c4a7467e55d379"}


# --------------------------------------------------------------------------- vision
@pytest.mark.parametrize("path", EXAMPLES + [FIXTURE], ids=lambda p: Path(p).name)
def test_preprocessing_matches_the_evaluated_decoder(path):
    digest = hashlib.sha256(load_image(path).tobytes()).hexdigest()[:16]
    assert digest == GOLDEN[Path(path).name], (
        "model input changed: an image library upgrade altered decoding/resizing. "
        "Re-measure prediction changes (results/metrics/decoder_parity.json) before upgrading.")


def test_tta_views_match_the_space():
    spec = importlib.util.spec_from_file_location("space_logic", REPO / "space" / "logic.py")
    logic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logic)
    x = np.random.default_rng(1).uniform(0, 255, (256, 256, 3)).astype("float32")
    np.testing.assert_array_equal(tta_views(x), logic.tta_views(x))


def test_quality_findings():
    gray = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (256, 1)))
    assert assess_quality(gray).grayscale_like and assess_quality(gray).findings == []
    flat = assess_quality(Image.new("L", (300, 300), 90))
    assert any(f.startswith("low_contrast") for f in flat.findings)
    wide = assess_quality(Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (100, 3))))
    assert any(f.startswith("unusual_aspect_ratio") for f in wide.findings)


def test_decompression_bomb_is_rejected_before_decoding():
    buf = io.BytesIO()
    Image.new("L", (6000, 6000)).save(buf, "PNG", optimize=True)
    with pytest.raises(InputRejected) as exc:
        decode_image(buf.getvalue(), 10 << 20, 25_000_000)
    assert exc.value.code == "too_many_pixels"


needs_bundle = pytest.mark.skipif(not (SERVING / "manifest.json").exists()
                                  or not (SERVING / "neuroclass_serving.onnx").exists(),
                                  reason="serving bundle not present (python -m btc.export_serving)")


@needs_bundle
def test_real_model_reproduces_known_outputs():
    from neuroclass.vision.service import VisionService

    svc = VisionService(SERVING)
    expected = {"glioma": ("glioma", 0.350), "meningioma": ("meningioma", 0.965),
                "notumor": ("notumor", 0.997), "pituitary": ("pituitary", 0.998)}
    for path in EXAMPLES:
        img = decode_image(path.read_bytes(), 10 << 20, 25_000_000)
        r = svc.predict(img, assess_quality(img))
        cls, conf = expected[path.name.split("__")[0]]
        assert r.predicted_class == cls and abs(r.calibrated_confidence - conf) < 1e-3
        assert abs(sum(p.calibrated for p in r.probabilities) - 1) < 1e-6
        assert r.gradcam_png_base64 and not r.ood_flag


@needs_bundle
def test_bundle_with_wrong_checksum_refuses_to_serve(tmp_path):
    import shutil

    from neuroclass.vision.service import ModelUnavailable, VisionService

    for f in SERVING.iterdir():
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "head.npz").write_bytes(b"tampered")
    with pytest.raises(ModelUnavailable, match="checksum"):
        VisionService(tmp_path).ensure_loaded()


# ------------------------------------------------------------------------ retrieval
def test_bm25_and_dense_find_the_obvious_passage(retriever):
    for mode in ("bm25", "dense", "hybrid", "routed"):
        passages, info = retriever.search("temperature scaling calibration error", k=2, mode=mode)
        assert passages[0].doc_id == "PMC2", mode
        assert info["index_version"] == "test-index"


def test_project_questions_are_routed_to_project_documents(retriever):
    passages, info = retriever.search("What accuracy does the NeuroClass deployed model reach?",
                                      k=3, mode="routed")
    assert info["filters"]["source_type"] == "project_document"
    assert {p.source_type for p in passages} == {"project_document"}
    passages, info = retriever.search("calibration of deep networks", k=3, mode="routed")
    assert info["filters"]["source_type"] is None


def test_explicit_filters_and_diversity(retriever):
    passages, _ = retriever.search("MRI", k=6, mode="routed", filters=SearchFilters(min_year=2023))
    assert {p.year for p in passages} <= {2023, 2026}
    passages, _ = retriever.search("grad-cam saliency tumour MRI", k=6, mode="routed")
    per_doc = {}
    for p in passages:
        per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + 1
    assert max(per_doc.values()) <= 2


def test_empty_retrieval_gives_no_answer_rather_than_a_guess():
    from neuroclass.agent.compose import compose_answer

    draft = compose_answer([], "anything")
    assert draft.statements == [] and "No indexed source" in draft.summary


# --------------------------------------------------------------------- verification
def _inference():
    from neuroclass.schemas import InputQuality

    q = InputQuality(width=300, height=300, mode="L", format="PNG", grayscale_like=True,
                     mean_intensity=100.0, std_intensity=40.0)
    return FakeVision(confidence=0.912).predict(None, q)


def test_numbers_normalisation():
    assert numbers("accuracy 94.0% on 1,600 images and 0.350") == {"94", "1600", "0.35"}


def test_verifier_rules(retriever):
    passages, _ = retriever.search("temperature scaling", k=4, mode="hybrid")
    pid = next(p.chunk_id for p in passages if p.doc_id == "PMC2" and "0.12" in p.text)
    draft = DraftExplanation(summary="Calibration matters.", statements=[
        Statement(kind="literature", text="After temperature scaling the expected calibration error fell from 0.12 to 0.03.",
                  citations=[pid]),
        Statement(kind="literature", text="After temperature scaling the expected calibration error fell from 0.12 to 0.01.",
                  citations=[pid]),                                    # wrong number
        Statement(kind="literature", text="Bananas ripen faster in paper bags at room temperature.",
                  citations=[pid]),                                    # unsupported
        Statement(kind="literature", text="Calibration was perfect.", citations=["PMC404#0"]),
        Statement(kind="model_output", text="Calibrated confidence is 91.2%."),
        Statement(kind="model_output", text="Calibrated confidence is 97.0%."),
    ])
    summary, kept, removed = verify(draft, passages, _inference(), None)
    assert [s.text for s in kept] == [draft.statements[0].text, "Calibrated confidence is 91.2%."]
    reasons = " | ".join(s.reason for s in removed)
    assert "['0.01']" in reasons and "low lexical support" in reasons
    assert "not retrieved" in reasons and "['97']" in reasons


def test_summary_with_untraceable_numbers_is_withheld(retriever):
    passages, _ = retriever.search("temperature", k=2, mode="hybrid")
    draft = DraftExplanation(summary="The model is 99.7% accurate.", statements=[])
    summary, _, _ = verify(draft, passages, _inference(), None)
    assert summary.startswith("Summary withheld")


def test_report_language_and_provenance():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    a = AnalysisOut(id="abc", status=AnalysisStatus.completed, created_at=now, updated_at=now,
                    sample_id="glioma", inference=_inference(),
                    explanation=Explanation(mode="extractive", summary="s", statements=[], passages=[]))
    md = render_markdown(a, FakeVision().model_info())
    assert "Model-predicted class" in md and "not a diagnosis" in md
    assert "Not reviewed." in md and "weights sha256" in md
    low = md.lower().replace("not a diagnosis", "")
    assert "diagnosis" not in low and "diagnosed" not in low
    html = render_html(md, "t")
    assert html.startswith("<!doctype html>") and "<script" not in html
