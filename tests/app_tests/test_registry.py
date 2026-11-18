"""The promotion gate refuses regressions."""

import copy
import json

from neuroclass.registry import REGISTRY_PATH, gate, main

from .conftest import REPO

COMPARISON = json.loads((REPO / "results/metrics/model_comparison.json").read_text())


def test_deployed_model_passes_against_its_predecessor():
    assert gate(COMPARISON)["passed"] is True


def _swap():
    c = copy.deepcopy(COMPARISON)
    c["previous"], c["current"] = c["current"], c["previous"]
    s = c["significance"]["full_test"]
    s["previous_only_correct"], s["current_only_correct"] = s["current_only_correct"], s["previous_only_correct"]
    return c


def test_rolling_back_to_the_old_model_would_be_refused():
    result = gate(_swap())
    assert result["passed"] is False
    failed = {c["metric"] for c in result["checks"] if not c["passed"]}
    assert {"test_accuracy", "glioma_to_notumor", "paired_test_full"} <= failed


def test_one_more_missed_tumour_is_enough_to_refuse():
    c = copy.deepcopy(COMPARISON)
    c["previous"] = copy.deepcopy(c["current"])
    c["current"]["glioma_to_notumor"] += 1
    result = gate(c)
    assert result["passed"] is False
    assert [x["metric"] for x in result["checks"] if not x["passed"]] == ["glioma_to_notumor"]


def test_small_regression_inside_tolerance_passes():
    c = copy.deepcopy(COMPARISON)
    c["previous"] = copy.deepcopy(c["current"])
    c["current"]["test_accuracy"] -= 0.004
    assert gate(c)["passed"] is True


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text(json.dumps(COMPARISON))
    bad.write_text(json.dumps(_swap()))
    assert main(["gate", "--comparison", str(good)]) == 0
    assert main(["gate", "--comparison", str(bad)]) == 1


def test_registry_matches_the_serving_manifest():
    reg = json.loads(REGISTRY_PATH.read_text())
    manifest = json.loads((REPO / "models/serving/manifest.json").read_text())
    deployed = [v for v in reg["versions"] if v["status"] == "deployed"]
    assert len(deployed) == 1 and deployed[0]["version"] == manifest["model_version"]
    assert deployed[0]["weights_sha256"] == manifest["source_weights_sha256"]
