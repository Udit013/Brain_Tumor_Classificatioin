"""Agent workflow: human-in-the-loop, bounded revisions, recovery, retries, LLM path."""

import importlib.util
from types import SimpleNamespace

import pytest

from neuroclass.agent.llm_anthropic import ClaudeExplainer
from neuroclass.schemas import DraftExplanation, Statement
from neuroclass.vision.service import InferenceTimeout

from .conftest import CLIENT, FakeVision, signup, wait_for

# The paid Claude adapter's SDK is not installed on the Space (zero-cost build).
needs_anthropic = pytest.mark.skipif(importlib.util.find_spec("anthropic") is None,
                                     reason="anthropic SDK not installed (optional, paid provider)")

LOW = dict(predicted="glioma", confidence=0.45, tta_agree=3)       # trips caution rules


def test_low_confidence_pauses_for_review_and_resumes_on_approval(make_client):
    c, _ = make_client(vision=FakeVision(**LOW))
    user = signup(c, "student@example.org")
    reviewer = signup(c, "reviewer@example.org")
    aid = c.post("/analyses", data={"sample_id": "glioma"}, headers=user).json()["id"]
    a = wait_for(c, aid, headers=user)
    assert a["status"] == "awaiting_review"
    assert a["reliability"]["requires_review"] is True
    assert [q["id"] for q in c.get("/analyses/review-queue", headers=reviewer).json()] == [aid]
    # a regular user cannot approve, even their own analysis
    assert c.post(f"/analyses/{aid}/reviews", json={"decision": "approve"}, headers=user).status_code == 403
    r = c.post(f"/analyses/{aid}/reviews", json={"decision": "approve", "note": "checked"},
               headers=reviewer)
    assert r.status_code == 200
    a = wait_for(c, aid, headers=user)
    assert a["status"] == "completed"
    assert a["reviews"][0]["decision"] == "approve" and a["reviews"][0]["reviewer"] == "reviewer@example.org"
    steps = [t["step"] for t in a["trace"] if t["status"] == "completed"]
    assert steps[-2:] == ["human_review", "report"]
    report = c.get(f"/analyses/{aid}/report", headers=user).text
    assert "not clinical validation" in report


def test_anonymous_runs_never_wait_for_a_reviewer(make_client):
    c, _ = make_client(vision=FakeVision(**LOW))
    a = wait_for(c, c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).json()["id"])
    assert a["status"] == "completed"
    assert "Review was recommended" in c.get(f"/analyses/{a['id']}/report", headers=CLIENT).text


def test_revision_requests_are_bounded(make_client):
    c, _ = make_client(vision=FakeVision(**LOW))
    user = signup(c, "student@example.org")
    reviewer = signup(c, "reviewer@example.org")
    aid = c.post("/analyses", data={"sample_id": "glioma"}, headers=user).json()["id"]
    for _ in range(2):                                     # MAX_REVISIONS
        assert wait_for(c, aid, headers=user)["status"] == "awaiting_review"
        c.post(f"/analyses/{aid}/reviews", json={"decision": "request_revision",
                                                 "note": "say more about calibration"}, headers=reviewer)
    assert wait_for(c, aid, headers=user)["status"] == "awaiting_review"
    c.post(f"/analyses/{aid}/reviews", json={"decision": "request_revision"}, headers=reviewer)
    a = wait_for(c, aid, headers=user)
    assert a["status"] == "completed"                      # third request ends the loop
    assert sum(t["step"] == "revise" and t["status"] == "completed" for t in a["trace"]) == 2


def test_paused_run_survives_a_restart(make_client, tmp_path):
    from neuroclass.runtime import Services

    c, services = make_client(vision=FakeVision(**LOW))
    user = signup(c, "student@example.org")
    reviewer = signup(c, "reviewer@example.org")
    aid = c.post("/analyses", data={"sample_id": "glioma"}, headers=user).json()["id"]
    assert wait_for(c, aid, headers=user)["status"] == "awaiting_review"
    # "restart": a fresh Services over the same database and checkpoint file
    fresh = Services(services.settings, vision=FakeVision(**LOW))
    fresh._retriever_error = "disabled"
    c.app.state.services = fresh
    c.post(f"/analyses/{aid}/reviews", json={"decision": "reject", "note": "artefact"}, headers=reviewer)
    a = wait_for(c, aid, headers=user)
    assert a["status"] == "completed" and a["reviews"][0]["decision"] == "reject"
    assert fresh.vision.calls == 0            # resumed from the checkpoint, no re-inference


def test_inference_timeout_is_retried_once(make_client):
    v = FakeVision(fail_times=1, exc=InferenceTimeout("slow"))
    c, _ = make_client(vision=v)
    a = wait_for(c, c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).json()["id"])
    assert a["status"] == "completed" and v.calls == 2
    statuses = [t["status"] for t in a["trace"] if t["step"] == "infer"]
    assert statuses.count("failed") == 1 and statuses.count("completed") == 1


def test_repeated_timeouts_fail_cleanly(make_client):
    c, _ = make_client(vision=FakeVision(fail_times=5, exc=InferenceTimeout("slow")))
    a = wait_for(c, c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).json()["id"])
    assert a["status"] == "failed" and "InferenceTimeout" in a["error"]


# --------------------------------------------------------------------------- #
# LLM path with a scripted fake Anthropic client
# --------------------------------------------------------------------------- #
class FakeAnthropic:
    """Replays scripted responses for client.beta.messages.parse."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


def resp(stop_reason, content=(), parsed=None, usage=(1000, 200)):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), parsed_output=parsed,
                           usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]))


def tool_use(name, args, id_="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id_)


def explainer(client, retriever):
    return ClaudeExplainer("claude-opus-5-5", "medium", 4000, 30.0,
                           search=lambda q, st: retriever.search(q, k=2, mode="hybrid")[0],
                           model_facts=lambda: {"test_accuracy": 0.93}, client=client)


@needs_anthropic
def test_llm_tool_loop_and_verification(make_client, retriever):
    good = DraftExplanation(summary="The model-predicted class is meningioma.", statements=[
        Statement(kind="model_output", text="Calibrated confidence is 95.0%."),
        Statement(kind="literature", text="Transfer learning from ImageNet reached 95 percent accuracy for meningioma MRI.",
                  citations=["PMC3#4"]),
        Statement(kind="literature", text="Grad-CAM proved the tumour causes the prediction.",
                  citations=["PMC9#1"]),                                 # invented id
        Statement(kind="model_output", text="The model is 99.9% accurate on this patient."),  # invented number
        Statement(kind="literature", text="Saliency maps can mislead."),  # uncited
    ])
    fake = FakeAnthropic([resp("tool_use", [tool_use("search_literature",
                                                     {"query": "meningioma transfer learning",
                                                      "source_type": None})]),
                          resp("end_turn", parsed=good)])
    c, services = make_client(retriever=retriever,
                              explainer_factory=lambda: explainer(fake, retriever))
    a = wait_for(c, c.post("/analyses", data={"sample_id": "meningioma"}, headers=CLIENT).json()["id"])
    e = a["explanation"]
    assert e["mode"] == "llm" and e["llm_model"] == "claude-opus-5-5"
    kept = {s["text"] for s in e["statements"]}
    removed = {s["text"]: s["reason"] for s in e["removed_statements"]}
    assert "Calibrated confidence is 95.0%." in kept
    assert any("ImageNet" in t for t in kept)
    assert "not retrieved" in removed["Grad-CAM proved the tumour causes the prediction."]
    assert "99.9" in removed["The model is 99.9% accurate on this patient."]
    assert "without a citation" in removed["Saliency maps can mislead."]
    # the tool result was sent back as untrusted, quoted passages
    second = fake.requests[1]["messages"][-1]["content"][0]
    assert second["type"] == "tool_result" and "<passage id=" in second["content"]
    assert fake.requests[0]["betas"] and fake.requests[0]["fallbacks"] == "default"
    # usage was recorded against the daily cap
    from neuroclass.db.models import LlmUsage
    from neuroclass.db.session import session_scope
    with session_scope() as db:
        assert db.query(LlmUsage).one().calls == 2


@needs_anthropic
def test_refusal_falls_back_to_extractive(make_client, retriever):
    fake = FakeAnthropic([resp("refusal")])
    c, _ = make_client(retriever=retriever, explainer_factory=lambda: explainer(fake, retriever))
    a = wait_for(c, c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).json()["id"])
    assert a["status"] == "completed" and a["explanation"]["mode"] == "extractive"
    detail = [t["detail"] for t in a["trace"] if t["step"] == "explain" and t["status"] == "completed"][0]
    assert "declined" in detail["fallback_reason"]


@needs_anthropic
def test_provider_outage_falls_back_to_extractive(make_client, retriever):
    import anthropic
    import httpx2

    class Down:
        def __init__(self):
            self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

        def _parse(self, **_):
            raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))

    c, _ = make_client(retriever=retriever, explainer_factory=lambda: explainer(Down(), retriever))
    a = wait_for(c, c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).json()["id"])
    assert a["status"] == "completed" and a["explanation"]["mode"] == "extractive"


def test_daily_cap_switches_to_extractive(make_client, retriever):
    calls = []
    c, services = make_client(retriever=retriever, llm_daily_call_cap=0,
                              explainer_factory=lambda: calls.append(1))
    a = wait_for(c, c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).json()["id"])
    assert a["explanation"]["mode"] == "extractive" and calls == []


@needs_anthropic
def test_tool_loop_is_bounded(retriever):
    loop = [resp("tool_use", [tool_use("get_model_facts", {}, id_=f"t{i}")]) for i in range(10)]
    fake = FakeAnthropic(loop)
    e = explainer(fake, retriever)
    try:
        e.explain_analysis({}, {"findings": []}, [], None)
    except Exception as exc:                                 # LlmUnavailable
        assert "did not finish" in str(exc) or "no structured output" in str(exc)
    assert len(fake.requests) == 4                           # MAX_TOOL_ROUNDS + final
    assert fake.requests[-1]["tools"] == []                  # last round forbids tools
