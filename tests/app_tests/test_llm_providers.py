"""Provider abstraction, Gemini free-tier adapter and the zero-cost quota guard.

Gemini responses are built from the SDK's own response types and replayed by a
fake client: no network, no key, no paid (or free) API call.
"""

from datetime import datetime

import pytest
from google.genai import errors, types

from neuroclass.agent.llm import PACIFIC, QuotaExhausted, QuotaGuard
from neuroclass.agent.llm_gemini import DRAFT_SCHEMA, GeminiExplainer
from neuroclass.settings import Settings

from .conftest import CLIENT, wait_for


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeGemini:
    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def gresp(parts, finish="STOP", usage=(900, 150)):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts),
                                    finish_reason=types.FinishReason(finish))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=usage[0], candidates_token_count=usage[1]))


def call(name, args):
    return types.Part.from_function_call(name=name, args=args)


def text(t):
    return types.Part.from_text(text=t)


def quota_error(per_day=False, delay="32s"):
    quota_id = ("GenerateRequestsPerDayPerProjectPerModel-FreeTier" if per_day
                else "GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    return errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota",
        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                     "violations": [{"quotaId": quota_id}]},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}})


GOOD_JSON = """{"summary": "The model-predicted class is meningioma.", "statements": [
 {"kind": "model_output", "text": "Calibrated confidence is 95.0%.", "citations": []},
 {"kind": "literature", "text": "Transfer learning from ImageNet reached 95 percent accuracy for meningioma MRI.",
  "citations": ["PMC3#4"]},
 {"kind": "literature", "text": "Grad-CAM proved the tumour causes the prediction.", "citations": ["PMC9#1"]}
]}"""


def gemini(fake, retriever, guard=None, rounds=1):
    return GeminiExplainer("gemini-3.5-flash-lite", 4000, 30.0,
                           search=lambda q, st: retriever.search(q, k=2, mode="hybrid")[0],
                           model_facts=lambda: {"test_accuracy": 0.93},
                           guard=guard or QuotaGuard(10, 100), tool_rounds=rounds, client=fake)


# --------------------------------------------------------------------------- #
# Provider selection: free by default, paid only by explicit opt-in
# --------------------------------------------------------------------------- #
def test_provider_resolution_never_selects_a_paid_provider_implicitly(monkeypatch):
    assert Settings().resolved_llm_provider == "none"                     # no keys
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert Settings().resolved_llm_provider == "none"                     # auto never picks paid
    monkeypatch.setenv("NEUROCLASS_LLM_PROVIDER", "anthropic")
    assert Settings().resolved_llm_provider == "none"                     # paid needs ALLOW_PAID
    monkeypatch.setenv("NEUROCLASS_ALLOW_PAID_LLM", "true")
    assert Settings().resolved_llm_provider == "anthropic"
    monkeypatch.delenv("NEUROCLASS_LLM_PROVIDER")
    monkeypatch.setenv("GEMINI_API_KEY", "free-key")
    s = Settings()
    assert s.resolved_llm_provider == "gemini" and s.llm_model == "gemini-3.5-flash-lite"
    monkeypatch.setenv("NEUROCLASS_LLM_MODEL", "gemini-3.8-flash")
    assert Settings().llm_model == "gemini-3.8-flash"


def test_portable_schema_inlines_refs_and_drops_unsupported_keywords():
    flat = str(DRAFT_SCHEMA)
    assert "$ref" not in flat and "$defs" not in flat and "additionalProperties" not in flat
    assert DRAFT_SCHEMA["properties"]["statements"]["items"]["properties"]["kind"]["enum"] == [
        "model_output", "limitation", "literature"]


# --------------------------------------------------------------------------- #
# QuotaGuard
# --------------------------------------------------------------------------- #
class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def test_quota_guard_minute_window_day_reset_and_pause():
    noon = datetime(2026, 11, 17, 12, 0, tzinfo=PACIFIC).timestamp()
    clock = Clock(noon)
    g = QuotaGuard(per_minute=2, per_day=3, clock=clock)
    g.acquire(), g.acquire()
    with pytest.raises(QuotaExhausted, match="per-minute"):
        g.acquire()
    clock.t += 61
    g.acquire()
    clock.t += 61
    with pytest.raises(QuotaExhausted, match="daily"):
        g.acquire()
    clock.t = datetime(2026, 11, 18, 0, 0, 1, tzinfo=PACIFIC).timestamp()     # Pacific midnight
    g.acquire()
    g.trip(30, "429")
    assert not g.available() and g.status()["pause_reason"] == "429"
    clock.t += 31
    assert g.available()
    g.trip(None, "daily quota", until_daily_reset=True)
    clock.t = datetime(2026, 11, 18, 23, 59, tzinfo=PACIFIC).timestamp()
    assert not g.available()
    clock.t = datetime(2026, 11, 19, 0, 1, tzinfo=PACIFIC).timestamp()
    assert g.available()


# --------------------------------------------------------------------------- #
# Gemini adapter
# --------------------------------------------------------------------------- #
def test_gemini_tool_round_then_structured_answer_is_verified(make_client, retriever):
    fake = FakeGemini([
        gresp([call("search_literature", {"query": "meningioma transfer learning", "source_type": None})]),
        gresp([text(GOOD_JSON)]),
    ])
    c, services = make_client(retriever=retriever, explainer_factory=lambda: gemini(fake, retriever))
    a = wait_for(c, c.post("/analyses", data={"sample_id": "meningioma"}, headers=CLIENT).json()["id"])
    e = a["explanation"]
    assert e["mode"] == "llm" and e["llm_model"] == "gemini-3.5-flash-lite"
    kept = {s["text"] for s in e["statements"]}
    removed = {s["text"]: s["reason"] for s in e["removed_statements"]}
    assert "Calibrated confidence is 95.0%." in kept and any("ImageNet" in t for t in kept)
    assert "not retrieved" in removed["Grad-CAM proved the tumour causes the prediction."]
    assert e["usage"]["cost_usd"] == 0.0 and e["usage"]["calls"] == 2
    first, final = fake.requests
    assert first["config"].tools and first["config"].response_json_schema is None
    assert first["config"].automatic_function_calling.disable is True
    assert final["config"].tools is None and final["config"].response_mime_type == "application/json"
    # tool output went back as a function response containing quoted, untrusted passages
    fn_parts = [p for p in final["contents"][-1].parts if p.function_response]
    assert fn_parts and "<passage id=" in fn_parts[0].function_response.response["result"]
    # the model turn (with any thought signatures) was echoed back unchanged
    assert final["contents"][1].role == "model" and final["contents"][1].parts[0].function_call


def test_gemini_calls_are_bounded_even_if_the_model_keeps_calling_tools(retriever):
    loop = [gresp([call("get_model_facts", {})]) for _ in range(2)] + [gresp([text(GOOD_JSON)])] \
        + [gresp([call("get_model_facts", {})]) for _ in range(5)]
    fake = FakeGemini(loop)
    gemini(fake, retriever, rounds=2).explain_analysis({}, {"findings": []}, [], None)
    assert len(fake.requests) == 3                          # tool_rounds + final, never more
    assert fake.requests[-1]["config"].tools is None        # the final call cannot call tools


def test_gemini_quota_429_pauses_llm_and_falls_back_without_more_requests(make_client, retriever):
    guard = QuotaGuard(10, 100)
    fake = FakeGemini([quota_error(per_day=True)])
    c, services = make_client(retriever=retriever,
                              explainer_factory=lambda: gemini(fake, retriever, guard=guard))
    services.quota_guard = guard
    a = wait_for(c, c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).json()["id"])
    assert a["status"] == "completed" and a["explanation"]["mode"] == "extractive"
    assert "quota" in a["explanation"]["fallback_reason"].lower()
    assert guard.status()["paused_until"] is not None                    # until the daily reset
    # next request: budget check fails first, so nothing is sent to the provider
    b = wait_for(c, c.post("/analyses", data={"sample_id": "pituitary"}, headers=CLIENT).json()["id"])
    assert b["explanation"]["mode"] == "extractive"
    assert "budget" in b["explanation"]["fallback_reason"]
    assert len(fake.requests) == 1


def test_gemini_rate_limit_uses_server_retry_delay(retriever):
    guard = QuotaGuard(10, 100)
    e = gemini(FakeGemini([quota_error(per_day=False, delay="45s")]), retriever, guard=guard)
    with pytest.raises(QuotaExhausted):
        e.answer_question("q", [])
    paused = datetime.fromisoformat(guard.status()["paused_until"]).timestamp()
    assert 40 < paused - datetime.now().timestamp() <= 46


def test_local_budget_blocks_before_any_request(retriever):
    fake = FakeGemini([])
    e = gemini(fake, retriever, guard=QuotaGuard(per_minute=0, per_day=100))
    with pytest.raises(QuotaExhausted):
        e.answer_question("q", [])
    assert fake.requests == []


@pytest.mark.parametrize("item,match", [
    (gresp([text("not json")]), "schema"),
    (gresp([text("{}")], finish="SAFETY"), "safety"),
    (gresp([text("{}")], finish="MAX_TOKENS"), "max_output_tokens"),
    (errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "x"}}), "503"),
])
def test_gemini_failures_raise_llm_unavailable(retriever, item, match):
    from neuroclass.agent.llm import LlmUnavailable

    with pytest.raises(LlmUnavailable, match=match):
        gemini(FakeGemini([item]), retriever, rounds=0).answer_question("q", [])


def test_research_answer_reports_why_it_used_extractive_mode(make_client, retriever):
    c, _ = make_client(retriever=retriever)
    r = c.post("/research/ask", json={"question": "What limits Grad-CAM?"}, headers=CLIENT).json()
    assert r["answer"]["mode"] == "extractive"
    assert r["answer"]["fallback_reason"] == "no LLM provider configured"
    h = c.get("/ready").json()
    assert h["checks"]["llm"]["provider"] == "none" and h["checks"]["llm"]["paid_allowed"] is False
