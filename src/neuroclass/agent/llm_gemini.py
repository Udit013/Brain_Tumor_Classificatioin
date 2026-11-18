"""Gemini adapter — the default hosted LLM, used on the Gemini API free tier only.

Zero-cost rules this adapter enforces or relies on:
- The API key must come from a Google AI Studio project with NO billing account
  linked. Google moves a project to a paid tier only after billing is linked, so
  an unbilled key cannot incur charges; requests over the free quota fail with
  HTTP 429 instead. (Operator responsibility; documented in docs/FREE_TIER.md.)
- Every request first takes a slot from `QuotaGuard` (per-minute and per-day
  budgets set below the free-tier limits). No slot -> no request -> extractive.
- The SDK's automatic retries are disabled (one attempt per call) so a 429 is
  never retried into the limit; a 429 pauses the LLM (until the daily reset
  for per-day quotas, otherwise for the server's retry delay).
- At most `tool_rounds` + 1 calls per request (default 1 + 1 = 2).
- Images are never sent. Free-tier content may be used by Google to improve its
  products, so only the inference numbers, the reliability findings, public
  literature excerpts and the user's question/note are sent; the UI says so.

Flow: (1) up to `tool_rounds` calls with function declarations and no schema;
the model may call search_literature / get_model_facts. (2) one final call with
`response_json_schema` and no tools, parsed and validated as DraftExplanation.
Splitting the phases keeps structured output independent of tool support.
"""

from __future__ import annotations

import os
import re
import time
from typing import Callable

from pydantic import ValidationError

from ..schemas import DraftExplanation, Passage
from .llm import (CLOSED_BOOK_PROMPT, QA_PROMPT, SYSTEM_PROMPT, TOOL_SPECS, LlmResult,
                  LlmUnavailable, QuotaExhausted, QuotaGuard, ToolRunner, analysis_content,
                  question_content)

_KEEP = {"type", "properties", "required", "items", "enum", "description", "minItems", "maxItems",
         "anyOf", "nullable", "format"}
_BAD_FINISH = {"MAX_TOKENS": "the response hit max_output_tokens", "SAFETY": "blocked by safety filters",
               "PROHIBITED_CONTENT": "blocked as prohibited content", "BLOCKLIST": "blocked",
               "SPII": "blocked (sensitive personal information)", "RECITATION": "blocked (recitation)",
               "MALFORMED_FUNCTION_CALL": "malformed function call",
               "MALFORMED_RESPONSE": "malformed response"}


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None


def portable_schema(schema: dict) -> dict:
    """Inline $refs and keep only widely supported JSON Schema keywords."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            out = {}
            for k, v in node.items():
                if k not in _KEEP:
                    continue
                if k == "type" and isinstance(v, list):          # ["string", "null"]
                    types_ = [t for t in v if t != "null"]
                    out["type"] = types_[0] if len(types_) == 1 else types_
                    if "null" in v:
                        out["nullable"] = True
                elif k == "enum":
                    out["enum"] = [e for e in v if e is not None]
                elif k == "properties":
                    out["properties"] = {pk: walk(pv) for pk, pv in v.items()}
                else:
                    out[k] = walk(v)
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


DRAFT_SCHEMA = portable_schema(DraftExplanation.model_json_schema())


def _retry_delay_s(details) -> float | None:
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(details))
    return float(m.group(1)) if m else None


class GeminiExplainer:
    provider = "gemini"

    def __init__(self, model: str, max_tokens: int, timeout_s: float,
                 search: Callable[[str, str | None], list[Passage]],
                 model_facts: Callable[[], dict], guard: QuotaGuard,
                 tool_rounds: int = 1, client=None):
        self.model, self.max_tokens, self.tool_rounds = model, max_tokens, max(0, tool_rounds)
        self.search, self.model_facts, self.guard = search, model_facts, guard
        if client is None:
            from google import genai
            from google.genai import types

            key = api_key()
            if not key:
                raise LlmUnavailable("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=1)))
        self.client = client

    # ------------------------------------------------------------ one call
    def _call(self, contents, config, usage: dict):
        from google.genai import errors

        self.guard.acquire()                                   # raises QuotaExhausted
        try:
            resp = self.client.models.generate_content(model=self.model, contents=contents,
                                                       config=config)
        except errors.APIError as exc:
            if exc.code == 429:
                daily = "PerDay" in str(exc.details) or "per day" in str(exc.message or "").lower()
                self.guard.trip(_retry_delay_s(exc.details), "Gemini free-tier quota reached (429)",
                                until_daily_reset=daily)
                raise QuotaExhausted("Gemini free-tier quota reached") from exc
            raise LlmUnavailable(f"Gemini API error {exc.code}: {exc.status}") from exc
        except Exception as exc:  # noqa: BLE001 — timeouts / network: degrade, never crash
            raise LlmUnavailable(f"Gemini request failed: {type(exc).__name__}") from exc
        usage["calls"] += 1
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage["input_tokens"] += um.prompt_token_count or 0
            usage["output_tokens"] += (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
        fb = getattr(resp, "prompt_feedback", None)
        if fb is not None and getattr(fb, "block_reason", None):
            raise LlmUnavailable(f"prompt blocked: {fb.block_reason}")
        if not resp.candidates:
            raise LlmUnavailable("no candidates returned")
        reason = getattr(resp.candidates[0].finish_reason, "value", resp.candidates[0].finish_reason)
        if reason in _BAD_FINISH:
            raise LlmUnavailable(_BAD_FINISH[reason])
        return resp

    def _run(self, system: str, user_content: str, passages: list[Passage],
             use_tools: bool = True) -> LlmResult:
        from google.genai import types

        seen = {p.chunk_id: p for p in passages}
        runner = ToolRunner(self.search, self.model_facts, seen)
        usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0.0}
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_content)])]
        t0 = time.perf_counter()

        if use_tools and self.tool_rounds:
            tool_cfg = types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=self.max_tokens,
                tools=[types.Tool(function_declarations=[
                    types.FunctionDeclaration(name=t["name"], description=t["description"],
                                              parameters_json_schema=t["parameters"])
                    for t in TOOL_SPECS])],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
            for _ in range(self.tool_rounds):
                resp = self._call(contents, tool_cfg, usage)
                calls = resp.function_calls or []
                if not calls:
                    break
                contents.append(resp.candidates[0].content)      # keeps thought signatures
                parts = []
                for fc in calls:
                    out, is_error = runner.run(fc.name, dict(fc.args or {}))
                    parts.append(types.Part.from_function_response(
                        name=fc.name, response={"error" if is_error else "result": out}))
                parts.append(types.Part.from_text(text="Return the final structured answer now."))
                contents.append(types.Content(role="user", parts=parts))

        final_cfg = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=self.max_tokens,
            response_mime_type="application/json", response_json_schema=DRAFT_SCHEMA)
        resp = self._call(contents, final_cfg, usage)
        try:
            draft = DraftExplanation.model_validate_json(resp.text or "")
        except ValidationError as exc:
            raise LlmUnavailable("structured output did not match the schema") from exc
        usage["generation_ms"] = (time.perf_counter() - t0) * 1000
        return LlmResult(draft=draft, passages=list(seen.values()), usage=usage,
                         tool_calls=runner.log)

    # ------------------------------------------------------------ contract
    def explain_analysis(self, inference_json: dict, reliability_json: dict,
                         passages: list[Passage], user_note: str | None) -> LlmResult:
        return self._run(SYSTEM_PROMPT, analysis_content(inference_json, reliability_json,
                                                         passages, user_note), passages)

    def answer_question(self, question: str, passages: list[Passage],
                        use_tools: bool = True) -> LlmResult:
        return self._run(SYSTEM_PROMPT + "\n\n" + QA_PROMPT, question_content(question, passages),
                         passages, use_tools=use_tools)

    def answer_closed_book(self, question: str) -> LlmResult:
        """LLM-only baseline for the generation evaluation (no passages, no tools)."""
        return self._run(SYSTEM_PROMPT + "\n\n" + CLOSED_BOOK_PROMPT, question_content(question, []), [],
                         use_tools=False)
