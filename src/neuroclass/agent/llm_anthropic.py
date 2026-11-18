"""Claude adapter (opt-in, PAID — not used by the $0 deployment).

Selected only with NEUROCLASS_LLM_PROVIDER=anthropic *and*
NEUROCLASS_ALLOW_PAID_LLM=true *and* ANTHROPIC_API_KEY set; otherwise the
runtime refuses to build it. Kept because it exercises the same provider
contract as the free Gemini adapter (bounded tool loop, structured output,
verification) and lets a deployment that does have a budget swap providers
without touching the graph.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from ..schemas import DraftExplanation, Passage
from .llm import (QA_PROMPT, SYSTEM_PROMPT, TOOL_SPECS, LlmResult, LlmUnavailable, QuotaExhausted,
                  QuotaGuard, ToolRunner, analysis_content, question_content)

MAX_TOOL_ROUNDS = 3
FALLBACK_BETA = "server-side-fallback-2026-07-01"
TOOLS = [{"name": t["name"], "description": t["description"], "strict": True,
          "input_schema": t["parameters"]} for t in TOOL_SPECS]


def credentials_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class ClaudeExplainer:
    provider = "anthropic"

    def __init__(self, model: str, effort: str, max_tokens: int, timeout_s: float,
                 search: Callable[[str, str | None], list[Passage]],
                 model_facts: Callable[[], dict], client=None,
                 input_usd_per_mtok: float = 4.0, output_usd_per_mtok: float = 20.0,
                 guard: QuotaGuard | None = None):
        import anthropic

        self.client = client or anthropic.Anthropic(timeout=timeout_s, max_retries=2)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.search, self.model_facts = search, model_facts
        self.in_rate, self.out_rate = input_usd_per_mtok, output_usd_per_mtok
        self.guard = guard

    def _run(self, system: str, user_content: str, passages: list[Passage]) -> LlmResult:
        import anthropic

        seen = {p.chunk_id: p for p in passages}
        tools_runner = ToolRunner(self.search, self.model_facts, seen)
        messages: list = [{"role": "user", "content": user_content}]
        usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        t0 = time.perf_counter()
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            tools = TOOLS if round_no < MAX_TOOL_ROUNDS else []
            if self.guard is not None:
                self.guard.acquire()
            try:
                resp = self.client.beta.messages.parse(
                    model=self.model, max_tokens=self.max_tokens, system=system,
                    messages=messages, tools=tools, output_format=DraftExplanation,
                    output_config={"effort": self.effort},
                    betas=[FALLBACK_BETA], fallbacks="default",
                    cache_control={"type": "ephemeral"})
            except anthropic.RateLimitError as exc:
                if self.guard is not None:
                    self.guard.trip(60, "provider rate limit (429)")
                raise QuotaExhausted(f"rate limited: {exc}") from exc
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
                raise LlmUnavailable(f"{type(exc).__name__}: {exc}") from exc
            usage["calls"] += 1
            usage["input_tokens"] += getattr(resp.usage, "input_tokens", 0) or 0
            usage["output_tokens"] += getattr(resp.usage, "output_tokens", 0) or 0
            if resp.stop_reason == "refusal":
                raise LlmUnavailable("the model declined this request")
            if resp.stop_reason == "max_tokens":
                raise LlmUnavailable("the response hit max_tokens")
            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    args = block.input if isinstance(block.input, dict) else {}
                    content, is_error = tools_runner.run(block.name, args)
                    item = {"type": "tool_result", "tool_use_id": block.id, "content": content}
                    if is_error:
                        item["is_error"] = True
                    results.append(item)
                messages.append({"role": "user", "content": results})
                continue
            draft = getattr(resp, "parsed_output", None)
            if draft is None:
                raise LlmUnavailable("no structured output returned")
            usage["generation_ms"] = (time.perf_counter() - t0) * 1000
            usage["cost_usd"] = (usage["input_tokens"] * self.in_rate
                                 + usage["output_tokens"] * self.out_rate) / 1e6
            return LlmResult(draft=draft, passages=list(seen.values()), usage=usage,
                             tool_calls=tools_runner.log)
        raise LlmUnavailable("tool loop did not finish")

    def explain_analysis(self, inference_json: dict, reliability_json: dict,
                         passages: list[Passage], user_note: str | None) -> LlmResult:
        return self._run(SYSTEM_PROMPT, analysis_content(inference_json, reliability_json,
                                                         passages, user_note), passages)

    def answer_question(self, question: str, passages: list[Passage]) -> LlmResult:
        return self._run(SYSTEM_PROMPT + "\n\n" + QA_PROMPT, question_content(question, passages),
                         passages)
