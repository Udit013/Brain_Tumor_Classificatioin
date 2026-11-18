"""Deterministic verification of explanation statements.

An LLM's own assertion is never evidence. Rules, applied to every statement:

  literature    must cite >= 1 passage; every cited id must be a passage that
                was actually retrieved for this request; the statement must
                share enough content words with a cited passage
                (SUPPORT_THRESHOLD) and every number in it must appear in a
                cited passage.
  model_output  every number must be a value of this inference result
  limitation    every number must appear in the reliability findings or the
                inference result

Failing statements are removed from the explanation and listed separately,
with the reason. This checks lexical support, not entailment: a statement can
pass while subtly misreading its source. The tests in tests/platform pin the
known failure modes (invented ids, wrong numbers, uncited claims).
"""

from __future__ import annotations

import re

from ..rag.retrieval import tokenize
from ..schemas import (DraftExplanation, InferenceResult, Passage, ReliabilityAssessment,
                       Statement, VerifiedStatement)

SUPPORT_THRESHOLD = 0.5
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")


def numbers(text: str) -> set[str]:
    """Numbers as normalised strings ('94.0' -> '94', '0.350' -> '0.35')."""
    out = set()
    for m in _NUM.findall(text.replace(",", "")):
        v = float(m)
        out.add(f"{v:g}")
    return out


def allowed_model_numbers(inf: InferenceResult | None, rel: ReliabilityAssessment | None) -> set[str]:
    vals: set[str] = set()
    if inf is not None:
        def add_prob(p: float):
            for scaled in (p, p * 100):
                for nd in (0, 1, 2, 3):
                    vals.add(f"{round(scaled, nd):g}")
        add_prob(inf.calibrated_confidence)
        add_prob(inf.raw_confidence)
        for cp in inf.probabilities:
            add_prob(cp.calibrated)
            add_prob(cp.raw)
        for v in (inf.tta_agreement, inf.tta_views, inf.input_quality.width, inf.input_quality.height):
            vals.add(f"{v:g}")
        for v in (inf.ood_distance, inf.ood_threshold, inf.tta_entropy):
            for nd in (0, 1, 2):
                vals.add(f"{round(v, nd):g}")
        vals |= {"256", "2", "3", "4", "6"}                     # model input size, class/view counts
    if rel is not None:
        for f in rel.findings:
            vals |= numbers(f.message)
    return vals


def support_score(statement: str, passage_text: str) -> float:
    s = set(tokenize(statement))
    if not s:
        return 0.0
    p = set(tokenize(passage_text))
    return len(s & p) / len(s)


def verify_statement(st: Statement, passages: dict[str, Passage],
                     allowed: set[str]) -> VerifiedStatement:
    nums = numbers(st.text)
    if st.kind == "literature":
        if not st.citations:
            return VerifiedStatement(**st.model_dump(), status="unsupported",
                                     reason="literature claim without a citation")
        unknown = [c for c in st.citations if c not in passages]
        if unknown:
            return VerifiedStatement(**st.model_dump(), status="unsupported",
                                     reason=f"cites passages that were not retrieved: {unknown}")
        cited = [passages[c] for c in st.citations]
        score = max(support_score(st.text, p.text) for p in cited)
        cited_nums = set().union(*(numbers(p.text) for p in cited))
        missing = sorted(nums - cited_nums)
        if missing:
            return VerifiedStatement(**st.model_dump(), status="unsupported", support_score=score,
                                     reason=f"numbers not found in the cited passages: {missing}")
        if score < SUPPORT_THRESHOLD:
            return VerifiedStatement(**st.model_dump(), status="unsupported", support_score=score,
                                     reason=f"low lexical support ({score:.2f} < {SUPPORT_THRESHOLD})")
        return VerifiedStatement(**st.model_dump(), status="supported", support_score=score)
    # model_output / limitation: grounded in this request's own data
    missing = sorted(nums - allowed)
    if missing:
        return VerifiedStatement(**st.model_dump(), status="unsupported",
                                 reason=f"numbers not present in the inference result or "
                                        f"reliability findings: {missing}")
    if st.citations and any(c not in passages for c in st.citations):
        return VerifiedStatement(**st.model_dump(), status="unsupported",
                                 reason="cites passages that were not retrieved")
    return VerifiedStatement(**st.model_dump(), status="not_applicable")


def verify(draft: DraftExplanation, passages: list[Passage], inference: InferenceResult | None,
           reliability: ReliabilityAssessment | None) -> tuple[str, list[VerifiedStatement],
                                                               list[VerifiedStatement]]:
    by_id = {p.chunk_id: p for p in passages}
    allowed = allowed_model_numbers(inference, reliability)
    kept, removed = [], []
    # Research Q&A has no inference result, so "model_output"/"limitation" labels
    # can't be checked against one: every statement must meet the literature rule
    # (cited, supported, numbers present in the cited passages). Measured on the live
    # Gemini evaluation: without this, correct cited facts labelled "limitation"
    # were removed for "numbers not present in the inference result".
    qa = inference is None and reliability is None
    for st in draft.statements:
        if qa and st.kind != "literature":
            v = verify_statement(st.model_copy(update={"kind": "literature"}), by_id, allowed)
            v = v.model_copy(update={"kind": st.kind})
        else:
            v = verify_statement(st, by_id, allowed)
        (removed if v.status == "unsupported" else kept).append(v)
    summary = draft.summary
    # The summary has no citations; it may only repeat numbers that are verifiable.
    if numbers(summary) - allowed - set().union(*(numbers(p.text) for p in passages), set()):
        summary = ("Summary withheld: it contained numbers that could not be traced to the "
                   "inference result or the retrieved sources. See the verified statements below.")
    return summary, kept, removed
