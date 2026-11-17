"""Extractive explanations: no LLM, every sentence traceable.

Used when no LLM is configured, the daily cap is reached, or the provider
fails. Model-output and limitation statements are templates over the actual
inference result and reliability findings; literature statements are the
single best-matching sentence quoted from each retrieved passage.
"""

from __future__ import annotations

import re

from ..rag.retrieval import tokenize
from ..schemas import DraftExplanation, InferenceResult, Passage, ReliabilityAssessment, Statement

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
DISPLAY = {"glioma": "glioma", "meningioma": "meningioma", "notumor": "no tumor",
           "pituitary": "pituitary tumor"}

GRADCAM_NOTE = ("The Grad-CAM heatmap shows which image regions most increased the model's score "
                "for its predicted class. It is not a tumor outline and does not show that those "
                "regions caused the prediction or that the prediction is correct.")
CONFIDENCE_NOTE = ("Calibrated confidence is the model's probability after temperature scaling on "
                   "held-out data. It describes how often predictions with this confidence were "
                   "right on the evaluation data, not the chance that this particular image has "
                   "this condition.")
SCOPE_NOTE = ("This is an experimental research model: its output is a model-predicted class, not "
              "a diagnosis, and literature excerpts give research context rather than verifying "
              "this image.")


def best_sentence(passage: Passage, query: str, max_words: int = 60) -> str:
    q = set(tokenize(query))
    sentences = [s.strip() for s in _SENT.split(passage.text) if 8 <= len(s.split()) <= max_words]
    if not sentences:
        words = passage.text.split()
        return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")
    return max(sentences, key=lambda s: (len(q & set(tokenize(s))), -abs(len(s.split()) - 25)))


def literature_statements(passages: list[Passage], query: str, limit: int = 4) -> list[Statement]:
    out = []
    for p in passages[:limit]:
        out.append(Statement(text=best_sentence(p, query), kind="literature", citations=[p.chunk_id]))
    return out


def compose_analysis(inf: InferenceResult, rel: ReliabilityAssessment,
                     passages: list[Passage], query: str) -> DraftExplanation:
    cls = DISPLAY[inf.predicted_class]
    second = sorted(inf.probabilities, key=lambda p: -p.calibrated)[1]
    statements = [
        Statement(kind="model_output", text=(
            f"The model-predicted class is {cls}, with calibrated confidence "
            f"{inf.calibrated_confidence * 100:.1f}% (uncalibrated {inf.raw_confidence * 100:.1f}%). "
            f"The next most likely class is {DISPLAY[second.label]} at {second.calibrated * 100:.1f}%.")),
        Statement(kind="model_output", text=(
            f"{inf.tta_agreement} of {inf.tta_views} slightly altered versions of the image "
            f"(flip, small rotations, brightness changes) give the same class.")),
        Statement(kind="model_output", text=(
            f"The input-shift heuristic distance is {inf.ood_distance:.1f} against a threshold of "
            f"{inf.ood_threshold:.1f}; " + ("the image is flagged as unusual relative to the training "
                                            "data." if inf.ood_flag else "the image is not flagged."))),
        Statement(kind="limitation", text=CONFIDENCE_NOTE),
        Statement(kind="limitation", text=GRADCAM_NOTE),
    ]
    for f in rel.findings:
        if f.severity.value in ("caution", "info"):
            statements.append(Statement(kind="limitation", text=f.message))
    statements += literature_statements(passages, query)
    level = {"standard": "No reliability rule was triggered",
             "caution": "Some reliability rules were triggered, so treat the result with caution",
             "unreliable": "The result should be treated as unreliable"}[rel.level]
    summary = (f"Model-predicted class: {cls} ({inf.calibrated_confidence * 100:.1f}% calibrated "
               f"confidence). {level}. {SCOPE_NOTE}")
    return DraftExplanation(summary=summary, statements=statements)


def compose_answer(passages: list[Passage], question: str) -> DraftExplanation:
    if not passages:
        return DraftExplanation(summary="No indexed source matched this question, so no answer "
                                        "is given. Try rephrasing, or ask about a topic the corpus "
                                        "covers.", statements=[])
    statements = literature_statements(passages, question, limit=5)
    summary = (f"Extracted from {len(statements)} retrieved passage(s); no language model was used, "
               "so these are quotations ranked by relevance, not a synthesised answer.")
    return DraftExplanation(summary=summary, statements=statements)
