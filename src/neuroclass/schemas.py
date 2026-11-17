"""Typed contracts shared by the API, the agent graph and the reports.

Naming is deliberate: a prediction is a "model-predicted class", never a
diagnosis; literature is "research context", never verification of an image.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
ClassName = Literal["glioma", "meningioma", "notumor", "pituitary"]


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #
class InputQuality(BaseModel):
    width: int
    height: int
    mode: str
    format: str | None
    grayscale_like: bool = Field(description="channels nearly identical, as in MRI exports")
    mean_intensity: float
    std_intensity: float
    findings: list[str] = []


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    source_weights_sha256: str
    onnx_sha256: str
    hub_repo: str
    temperature: float
    class_names: list[str]
    preprocessing: str
    ood_threshold: float
    ood_method: str


class ClassProbability(BaseModel):
    label: ClassName
    calibrated: float
    raw: float


class InferenceResult(BaseModel):
    model_version: str
    predicted_class: ClassName
    calibrated_confidence: float
    raw_confidence: float
    probabilities: list[ClassProbability]
    tta_agreement: int = Field(description="how many of the TTA views agree with the prediction")
    tta_views: int
    tta_entropy: float
    ood_distance: float = Field(description="Mahalanobis distance of pooled features to training data")
    ood_threshold: float
    ood_flag: bool
    input_quality: InputQuality
    gradcam_png_base64: str | None = None
    latency_ms: dict[str, float]


# --------------------------------------------------------------------------- #
# Reliability (deterministic)
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    info = "info"
    caution = "caution"
    block = "block"


class ReliabilityFinding(BaseModel):
    code: str
    severity: Severity
    message: str


class ReliabilityAssessment(BaseModel):
    level: Literal["standard", "caution", "unreliable"]
    findings: list[ReliabilityFinding]
    requires_review: bool
    rules_version: str


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class Passage(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source_url: str
    license: str
    year: int | None = None
    journal: str | None = None
    source_type: Literal["pmc_article", "project_document"]
    score: float = 0.0


class SearchFilters(BaseModel):
    source_type: Literal["pmc_article", "project_document"] | None = None
    min_year: int | None = None
    topics: list[str] | None = None


# --------------------------------------------------------------------------- #
# Explanations and verification
# --------------------------------------------------------------------------- #
class Statement(BaseModel):
    """One claim, with the passages it relies on."""
    model_config = ConfigDict(extra="forbid")
    text: str
    kind: Literal["model_output", "limitation", "literature"]
    citations: list[str] = Field(default_factory=list,
                                 description="chunk_ids of supporting passages; required for literature")


class DraftExplanation(BaseModel):
    """Structured output requested from the LLM."""
    model_config = ConfigDict(extra="forbid")
    summary: str
    statements: list[Statement]


class VerifiedStatement(Statement):
    status: Literal["supported", "unsupported", "not_applicable"]
    support_score: float | None = None
    reason: str | None = None


class Explanation(BaseModel):
    mode: Literal["llm", "extractive"]
    llm_model: str | None = None
    summary: str
    statements: list[VerifiedStatement]
    removed_statements: list[VerifiedStatement] = []
    passages: list[Passage]
    usage: dict[str, float] = {}


# --------------------------------------------------------------------------- #
# Analyses, review, reports
# --------------------------------------------------------------------------- #
class AnalysisStatus(str, Enum):
    queued = "queued"
    running = "running"
    awaiting_review = "awaiting_review"
    completed = "completed"
    failed = "failed"


class ReviewDecision(str, Enum):
    approve = "approve"
    reject = "reject"
    request_revision = "request_revision"


class ReviewIn(BaseModel):
    decision: ReviewDecision
    note: str = Field(default="", max_length=4000)


class ReviewOut(ReviewIn):
    reviewer: str
    created_at: datetime


class TraceEvent(BaseModel):
    step: str
    status: Literal["started", "completed", "failed", "skipped", "interrupted"]
    at: datetime
    duration_ms: float | None = None
    detail: dict = {}


class AnalysisOut(BaseModel):
    id: str
    status: AnalysisStatus
    created_at: datetime
    updated_at: datetime
    sample_id: str | None = None
    question: str | None = None
    model_version: str | None = None
    inference: InferenceResult | None = None
    reliability: ReliabilityAssessment | None = None
    explanation: Explanation | None = None
    reviews: list[ReviewOut] = []
    error: str | None = None
    trace: list[TraceEvent] = []
    owner: str | None = None


class AnalysisSummary(BaseModel):
    id: str
    status: AnalysisStatus
    created_at: datetime
    sample_id: str | None
    predicted_class: str | None
    calibrated_confidence: float | None
    reliability_level: str | None


class ResearchQuestion(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: SearchFilters | None = None
    top_k: int = Field(default=6, ge=1, le=12)


class ResearchAnswer(BaseModel):
    question: str
    answer: Explanation
    retrieval_ms: float
    generation_ms: float
