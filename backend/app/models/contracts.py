"""RUNWISE shared request/response contracts.

Owner: the shared contracts area (initial owner, see PRD section 19).
Shared: imported by the frontend owner (frontend TypeScript types) and the integration owner (API/E2E tests).

Contract rules (PRD sections 19-20):
  * Additive and backward compatible. New fields must be optional with a default.
  * The frontend must tolerate optional fields and empty lists, so nothing here
    may be a required field on a response model that can legitimately be empty.
  * Never fabricate values. Missing observations are ``None``, never ``0``.

Response shapes marked "PRD 20" are interface contracts from the PRD. The
example values in the PRD are illustrative only and are never hardcoded here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ExperimentStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ComparabilityStatus(str, Enum):
    """FR-4. Exactly one of the four defined outcomes."""

    COMPARABLE = "COMPARABLE"
    COMPARABLE_WITH_CAVEATS = "COMPARABLE_WITH_CAVEATS"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class Severity(str, Enum):
    """How much a single comparability finding matters."""

    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class ComparabilityReasonCode(str, Enum):
    """Stable machine-readable reason codes (FR-4, FR-8).

    Codes are part of the public contract: the frontend may switch on them and
    the evaluation harness scores against them. Never repurpose a code; add a
    new one instead.
    """

    IDENTICAL = "IDENTICAL"
    DATASET_MISSING = "DATASET_MISSING"
    DATASET_MISMATCH = "DATASET_MISMATCH"
    DATASET_VERSION_MISMATCH = "DATASET_VERSION_MISMATCH"
    EVAL_SPLIT_MISMATCH = "EVAL_SPLIT_MISMATCH"
    EVAL_PROTOCOL_MISMATCH = "EVAL_PROTOCOL_MISMATCH"
    METRIC_MISMATCH = "METRIC_MISMATCH"
    METRIC_DIRECTION_MISMATCH = "METRIC_DIRECTION_MISMATCH"
    MODEL_FAMILY_MISMATCH = "MODEL_FAMILY_MISMATCH"
    MODEL_CONFIG_MISMATCH = "MODEL_CONFIG_MISMATCH"
    PREPROCESSING_MISMATCH = "PREPROCESSING_MISMATCH"
    TRAINING_BUDGET_MISMATCH = "TRAINING_BUDGET_MISMATCH"
    RANDOM_SEED_MISMATCH = "RANDOM_SEED_MISMATCH"
    METRIC_MISSING = "METRIC_MISSING"
    DATASET_VERSION_MISSING = "DATASET_VERSION_MISSING"
    EVAL_SPLIT_MISSING = "EVAL_SPLIT_MISSING"
    EVAL_PROTOCOL_MISSING = "EVAL_PROTOCOL_MISSING"
    SEED_MISSING = "SEED_MISSING"
    CONFIG_MISSING = "CONFIG_MISSING"
    DISPUTED_EVIDENCE = "DISPUTED_EVIDENCE"


class EvidenceStatus(str, Enum):
    """FR-7 provenance grade. LLM output may never be CONFIRMED (PRD 12)."""

    OBSERVED = "observed"
    """Measured and present in a stored experiment record."""

    USER_ASSERTED = "user_asserted"
    """Stated by a researcher and recorded, but not independently verified."""

    INFERRED = "inferred"
    """A derived explanation. Must be labelled as such in every answer."""

    CONFIRMED = "confirmed"
    """Explicitly signed off by a researcher through an approval flow."""

    DISPUTED = "disputed"
    """Challenged by an open correction."""

    SUPERSEDED = "superseded"
    """Replaced by a later claim or result; kept for history, not for trust."""

    UNRESOLVED = "unresolved"
    """In the revalidation queue with no decision yet."""


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class RecommendationStatus(str, Enum):
    """FR-9. Recommendations are advisory; a proposal is never executed."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    SUPERSEDED = "superseded"


class Decision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


class ReviewItemStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class CandidateKind(str, Enum):
    """The shape of the gap a recommendation closes (FR-6 gap analysis)."""

    SWEEP_PARAMETER = "sweep_parameter"
    TRY_DATASET_VERSION = "try_dataset_version"
    NEW_MODEL_FAMILY = "new_model_family"
    REPLICATE_SEED = "replicate_seed"
    EVALUATE_MISSING_METRIC = "evaluate_missing_metric"


class AnswerMode(str, Enum):
    """PRD 15: never pretend an external model was called."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Record(BaseModel):
    """Base for stored-record shapes.

    ``extra="allow"`` is deliberate: FR-1 requires model-specific configuration
    to be preserved rather than discarded, so unknown keys survive a round trip.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Core experiment records
# ---------------------------------------------------------------------------


class MetricValue(Record):
    """PRD 20 experiment summary metric shape, plus protocol detail for FR-4."""

    name: str
    value: float | None = None
    direction: MetricDirection = MetricDirection.MAXIMIZE
    evaluation_split: str | None = None
    evaluation_protocol: str | None = None

    @field_validator("direction", mode="before")
    @classmethod
    def _normalise_direction(cls, value: Any) -> Any:
        """Accept the direction spellings that appear in real CSV exports.

        Unknown spellings are rejected rather than silently coerced to
        ``maximize``, which would invert a researcher's optimisation goal.
        """
        if value is None or value == "":
            return MetricDirection.MAXIMIZE
        if isinstance(value, MetricDirection):
            return value
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "maximize": MetricDirection.MAXIMIZE,
            "maximise": MetricDirection.MAXIMIZE,
            "max": MetricDirection.MAXIMIZE,
            "higher_is_better": MetricDirection.MAXIMIZE,
            "higher": MetricDirection.MAXIMIZE,
            "up": MetricDirection.MAXIMIZE,
            "minimize": MetricDirection.MINIMIZE,
            "minimise": MetricDirection.MINIMIZE,
            "min": MetricDirection.MINIMIZE,
            "lower_is_better": MetricDirection.MINIMIZE,
            "lower": MetricDirection.MINIMIZE,
            "down": MetricDirection.MINIMIZE,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(f"unrecognised metric direction: {value!r}")


class ExperimentSummary(Record):
    """PRD 20 contract: the public experiment list shape."""

    id: str
    project_id: str
    name: str
    model_family: str
    dataset_version_id: str | None = None
    status: ExperimentStatus = ExperimentStatus.UNKNOWN
    metrics: list[MetricValue] = Field(default_factory=list)
    compute_cost_gpu_hours: float | None = None
    created_at: datetime | None = None
    duration_seconds: float | None = None


class ExperimentRecord(ExperimentSummary):
    """Internal analysis record (analysis engines, graph adapter).

    Extends :class:`ExperimentSummary` additively. Comparability (FR-4) and the
    recommendation engine need the dataset, protocol, seed and configuration
    context that the list view leaves out. Every field added here is optional so
    that a partially populated graph yields ``INSUFFICIENT_INFORMATION`` rather
    than a crash.
    """

    description: str | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    dataset_quality_status: EvidenceStatus | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    """Model configuration. Named ``configuration`` rather than ``model_config``
    because pydantic reserves the latter for model configuration itself."""
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    random_seed: int | None = None
    training_steps: int | None = None
    max_training_steps: int | None = None
    source_reference: str | None = None
    notes: str | None = None

    def metric(self, name: str) -> MetricValue | None:
        """Return the metric called ``name``, or ``None``.

        Lookup is case-insensitive because metric names arrive from free-form
        CSV headers. Returning ``None`` rather than raising keeps "missing
        metric" a normal, reportable state (FR-12: never invent data).
        """
        target = name.strip().lower()
        for candidate in self.metrics:
            if candidate.name.strip().lower() == target:
                return candidate
        return None

    def metric_names(self) -> list[str]:
        return [m.name for m in self.metrics]


# ---------------------------------------------------------------------------
# Datasets (FR-4 tracks dataset version as a comparability dimension)
# ---------------------------------------------------------------------------


class DatasetSummary(Record):
    id: str
    name: str
    description: str | None = None


class DatasetVersionSummary(Record):
    id: str
    dataset_id: str
    version: str
    quality_status: EvidenceStatus = EvidenceStatus.OBSERVED
    """Becomes DISPUTED when a correction challenges this version (FR-6)."""
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Comparability (FR-4)
# ---------------------------------------------------------------------------


class ComparabilityReason(Record):
    """One explainable finding behind a comparability verdict."""

    code: ComparabilityReasonCode
    dimension: str
    severity: Severity
    message: str
    left_value: Any | None = None
    right_value: Any | None = None


class ComparabilityAssessment(Record):
    """FR-4 result: a defined status plus the reasoning that produced it."""

    status: ComparabilityStatus
    reasons: list[ComparabilityReason] = Field(default_factory=list)
    matched_dimensions: list[str] = Field(default_factory=list)
    differing_dimensions: list[str] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    score: float = 0.0
    """Heuristic ordering aid in ``[0, 1]``. Not a probability (PRD 5)."""
    score_kind: str = "severity_penalty_heuristic"
    confidence: str = "medium"
    left_experiment_id: str | None = None
    right_experiment_id: str | None = None
    explanation: str = ""

    @property
    def is_usable_as_baseline(self) -> bool:
        """True when the pair may serve as a controlled-comparison baseline."""
        return self.status in (
            ComparabilityStatus.COMPARABLE,
            ComparabilityStatus.COMPARABLE_WITH_CAVEATS,
        )


# ---------------------------------------------------------------------------
# Recommendations (FR-5)
# ---------------------------------------------------------------------------


class ScoreComponent(Record):
    """One weighted contribution to a heuristic score (FR-5, FR-8)."""

    name: str
    weight: float
    value: float
    weighted_value: float
    rationale: str


class ValidationCriterion(Record):
    """How the researcher should decide the experiment is finished."""

    kind: str
    description: str
    is_suggested_default: bool = True
    """True means the product supplied this, the researcher did not."""

    threshold: float | None = None
    metric_name: str | None = None


class Candidate(Record):
    """A proposed follow-up experiment before ranking and budget filtering."""

    kind: CandidateKind
    hypothesis: str
    configuration: dict[str, Any] = Field(default_factory=dict)
    independent_variables: list[str] = Field(default_factory=list)
    controlled_variables: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    baseline_experiment_ids: list[str] = Field(default_factory=list)
    supporting_experiment_ids: list[str] = Field(default_factory=list)
    estimated_gpu_hours: float | None = None
    estimate_source: str | None = None
    uncertainties: list[str] = Field(default_factory=list)
    validation_criteria: list[ValidationCriterion] = Field(default_factory=list)
    model_family: str | None = None
    dataset_version_id: str | None = None
    independent_variable_values: dict[str, Any] = Field(default_factory=dict)

    def to_configuration(self) -> dict[str, Any]:
        """The proposed configuration, safe to export as JSON (FR-9)."""
        return dict(self.configuration)


class ExcludedEvidence(Record):
    """Why a record was withheld from a recommendation (FR-8, FR-12)."""

    experiment_id: str
    reason: str
    detail: str
    status: EvidenceStatus


class Recommendation(Record):
    """PRD 20 contract: a next-experiment recommendation."""

    id: str
    project_id: str
    objective: str
    hypothesis: str
    candidate_kind: CandidateKind = CandidateKind.SWEEP_PARAMETER
    baseline_experiment_ids: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    independent_variables: list[str] = Field(default_factory=list)
    controlled_variables: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    estimated_gpu_hours: float | None = None
    estimate_source: str | None = None
    supporting_experiment_ids: list[str] = Field(default_factory=list)
    independent_variable_values: dict[str, Any] = Field(default_factory=dict)
    """The proposed value per independent variable, so the UI can render and edit
    the change without parsing the hypothesis text (FR-9 exports this as JSON)."""
    uncertainties: list[str] = Field(default_factory=list)
    validation_criteria: list[ValidationCriterion] = Field(default_factory=list)
    rationale: str = ""
    score: float = 0.0
    """Explainable heuristic score. Not a calibrated probability (PRD 5)."""
    score_kind: str = "weighted_heuristic"
    score_components: list[ScoreComponent] = Field(default_factory=list)
    evidence_status: EvidenceStatus = EvidenceStatus.INFERRED
    """Always INFERRED on creation: a proposal is not an observation."""
    excluded_evidence_ids: list[str] = Field(default_factory=list)
    status: RecommendationStatus = RecommendationStatus.PROPOSED
    created_at: datetime = Field(default_factory=utcnow)


class RecommendationSet(Record):
    """The ranked, budget-compliant answer to one recommendation request."""

    recommendations: list[Recommendation] = Field(default_factory=list)
    compute_budget_gpu_hours: float | None = None
    estimated_gpu_hours: float | None = None
    budget_exhausted: bool = False
    """True when the budget bound the selection (PRD 5)."""
    unselected_candidates: list[Candidate] = Field(default_factory=list)
    considered_experiment_ids: list[str] = Field(default_factory=list)
    excluded_evidence: list[ExcludedEvidence] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Evidence, claims and corrections (FR-6, FR-7, FR-8)
# ---------------------------------------------------------------------------


class EvidenceRef(Record):
    """A citable pointer back to stored data."""

    experiment_id: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    source_reference: str | None = None
    evidence_status: EvidenceStatus = EvidenceStatus.OBSERVED
    node_type: str = "Experiment"


class Claim(Record):
    """A statement in an answer, with its provenance and epistemic grade."""

    id: str
    text: str
    status: ClaimStatus = ClaimStatus.PROPOSED
    evidence_status: EvidenceStatus
    support: list[EvidenceRef] = Field(default_factory=list)
    detail: str | None = None


class ClaimPath(Record):
    """A graph path explaining how a conclusion depends on a source."""

    nodes: list[str] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)


class CorrectionImpact(Record):
    """PRD 20 contract: the blast radius of one correction (FR-6)."""

    correction_id: str
    source_id: str
    summary: str
    directly_affected_experiment_ids: list[str] = Field(default_factory=list)
    indirectly_affected_experiment_ids: list[str] = Field(default_factory=list)
    potentially_affected_claim_ids: list[str] = Field(default_factory=list)
    unaffected_experiment_ids: list[str] = Field(default_factory=list)
    review_item_ids: list[str] = Field(default_factory=list)
    paths: list[ClaimPath] = Field(default_factory=list)


class CorrectionRecord(Record):
    """FR-6 correction event."""

    id: str
    project_id: str
    source_id: str
    source_type: str
    description: str
    evidence_reference: str | None = None
    status: str = "open"
    created_by: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class ReviewItem(Record):
    """FR-6 revalidation queue entry."""

    id: str
    project_id: str
    reason: str
    status: ReviewItemStatus = ReviewItemStatus.OPEN
    experiment_id: str | None = None
    claim_id: str | None = None
    correction_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Query / copilot (FR-3)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """PRD 9 example body. Unknown keys ignored for forward compatibility."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1)
    compute_budget_gpu_hours: float | None = Field(default=None, ge=0)
    max_recommendations: int = Field(default=3, ge=1, le=20)


class GroundingReport(Record):
    """Proof that an answer cites only records the backend actually returned.

    FR-3 and FR-12: the copilot must never invent experiment IDs, metrics or
    evidence. ``is_grounded`` is false whenever any citation is unknown, and the
    offending identifiers are listed so the caller can fail loudly.
    """

    cited_experiment_ids: list[str] = Field(default_factory=list)
    unknown_citations: list[str] = Field(default_factory=list)
    available_experiment_count: int = 0
    citation_coverage: float = 0.0
    is_grounded: bool = True


class ResearchAnswer(Record):
    """FR-3: an answer plus the records that support it."""

    question: str
    answer: str
    mode: AnswerMode = AnswerMode.DETERMINISTIC
    llm_used: bool = False
    """Explicit so the UI can never imply a model was consulted (PRD 15)."""
    grounding: GroundingReport | None = None
    claims: list[Claim] = Field(default_factory=list)
    supporting_experiments: list[ExperimentSummary] = Field(default_factory=list)
    comparisons: list[ComparabilityAssessment] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    corrections_considered: list[CorrectionImpact] = Field(default_factory=list)
    excluded_evidence: list[ExcludedEvidence] = Field(default_factory=list)
    comparable_experiment_ids: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class KnowledgeFact(Record):
    """FR-7 research memory entry."""

    id: str
    text: str
    status: ClaimStatus = ClaimStatus.PROPOSED
    evidence_status: EvidenceStatus = EvidenceStatus.INFERRED
    support: list[EvidenceRef] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Requests for the recommendation and copilot routers
# ---------------------------------------------------------------------------


class RecommendationRequest(BaseModel):
    """PRD 9 body for ``POST /projects/{project_id}/recommendations``."""

    model_config = ConfigDict(extra="ignore")

    objective: str = Field(min_length=1)
    compute_budget_gpu_hours: float | None = Field(default=None, ge=0)
    max_recommendations: int = Field(default=3, ge=1, le=20)
    target_metric: str | None = None
    model_family: str | None = None
    dataset_id: str | None = None
    include_disputed_evidence: bool = False
    """Off by default: disputed evidence is qualified, never silently trusted."""


class RecommendationDecisionRequest(BaseModel):
    """FR-9: recommendations are advisory, so a human decides."""

    model_config = ConfigDict(extra="ignore")

    decision: Decision
    decided_by: str = Field(min_length=1)
    note: str | None = None
    edited_configuration: dict[str, Any] | None = None
    edited_hypothesis: str | None = None


class RecommendationDecision(Record):
    id: str
    recommendation_id: str
    decision: Decision
    decided_by: str
    note: str | None = None
    edited_configuration: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Errors (PRD 9: structured errors on every endpoint)
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    detail: list[ErrorDetail] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Person A: Graph, Lineage, Ingestion & Correction Contracts
# ---------------------------------------------------------------------------

class MetricModel(MetricValue):
    pass


class ConfigurationModel(BaseModel):
    id: str | None = None
    canonical_json: str | None = None
    config_hash: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict, description="Arbitrary model-specific parameters")


class ExperimentDetail(ExperimentSummary):
    description: str | None = None
    configuration: dict[str, Any] | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    random_seed: int | None = None
    source_reference: str | None = None
    notes: str | None = None
    parent_experiment_ids: list[str] = Field(default_factory=list)
    child_experiment_ids: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    review_items: list[dict[str, Any]] = Field(default_factory=list)
    quality_status: str | None = "valid"


class LineageNode(BaseModel):
    id: str
    name: str
    label: str = "Experiment"
    model_family: str | None = None
    status: str | None = None
    quality_status: str | None = None
    dataset_version_id: str | None = None
    depth: int = 0
    is_root: bool = False


class LineageEdge(BaseModel):
    source: str
    target: str
    relationship_type: str = "PARENT_OF"


class ExperimentLineageResponse(BaseModel):
    root_id: str
    nodes: list[LineageNode] = Field(default_factory=list)
    edges: list[LineageEdge] = Field(default_factory=list)


class ImportErrorItem(BaseModel):
    row: int | None = None
    experiment_id: str | None = None
    reason: str


class ImportSummary(BaseModel):
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[ImportErrorItem] = Field(default_factory=list)


class ExperimentIngestRecord(BaseModel):
    id: str
    project_id: str
    name: str
    model_family: str
    status: str = "completed"
    description: str | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    dataset_version_id: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: list[MetricModel] = Field(default_factory=list)
    duration_seconds: float | None = None
    compute_cost_gpu_hours: float | None = None
    random_seed: int | None = None
    source_reference: str | None = None
    notes: str | None = None
    parent_experiment_id: str | None = None
    created_at: str | None = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class DependencyPath(BaseModel):
    nodes: list[str] = Field(..., description="Ordered list of node IDs forming the dependency path")
    relationship_types: list[str] = Field(..., description="Ordered list of relationships traversed")
    reason: str | None = None


class CorrectionCreate(BaseModel):
    source_id: str = Field(..., description="ID of the affected node")
    source_type: str = Field(default="dataset_version", description="Type: dataset_version, experiment, evidence, claim, metric")
    description: str = Field(..., description="Reason/evidence for why this item is invalid or unreliable")
    created_by: str | None = "system"


class CorrectionImpactResponse(BaseModel):
    correction_id: str
    source_id: str
    summary: str
    directly_affected_experiment_ids: list[str] = Field(default_factory=list)
    potentially_affected_claim_ids: list[str] = Field(default_factory=list)
    unaffected_experiment_ids: list[str] = Field(default_factory=list)
    review_item_ids: list[str] = Field(default_factory=list)
    paths: list[DependencyPath] = Field(default_factory=list)


class ReviewItemResponse(BaseModel):
    id: str
    experiment_id: str
    reason: str
    status: str = "NEEDS_REVIEW"
    created_at: str


class ReviewItemResolve(BaseModel):
    status: str = Field(default="RESOLVED", description="New status: RESOLVED, DISMISSED, SUPERSEDED")
    notes: str | None = None
    resolved_by: str | None = None


class ProjectCreate(BaseModel):
    id: str
    name: str
    description: str | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: str | None = None
    experiment_count: int = 0


class ProjectGraphSummary(BaseModel):
    project_id: str
    node_counts: dict[str, int] = Field(default_factory=dict)
    relationship_counts: dict[str, int] = Field(default_factory=dict)


class GraphNodeData(BaseModel):
    id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeData(BaseModel):
    source: str
    target: str
    relationship_type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ProjectGraphResponse(BaseModel):
    project_id: str
    nodes: list[GraphNodeData] = Field(default_factory=list)
    edges: list[GraphEdgeData] = Field(default_factory=list)