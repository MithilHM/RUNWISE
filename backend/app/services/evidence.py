"""Evidence grounding and provenance (FR-7, FR-8).

Area: evidence grounding and provenance.

This module is the guard against the failure mode the PRD names first in its
trust section: a copilot that states something plausible but unbacked. Two rules
are enforced here.

1. Nothing is cited unless the backend returned it. :class:`EvidenceLedger`
   holds the set of records a response was built from; any identifier that is not
   in that set is an unknown citation and fails the grounding report.
2. Epistemic status is never inflated. A measurement is OBSERVED, a researcher's
   note is USER_ASSERTED, and anything this product derives is INFERRED. Nothing
   becomes CONFIRMED without a human approval, which this module cannot grant.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from app.models.contracts import (
    Claim,
    ClaimPath,
    ClaimStatus,
    EvidenceRef,
    EvidenceStatus,
    ExcludedEvidence,
    ExperimentRecord,
    GroundingReport,
    MetricValue,
)


class GroundingViolation(RuntimeError):
    """Raised when a response cites identifiers that were never retrieved."""

    def __init__(self, unknown_citations: Sequence[str]) -> None:
        self.unknown_citations = list(unknown_citations)
        super().__init__(
            "response cited identifiers that are not in the retrieved evidence: "
            + ", ".join(self.unknown_citations)
        )


@dataclass
class EvidenceLedger:
    """The records a single response is allowed to cite.

    Built from what the graph service returned for this request. The ledger is
    request-scoped on purpose: a later request that retrieves different data gets
    a different ledger, so a stale citation cannot pass unnoticed.
    """

    experiments: dict[str, ExperimentRecord] = field(default_factory=dict)
    corrections: dict[str, str] = field(default_factory=dict)
    """Correction id -> the source identifier it challenges."""

    def register(self, experiments: Iterable[ExperimentRecord]) -> None:
        for experiment in experiments:
            self.experiments[experiment.id] = experiment

    @property
    def known_ids(self) -> set[str]:
        return set(self.experiments)

    def contains(self, experiment_id: str) -> bool:
        return experiment_id in self.experiments

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self.experiments.get(experiment_id)

    # -- citation ----------------------------------------------------------

    def ref(
        self,
        experiment_id: str,
        metric: MetricValue | str | None = None,
        *,
        evidence_status: EvidenceStatus = EvidenceStatus.OBSERVED,
    ) -> EvidenceRef:
        """Build a citation to a known experiment, optionally one of its metrics."""
        if experiment_id not in self.experiments:
            raise GroundingViolation([experiment_id])

        experiment = self.experiments[experiment_id]
        if isinstance(metric, str):
            found = experiment.metric(metric)
        else:
            found = metric

        return EvidenceRef(
            experiment_id=experiment_id,
            metric_name=found.name if found else None,
            metric_value=found.value if found else None,
            source_reference=experiment.source_reference,
            evidence_status=evidence_status,
        )

    def claim(
        self,
        claim_id: str,
        text: str,
        support: Sequence[str | tuple[str, str]],
        *,
        status: ClaimStatus = ClaimStatus.SUPPORTED,
        evidence_status: EvidenceStatus = EvidenceStatus.OBSERVED,
        detail: str | None = None,
    ) -> Claim:
        """Build a claim whose support must resolve against this ledger.

        ``support`` entries are an experiment id, or an ``(experiment_id,
        metric_name)`` pair. An unknown id raises, so an unbacked claim cannot be
        constructed in the first place.
        """
        refs: list[EvidenceRef] = []
        for item in support:
            if isinstance(item, tuple):
                experiment_id, metric_name = item
                refs.append(self.ref(experiment_id, metric_name, evidence_status=evidence_status))
            else:
                refs.append(self.ref(item, evidence_status=evidence_status))

        return Claim(
            id=claim_id,
            text=text,
            status=status,
            evidence_status=evidence_status,
            support=refs,
            detail=detail,
        )

    # -- verification ------------------------------------------------------

    def report(self, cited_experiment_ids: Iterable[str]) -> GroundingReport:
        """Check citations against the ledger without raising."""
        cited = sorted({cid for cid in cited_experiment_ids if cid})
        unknown = sorted(cid for cid in cited if cid not in self.experiments)
        grounded_count = len(cited) - len(unknown)
        coverage = round(grounded_count / len(cited), 4) if cited else 0.0
        return GroundingReport(
            cited_experiment_ids=cited,
            unknown_citations=unknown,
            available_experiment_count=len(self.experiments),
            citation_coverage=coverage,
            is_grounded=not unknown,
        )

    def verify(self, cited_experiment_ids: Iterable[str]) -> GroundingReport:
        """Like :meth:`report`, but raises when a citation is unknown."""
        report = self.report(cited_experiment_ids)
        if not report.is_grounded:
            raise GroundingViolation(report.unknown_citations)
        return report


# ---------------------------------------------------------------------------
# Epistemic status
# ---------------------------------------------------------------------------


#: Grades that must not be presented as plain observations.
UNTRUSTED_QUALITY_STATUSES: frozenset[EvidenceStatus] = frozenset(
    {
        EvidenceStatus.DISPUTED,
        EvidenceStatus.SUPERSEDED,
        EvidenceStatus.UNRESOLVED,
    }
)

#: Wording used when explaining a withheld record.
UNTRUSTED_QUALITY_DETAIL: dict[EvidenceStatus, str] = {
    EvidenceStatus.DISPUTED: "is disputed",
    EvidenceStatus.SUPERSEDED: "has been superseded and is kept for history only",
    EvidenceStatus.UNRESOLVED: "is unresolved and awaiting a decision",
}


def evidence_status_for(
    experiment: ExperimentRecord,
    *,
    disputed_experiment_ids: Iterable[str] = (),
    corrected_dataset_version_ids: Iterable[str] = (),
) -> EvidenceStatus:
    """Grade one record's trustworthiness (FR-7, FR-12).

    Precedence: a record the researcher has stopped trusting is never presented as
    a plain observation. ``SUPERSEDED`` and ``UNRESOLVED`` are returned as-is
    rather than collapsed into ``OBSERVED``: collapsing them would promote
    precisely the grades the contract says not to trust.
    """
    disputed = set(disputed_experiment_ids)
    corrected_versions = set(corrected_dataset_version_ids)

    if experiment.id in disputed:
        return EvidenceStatus.DISPUTED
    if experiment.dataset_version_id and experiment.dataset_version_id in corrected_versions:
        return EvidenceStatus.DISPUTED

    quality = experiment.dataset_quality_status
    if quality is not None and quality is not EvidenceStatus.OBSERVED:
        # A genuine grade (CONFIRMED, USER_ASSERTED, DISPUTED, SUPERSEDED,
        # UNRESOLVED) passes through rather than being inferred.
        return quality
    return EvidenceStatus.OBSERVED


def partition_by_trust(
    experiments: Sequence[ExperimentRecord],
    *,
    disputed_experiment_ids: Iterable[str] = (),
    corrected_dataset_version_ids: Iterable[str] = (),
) -> tuple[list[ExperimentRecord], list[ExcludedEvidence], dict[str, EvidenceStatus]]:
    """Split retrieved records into trusted evidence and withheld evidence.

    Returned as ``(trusted, excluded, statuses)``. ``excluded`` always carries a
    reason, because FR-8 requires the system to state why conflicting evidence
    was left out rather than silently dropping it.
    """
    disputed = set(disputed_experiment_ids)
    corrected_versions = set(corrected_dataset_version_ids)

    trusted: list[ExperimentRecord] = []
    excluded: list[ExcludedEvidence] = []
    statuses: dict[str, EvidenceStatus] = {}

    for experiment in experiments:
        status = evidence_status_for(
            experiment,
            disputed_experiment_ids=disputed,
            corrected_dataset_version_ids=corrected_versions,
        )
        statuses[experiment.id] = status

        if status in UNTRUSTED_QUALITY_STATUSES:
            excluded.append(
                ExcludedEvidence(
                    experiment_id=experiment.id,
                    reason=f"experiment_{status.value}",
                    detail=_withheld_detail(experiment, status, disputed, corrected_versions),
                    status=status,
                )
            )
            continue

        trusted.append(experiment)

    return trusted, excluded, statuses


def _withheld_detail(
    experiment: ExperimentRecord,
    status: EvidenceStatus,
    disputed: set[str],
    corrected_versions: set[str],
) -> str:
    """Say in words why this record is not being used as support."""
    if experiment.id in disputed:
        return "A correction challenges this experiment directly."
    if experiment.dataset_version_id in corrected_versions:
        return (
            f"The dataset version {experiment.dataset_version_id} used by this experiment "
            "is under an open correction."
        )
    wording = UNTRUSTED_QUALITY_DETAIL.get(status, "is not trusted")
    return f"Its dataset version {wording}."


# ---------------------------------------------------------------------------
# Graph paths
# ---------------------------------------------------------------------------


def describe_paths(
    edges: Iterable[Mapping[str, str]],
    *,
    source_ids: Iterable[str],
    limit: int = 50,
) -> list[ClaimPath]:
    """Render lineage edges as human-readable dependency paths (FR-8).

    ``edges`` are ``{"from", "to", "type"}`` mappings from the graph service.
    Comparison and similarity links are intentionally *not* traversed: PRD 8
    requires impact to follow real dependencies, not co-membership.
    """
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        kind = edge.get("type", "RELATED")
        if not source or not target:
            continue
        adjacency.setdefault(source, []).append((target, kind))

    sources = list(source_ids)
    paths: list[ClaimPath] = []
    seen: set[tuple[str, ...]] = set()

    for source in sources:
        # Breadth-first so the shortest explanation of a dependency is shown first.
        frontier: list[tuple[str, list[str], list[str]]] = [(source, [source], [])]
        while frontier and len(paths) < limit:
            node, nodes, types = frontier.pop(0)
            if len(nodes) > 6:
                continue
            for target, kind in adjacency.get(node, []):
                if target in nodes:
                    continue
                new_nodes = [*nodes, target]
                new_types = [*types, kind]
                key = tuple(new_nodes)
                if key not in seen:
                    seen.add(key)
                    paths.append(ClaimPath(nodes=new_nodes, relationship_types=new_types))
                frontier.append((target, new_nodes, new_types))

    return paths[:limit]