"""
Tests for Evidence Correction and Impact Analysis workflows.
Verifies PRD Section 5 (FR-6) and Section 20 contracts.
"""

from app.models.contracts import CorrectionCreate
from app.services.seed_data import get_demo_lineage_scenario


def test_correction_impact_propagation(test_repo, test_ingestion_service, test_impact_service):
    """
    Validates core ground-truth impact propagation scenario:
    - E1 uses D1
    - E2 uses D1 and derives from E1
    - E3 derives from E2
    - E_control uses D2 and is unaffected
    - Marking D1 unreliable impacts E1, E2, and E3 with explanatory paths,
      while E_control is untouched.
    """
    proj_id = "test_impact_proj"
    scenario = get_demo_lineage_scenario(project_id=proj_id)

    # Ingest scenario records
    summary = test_ingestion_service.ingest_records(scenario, project_id=proj_id)
    assert summary.imported == 4
    assert summary.failed == 0

    # Add a claim derived from exp_003
    test_repo.create_claim("claim_fraud_acc", "Model achieves 92.5% accuracy on fraud detection")
    test_repo.link_experiment_claim("exp_003", "claim_fraud_acc")

    # Apply correction: D1 has data leakage
    corr_payload = CorrectionCreate(
        source_id="tabular_fraud_v1",
        source_type="dataset_version",
        description="Data leakage detected between training and test folds",
        created_by="auditor"
    )

    impact = test_impact_service.analyze_and_apply_correction(corr_payload, project_id=proj_id)

    # Direct impact
    assert "exp_001" in impact.directly_affected_experiment_ids
    assert "exp_002" in impact.directly_affected_experiment_ids

    # Transitive downstream impact
    # exp_003 derived from exp_002
    all_affected = set(impact.directly_affected_experiment_ids) | {
        p.nodes[-1] for p in impact.paths if p.nodes[-1].startswith("exp_")
    }
    assert "exp_003" in all_affected

    # Unaffected control experiment
    assert "exp_control" in impact.unaffected_experiment_ids
    assert "exp_control" not in impact.directly_affected_experiment_ids

    # Check claims affected
    assert "claim_fraud_acc" in impact.potentially_affected_claim_ids

    # Check dependency paths exist and have valid structure
    assert len(impact.paths) > 0
    for path in impact.paths:
        assert len(path.nodes) >= 1
        assert path.reason is not None

    # Check ReviewItems were created in queue
    queue = test_repo.get_review_queue(project_id=proj_id)
    assert len(queue) >= 3
    review_exp_ids = {item.experiment_id for item in queue}
    assert "exp_001" in review_exp_ids
    assert "exp_002" in review_exp_ids

    # Check quality status updated conservatively to needs_review
    exp1_detail = test_repo.get_experiment("exp_001")
    assert exp1_detail.quality_status == "needs_review"

    # Control remains valid
    exp_ctrl = test_repo.get_experiment("exp_control")
    assert exp_ctrl.quality_status == "valid"


def test_duplicate_correction_idempotency(test_repo, test_ingestion_service, test_impact_service):
    """Verify applying the same correction multiple times is idempotent."""
    proj_id = "test_dup_proj"
    scenario = get_demo_lineage_scenario(project_id=proj_id)
    test_ingestion_service.ingest_records(scenario, project_id=proj_id)

    corr = CorrectionCreate(
        source_id="tabular_fraud_v1",
        source_type="dataset_version",
        description="Duplicate correction test",
        created_by="tester"
    )

    # First execution
    res1 = test_impact_service.analyze_and_apply_correction(corr, project_id=proj_id, correction_id="corr_fixed_01")
    # Second execution with same correction_id
    res2 = test_impact_service.analyze_and_apply_correction(corr, project_id=proj_id, correction_id="corr_fixed_01")

    assert res1.directly_affected_experiment_ids == res2.directly_affected_experiment_ids
    assert res1.unaffected_experiment_ids == res2.unaffected_experiment_ids


def test_resolve_review_item(test_repo, test_ingestion_service, test_impact_service):
    """Test resolving a review item from the review queue."""
    proj_id = "test_resolve_proj"
    scenario = get_demo_lineage_scenario(project_id=proj_id)
    test_ingestion_service.ingest_records(scenario, project_id=proj_id)

    corr = CorrectionCreate(
        source_id="tabular_fraud_v1",
        source_type="dataset_version",
        description="Resolution test",
        created_by="tester"
    )
    impact = test_impact_service.analyze_and_apply_correction(corr, project_id=proj_id)

    first_ri_id = impact.review_item_ids[0]
    resolved = test_repo.resolve_review_item(
        review_item_id=first_ri_id,
        status="RESOLVED",
        notes="Re-evaluated with corrected leak-free split. Performance holds.",
        resolved_by="lead_researcher"
    )
    assert resolved is not None
    assert resolved["status"] == "RESOLVED"
    assert resolved["resolved_by"] == "lead_researcher"
