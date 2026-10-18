"""
Seed script to populate FalkorDB with synthetic experiment data and demonstrate
lineage and correction propagation.
"""

import sys
import os
import logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure backend and project root are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.graph.client import graph_client
from app.graph.schema import initialize_schema
from app.services.ingestion import IngestionService
from app.services.impact_analysis import ImpactAnalysisService
from app.services.seed_data import generate_synthetic_experiments
from app.models.contracts import CorrectionCreate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed")


def run_seed(reset: bool = False):
    logger.info(f"Connecting to FalkorDB on {settings.FALKORDB_HOST}:{settings.FALKORDB_PORT} (graph: {settings.FALKORDB_GRAPH})...")
    health = graph_client.check_health()
    if health.get("status") != "healthy":
        logger.error(f"Cannot connect to FalkorDB: {health}")
        sys.exit(1)

    if reset:
        logger.warning(f"--reset: dropping graph {settings.FALKORDB_GRAPH!r} before seeding")
        graph_client.delete_graph(settings.FALKORDB_GRAPH)

    logger.info("Initializing graph schema and indices...")
    graph = graph_client.get_graph()
    init_res = initialize_schema(graph)
    logger.info(f"Schema initialized: {init_res}")

    total = 53
    logger.info(f"Generating {total} synthetic experiments across multiple model families...")
    project_id = "project_demo"
    experiments = generate_synthetic_experiments(project_id=project_id, total=total)

    ingestion = IngestionService()
    logger.info(f"Ingesting {len(experiments)} experiments into FalkorDB...")
    summary = ingestion.ingest_records(experiments, project_id=project_id, project_name="RUNWISE Demonstration Project")
    logger.info(f"Ingestion complete: {summary.imported} imported, {summary.skipped} skipped, {summary.failed} failed.")
    if summary.errors:
        for item in summary.errors[:10]:
            logger.warning(f"  row {item.row}: {item.reason}")

    # Demonstration of correction impact propagation.
    # Guarded so a re-run does not pile up duplicate corrections and review items:
    # the demo state must be the same every time, or "run this and look" stops
    # meaning anything.
    logger.info("\n--- Demonstrating Evidence Correction Impact Workflow ---")
    impact_svc = ImpactAnalysisService()
    source_id = "tabular_fraud_v1"
    existing = graph.query(
        "MATCH (co:Correction) WHERE co.source_id = $sid RETURN co.id",
        {"sid": source_id},
    ).result_set
    if existing:
        logger.info(f"Correction for {source_id} already recorded as {existing[0][0]}; reusing it.")
        impact = impact_svc.get_correction_impact(existing[0][0])
    else:
        impact = impact_svc.analyze_and_apply_correction(
            CorrectionCreate(
                source_id=source_id,
                source_type="dataset_version",
                description="Data leakage identified between train and validation splits in tabular_fraud v1.",
                created_by="researcher_alice",
            ),
            project_id=project_id,
        )
    logger.info(f"Correction Summary: {impact.summary}")
    logger.info(f"Directly Affected Experiments: {impact.directly_affected_experiment_ids}")
    logger.info(f"Downstream (indirect) Experiments: {impact.indirectly_affected_experiment_ids}")
    logger.info(f"Unaffected Experiments Count: {len(impact.unaffected_experiment_ids)}")
    logger.info(f"Review Items Created: {impact.review_item_ids}")
    logger.info(f"Sample Dependency Path: {impact.paths[0] if impact.paths else 'None'}")
    logger.info("Seed and demonstration completed successfully!")


if __name__ == "__main__":
    run_seed(reset="--reset" in sys.argv)
