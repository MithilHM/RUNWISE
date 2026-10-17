"""Probe which Cypher constructs this FalkorDB build accepts.

FalkorDB 4.22 rejects EXISTS subqueries and pattern comprehensions, so the
repository has to express every filter with constructs the server actually supports.
This script is the evidence for those choices; re-run it after a server upgrade.

    python scripts/probe_cypher_support.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.graph.client import graph_client  # noqa: E402

GRAPH = graph_client.get_graph()
PARAMS = {
    "pid": "project_demo",
    "ds": "tabular_fraud",
    "metric": "VALIDATION_ACCURACY",
    "split": "validation",
    "fid": "exp_001",
}

PROBES = {
    "pattern predicate in WHERE": """
        MATCH (e:Experiment)
        WHERE (e)-[:USES_DATASET_VERSION]->(:DatasetVersion {dataset_id: $ds})
        RETURN count(e)
    """,
    "EXISTS subquery": """
        MATCH (e:Experiment)
        WHERE EXISTS { MATCH (e)-[:PRODUCES_METRIC]->(:Metric {name: 'validation_accuracy'}) }
        RETURN count(e)
    """,
    "exists() function": """
        MATCH (e:Experiment)
        WHERE exists((e)-[:PRODUCES_METRIC]->(:Metric {name: 'validation_accuracy'}))
        RETURN count(e)
    """,
    "pattern comprehension": """
        MATCH (e:Experiment)
        WHERE size([(e)-[:PRODUCES_METRIC]->(m:Metric) WHERE m.name = 'validation_accuracy' | m]) > 0
        RETURN count(e)
    """,
    "multi-type relationship": """
        MATCH (co:Correction)-[:AFFECTS|CHALLENGES]->(t)
        RETURN count(co)
    """,
    "any() over a collected list": """
        MATCH (co:Correction)
        OPTIONAL MATCH (co)-[:AFFECTS|CHALLENGES]->(t)
        WITH co, collect(t) AS targets
        WHERE co.project_id = $pid OR any(x IN targets WHERE x.project_id IS NOT NULL)
        RETURN count(co)
    """,
    "case-insensitive metric aggregate filter": """
        MATCH (e:Experiment)
        OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
        WITH e, sum(CASE WHEN toLower(m.name) = toLower($metric) THEN 1 ELSE 0 END) AS hits
        WHERE hits > 0
        RETURN count(e) AS matched
    """,
    "split aggregate filter": """
        MATCH (e:Experiment)
        OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
        WITH e, sum(CASE WHEN m.evaluation_split = $split THEN 1 ELSE 0 END) AS hits
        WHERE hits > 0
        RETURN count(e) AS matched
    """,
    "optional match then delete (null-safe edge replace)": """
        MATCH (e:Experiment {id: $fid})
        OPTIONAL MATCH (e)-[r:USES_CONFIG]->(:Configuration)
        WITH e, r LIMIT 1
        RETURN count(e) AS matched
    """,
    "union of two traversals": """
        MATCH (co:Correction {id: 'nope'})-[:AFFECTS|CHALLENGES]->(t:DatasetVersion)
        MATCH (e:Experiment)-[:USES_DATASET_VERSION]->(t)
        RETURN e.id
        UNION
        MATCH (co:Correction {id: 'nope'})-[:AFFECTS]->(x:Experiment)
        RETURN x.id
    """,
    "unwind relationships on a path": """
        MATCH path = (root:Experiment {id: $fid})-[:PARENT_OF*0..3]-(other:Experiment)
        UNWIND relationships(path) AS rel
        RETURN DISTINCT startNode(rel).id, type(rel)
    """,
    "coalesce ordering key": """
        MATCH (e:Experiment)
        RETURN e.id ORDER BY coalesce(e.created_at, e.ingested_at) DESC LIMIT 3
    """,
}


def main() -> int:
    print(f"probing FalkorDB at {GRAPH}\n")
    unsupported = 0
    for name, query in PROBES.items():
        try:
            result = GRAPH.query(query, PARAMS).result_set
            head = str(result)[:70]
            print(f"  ok       {name}: {head}")
        except Exception as error:  # noqa: BLE001 - the point is to report anything
            unsupported += 1
            message = str(error).replace("\n", " ")[:100]
            print(f"  UNSUPPORTED {name}: {message}")
    print(f"\n{len(PROBES) - unsupported}/{len(PROBES)} constructs supported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
