"""Drop the RUNWISE graph so a demo can start from a known state.

PRD 15 requires a demo reset procedure. This is destructive: it deletes every
Project, Experiment, Metric, Correction and ReviewItem node in the configured graph.
Nothing else is touched, and no other database is involved.

    python scripts/reset_graph.py            # drop the configured graph
    python scripts/reset_graph.py --yes      # skip the confirmation prompt
    python scripts/reset_graph.py --seed     # drop, then re-seed demo data
"""

from __future__ import annotations

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.core.config import settings  # noqa: E402
from app.graph.client import graph_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset the RUNWISE graph.")
    parser.add_argument("--yes", action="store_true", help="Do not prompt for confirmation.")
    parser.add_argument("--seed", action="store_true", help="Re-seed demo data after the reset.")
    args = parser.parse_args()

    health = graph_client.check_health()
    if health.get("status") != "healthy":
        print(f"Cannot reach FalkorDB at {settings.FALKORDB_HOST}:{settings.FALKORDB_PORT}: {health}")
        print("Start it with `docker compose up -d` from the repository root.")
        return 1

    graph_name = settings.FALKORDB_GRAPH
    if not args.yes:
        answer = input(f"This deletes every node in graph {graph_name!r}. Type 'reset' to continue: ")
        if answer.strip().lower() != "reset":
            print("Cancelled. Nothing was deleted.")
            return 0

    dropped = graph_client.delete_graph(graph_name)
    if not dropped:
        print(f"Could not drop graph {graph_name!r}. It may not exist, which is the desired state anyway.")
    else:
        print(f"Dropped graph {graph_name!r}.")

    if args.seed:
        from seed_demo import run_seed  # noqa: PLC0415

        run_seed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
