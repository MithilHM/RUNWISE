"""
1-click demo preparation script for RUNWISE (Person D).
Checks FalkorDB connectivity, resets graph, seeds 50 synthetic experiments,
and runs the verification smoke test.

Usage:
    python scripts/prepare_demo.py
"""

from __future__ import annotations

import os
import subprocess
import sys

# Ensure backend is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.graph.client import graph_client


def main() -> int:
    print("=" * 60)
    print("RUNWISE — 1-Click Demo Preparation")
    print("=" * 60)

    # 1. Check FalkorDB Health
    print(f"\n[1/3] Checking FalkorDB on {settings.FALKORDB_HOST}:{settings.FALKORDB_PORT}...")
    health = graph_client.check_health()
    if health.get("status") != "healthy":
        print(f"❌ FalkorDB unreachable ({health}).")
        print("   Run: docker compose up -d")
        return 1
    print("✅ FalkorDB is online.")

    # 2. Reset and Seed 50 Experiments
    print("\n[2/3] Resetting and seeding 50 synthetic runs with ground truth...")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    seed_proc = subprocess.run([sys.executable, "scripts/seed_demo.py", "--reset"], cwd=root)
    if seed_proc.returncode != 0:
        print("❌ Seeding failed.")
        return seed_proc.returncode
    print("✅ Demo data seeded successfully.")

    # 3. Optional Smoke Test Verification
    print("\n[3/3] Running smoke test gate...")
    # ponytail: run smoke_test only if backend is listening; otherwise print start command
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        smoke_proc = subprocess.run([sys.executable, "scripts/smoke_test.py"], cwd=root)
        if smoke_proc.returncode != 0:
            print("❌ Smoke test reported issues.")
            return smoke_proc.returncode
        print("✅ Smoke test passed.")
    except Exception:
        print("ℹ️ Backend server is not running on :8000 yet.")
        print("   Start server in another terminal: uvicorn app.main:app --reload --port 8000")

    print("\n" + "=" * 60)
    print("🎉 DEMO READY!")
    print("   1. Backend:  cd backend && uvicorn app.main:app --reload --port 8000")
    print("   2. Frontend: cd frontend && npm run dev")
    print("   3. Demo Guide: see docs/DEMO_SCRIPT.md")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
