"""
Test Runner & Verification Gate for RUNWISE.
Executes unit tests, API tests, and E2E integration tests.
Usage:
    python scripts/run_all_tests.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    print("=" * 60)
    print("RUNWISE — Automated Verification Suite (Person D)")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-v",
        "--tb=short"
    ]

    print(f"\nExecuting: {' '.join(cmd)} (CWD: {backend_dir})\n")
    result = subprocess.run(cmd, cwd=backend_dir)
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED: Ready for integration / deployment")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ TEST FAILURES DETECTED: Review failure logs above")
        print("=" * 60)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
