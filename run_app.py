"""
InsightOptima launcher — validates files then starts Streamlit.

Usage:
    python run_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent

    print("=" * 50)
    print("  InsightOptima — Starting...")
    print("=" * 50)

    # Step 1: verify service files on disk
    try:
        from loader import LOADER_VERSION, verify_service_files

        errors = verify_service_files()
        if errors:
            print("\nERROR: Service files incomplete:\n")
            for e in errors:
                print(f"  - {e}")
            print("\nFix: Wait for OneDrive to sync, then retry.")
            sys.exit(1)
        print(f"Loader version: {LOADER_VERSION} — all files OK")
    except Exception as exc:
        print(f"\nImport check failed: {exc}")
        sys.exit(1)

    app_path = project_root / "app.py"
    if not app_path.exists():
        print(f"\nERROR: app.py not found at {app_path}")
        sys.exit(1)

    print(f"Project folder: {project_root}")
    print(f"Running: {app_path}")

    # Step 2: start Streamlit (explicit absolute path)
    print("\nStarting Streamlit... Open http://localhost:8501 in your browser.\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            "8501",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(project_root),
    )


if __name__ == "__main__":
    main()
