"""Runs the whole data + model pipeline end to end.

    python run_pipeline.py

Equivalent to running, in order:
    python src/generate_data.py
    python src/eda.py
    python src/train_model.py

After this finishes the UI is ready: `streamlit run app.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Data Layer -- generating synthetic dataset", "src/generate_data.py"),
    ("Data Layer -- EDA and preprocessing checks", "src/eda.py"),
    ("Model Layer -- training, evaluation, interpretation", "src/train_model.py"),
]


def main() -> int:
    for i, (title, script) in enumerate(STEPS, start=1):
        print("\n" + "=" * 78)
        print(f"STEP {i}/{len(STEPS)}  {title}")
        print("=" * 78)
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStep {i} failed ({script}). Stopping.")
            return result.returncode

    print("\n" + "=" * 78)
    print("Pipeline complete.")
    print("=" * 78)
    print("  Dataset  -> data/credit_risk_synthetic.csv")
    print("  Model    -> models/logistic_model.joblib")
    print("  Report   -> outputs/evaluation_report.md")
    print("  Figures  -> outputs/figures/")
    print("\nNext, launch the UI:")
    print("  streamlit run app.py    ->  http://localhost:8501")
    print("\nOr score from the command line:  python demo_tool.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
