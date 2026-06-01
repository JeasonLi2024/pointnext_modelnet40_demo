"""Run test-set prediction for multiple checkpoints and print a comparison table."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICT_CFG = ROOT / "configs/predict_selected_model/predict.yaml"
PYTHON = ROOT / "venv/Scripts/python.exe"

MODELS = [
    {
        "name": "pointnext_s_c64_base_v2",
        "checkpoint": "runs/pointnext_s_c64_base_v2/best.pt",
        "out_csv": "runs/predict_compare/pointnext_s_c64_base_v2_test.csv",
        "variant": "s",
        "width": 64,
        "votes": 1,
        "predict_batch_size": 64,
    },
    {
        "name": "pointnext_b_c64_no_rotate_stage2",
        "checkpoint": "runs/pointnext_b_c64_no_rotate_stage2/best.pt",
        "out_csv": "runs/predict_compare/pointnext_b_c64_no_rotate_stage2_test.csv",
        "variant": "b",
        "width": 64,
        "votes": 1,
        "predict_batch_size": 32,
    },
    {
        "name": "pointnext_b_c64_rotate_stage2",
        "checkpoint": "runs/pointnext_b_c64_rotate_stage2/best.pt",
        "out_csv": "runs/predict_compare/pointnext_b_c64_rotate_stage2_test.csv",
        "variant": "b",
        "width": 64,
        "votes": 3,
        "predict_batch_size": 32,
    },
    {
        "name": "pointnext_b_c96_no_rotate_stage2",
        "checkpoint": "runs/pointnext_b_c96_no_rotate_stage2/best.pt",
        "out_csv": "runs/predict_compare/pointnext_b_c96_no_rotate_stage2_test.csv",
        "variant": "b",
        "width": 96,
        "votes": 1,
        "predict_batch_size": 24,
    },
]


def parse_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        if "Test Instance Accuracy:" in line:
            metrics["instance_acc"] = float(line.split(":")[1].split("(")[0].strip())
        if "Class Accuracy:" in line:
            metrics["class_acc"] = float(line.split(":")[1].split("(")[0].strip())
        if "device=" in line and "predict:" in line:
            metrics["device"] = line.split("device=")[-1].strip()
    return metrics


def main() -> None:
    if not PYTHON.is_file():
        raise SystemExit(f"venv python not found: {PYTHON}")

    (ROOT / "runs/predict_compare").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for spec in MODELS:
        cmd = [
            str(PYTHON),
            "-m",
            "src.pointnext_demo.predict",
            "--config",
            str(PREDICT_CFG),
            "--checkpoint",
            spec["checkpoint"],
            "--out-csv",
            spec["out_csv"],
            "--variant",
            spec["variant"],
            "--width",
            str(spec["width"]),
            "--num-points",
            "2048",
            "--votes",
            str(spec["votes"]),
            "--batch-size",
            str(spec["predict_batch_size"]),
            "--use-gpu",
            "--eval",
        ]
        print("\n" + "=" * 72)
        print("RUN", spec["name"], f"votes={spec['votes']}")
        print("=" * 72)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(proc.returncode)
        row = {"name": spec["name"], "votes": spec["votes"], "out_csv": spec["out_csv"], **parse_metrics(proc.stdout)}
        results.append(row)

    print("\n" + "=" * 72)
    print("COMPARISON (sorted by Class Accuracy)")
    print("=" * 72)
    results.sort(key=lambda r: (r.get("class_acc", 0.0), r.get("instance_acc", 0.0)), reverse=True)
    print(f"{'Model':<42} {'Votes':>5} {'Instance':>10} {'Class':>10} {'Device':>8}")
    print("-" * 72)
    for row in results:
        print(
            f"{row['name']:<42} {row['votes']:>5} "
            f"{row.get('instance_acc', 0):>10.4f} {row.get('class_acc', 0):>10.4f} "
            f"{row.get('device', '?'):>8}"
        )
    best = results[0]
    print("\nBest:", best["name"], f"instance={best.get('instance_acc')}", f"class={best.get('class_acc')}")
    print("CSV:", best["out_csv"])


if __name__ == "__main__":
    main()
