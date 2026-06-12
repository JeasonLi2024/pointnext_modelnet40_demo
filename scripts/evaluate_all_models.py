"""Evaluate every runs/**/best.pt with its training configuration."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULT_COLUMNS = ["Model Name", "Test Instance Accuracy", "Class Accuracy"]

# Checkpoints keep the config path that existed when training started. These
# aliases preserve compatibility after the configs were reorganized.
CONFIG_ALIASES = {
    "configs/pointmlp_elite_c32.yaml": "configs/pointmlp/pointmlp_elite_c32.yaml",
    "configs/pointmlp_c64.yaml": "configs/pointmlp/pointmlp_c64.yaml",
    "configs/pointnext_s_c64.yaml": "configs/pointnext_s_c64/pointnext_s_c64.yaml",
    "configs/pointnext_b_c64_stage1.yaml": "configs/pointnext_b_c64_rotate/stage1_old.yaml",
    "configs/pointnext_b_c64_stage2.yaml": "configs/pointnext_b_c64_rotate/stage2_old.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all trained best.pt checkpoints and update runs/result.csv."
    )
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--result-csv", type=Path, default=ROOT / "runs" / "result.csv")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one model fails.",
    )
    return parser.parse_args()


def config_index() -> dict[str, Path]:
    by_out_dir: dict[str, Path] = {}
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out_dir = data.get("out_dir")
        if out_dir:
            by_out_dir[str(Path(out_dir)).replace("\\", "/")] = path
    return by_out_dir


def resolve_config(checkpoint: Path, train_args: dict, by_out_dir: dict[str, Path]) -> Path:
    out_dir = str(Path(train_args.get("out_dir", checkpoint.parent))).replace("\\", "/")
    if out_dir in by_out_dir:
        return by_out_dir[out_dir]

    saved_config = str(train_args.get("config", "")).replace("\\", "/")
    saved_config = CONFIG_ALIASES.get(saved_config, saved_config)
    candidate = ROOT / saved_config
    if saved_config and candidate.is_file():
        return candidate

    # The first S/C64 run used the same architecture file before its output
    # directory was changed to base_v2. Its exact model settings remain stored
    # inside best.pt and are loaded by predict.py.
    if checkpoint.parent.name == "pointnext_s_c64_normals":
        return ROOT / "configs/pointnext_s_c64/pointnext_s_c64.yaml"

    raise FileNotFoundError(f"no training config found for {checkpoint}")


def parse_metrics(output: str) -> tuple[float, float]:
    instance = re.search(r"Test Instance Accuracy:\s*([0-9.]+)", output)
    class_acc = re.search(r"Class Accuracy:\s*([0-9.]+)", output)
    if not instance or not class_acc:
        raise ValueError("prediction output does not contain both accuracy metrics")
    return float(instance.group(1)), float(class_acc.group(1))


def load_existing_results(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["Model Name"]: row for row in csv.DictReader(handle)}


def write_results(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows))


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    checkpoints = sorted(runs_dir.rglob("best.pt"))
    if not checkpoints:
        raise SystemExit(f"no best.pt files found under {runs_dir}")

    configs_by_out_dir = config_index()
    results = load_existing_results(args.result_csv)
    failures: list[str] = []

    for index, checkpoint in enumerate(checkpoints, 1):
        model_name = checkpoint.parent.name
        ckpt = torch.load(checkpoint, map_location="cpu")
        train_args = ckpt.get("args", {})
        config = resolve_config(checkpoint, train_args, configs_by_out_dir)
        predictions = checkpoint.parent / "test_predictions.csv"
        log_path = checkpoint.parent / "test_eval.log"

        command = [
            sys.executable,
            "-m",
            "src.pointnext_demo.predict",
            "--config",
            str(config.relative_to(ROOT)),
            "--checkpoint",
            str(checkpoint.relative_to(ROOT)),
            "--out-csv",
            str(predictions.relative_to(ROOT)),
            "--eval",
        ]
        command.append("--cpu" if args.cpu else "--use-gpu")

        print(f"[{index}/{len(checkpoints)}] {model_name}")
        print(f"  config={config.relative_to(ROOT)}")
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        output = process.stdout + ("\n" + process.stderr if process.stderr else "")
        log_path.write_text(output, encoding="utf-8")

        if process.returncode != 0:
            message = f"{model_name}: exit code {process.returncode}"
            failures.append(message)
            print(f"  FAILED: see {log_path.relative_to(ROOT)}")
            if args.fail_fast:
                raise SystemExit(message)
            continue

        instance_acc, class_acc = parse_metrics(output)
        results[model_name] = {
            "Model Name": model_name,
            "Test Instance Accuracy": f"{instance_acc:.4f}",
            "Class Accuracy": f"{class_acc:.4f}",
        }
        write_results(args.result_csv, results)
        print(f"  instance={instance_acc:.4f} class={class_acc:.4f}")

    write_results(args.result_csv, results)
    print(f"updated {args.result_csv}")
    if failures:
        print("failed models:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
