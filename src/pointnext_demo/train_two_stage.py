"""Run PointNeXt two-stage training: stage1 (val split) then stage2 (full data)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .train import namespace_from_config, run_training
from .utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage PointNeXt training (B model).")
    parser.add_argument("--stage1-config", default="configs/pointnext_b_c64_rotate/stage1.yaml")
    parser.add_argument("--stage2-config", default="configs/pointnext_b_c64_rotate/stage2.yaml")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    cli = parser.parse_args()

    if cli.stage in ("1", "both"):
        print("=" * 60)
        print("Stage 1: train with validation split + early stopping")
        print("=" * 60)
        args1 = namespace_from_config(cli.stage1_config)
        run_training(args1)
        if not (Path(args1.out_dir) / "best.pt").is_file():
            raise FileNotFoundError(f"stage 1 did not produce {args1.out_dir}/best.pt")

    if cli.stage in ("2", "both"):
        print("=" * 60)
        print("Stage 2: full-data fine-tune from stage 1 checkpoint")
        print("=" * 60)
        cfg2 = load_config(cli.stage2_config)
        resume = cfg2.get("resume_checkpoint")
        if not resume or not Path(resume).is_file():
            raise FileNotFoundError(f"stage 2 resume_checkpoint not found: {resume}")
        args2 = namespace_from_config(cli.stage2_config)
        run_training(args2)

    print("two-stage training finished.")
    if cli.stage in ("2", "both"):
        print(f"predict: python -m src.pointnext_demo.predict --config {cli.stage2_config}")


if __name__ == "__main__":
    main()
