from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ModelNetLikeDataset, collate_batch
from .inference import forward_vote_probs, load_classifier_checkpoint
from .predict import evaluate_predictions
from .utils import load_config, load_labels, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble multiple checkpoints by averaging softmax probabilities."
    )
    parser.add_argument("--config", default="configs/predict_selected_model/ensemble.yaml")
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None)
    parser.add_argument("--cpu", dest="use_gpu", action="store_false")
    parser.add_argument("--eval", dest="eval_on_test", action="store_true", default=None)
    parser.add_argument("--no-eval", dest="eval_on_test", action="store_false")
    cli = parser.parse_args()

    config = {
        "labels": "labels/modelnet40.txt",
        "test_data_root": "modelnet40_test_data/modelnet40_normal_resampled",
        "test_split": "test",
        "out_csv": "runs/predict_compare/ensemble_prob_avg.csv",
        "predict_num_points": 2048,
        "num_workers": 0,
        "seed": 42,
        "use_gpu": True,
        "eval_on_test": True,
        "members": [],
    }
    config.update(load_config(cli.config))
    for key, value in vars(cli).items():
        if key != "config" and value is not None:
            config[key] = value
    config["config"] = cli.config
    return argparse.Namespace(**config)


def resolve_member_settings(member: dict, defaults: argparse.Namespace) -> dict:
    return {
        "name": member["name"],
        "checkpoint": Path(member["checkpoint"]),
        "variant": member.get("variant"),
        "width": member.get("width"),
        "nsample": member.get("nsample"),
        "use_normals": member.get("use_normals", True),
        "votes": int(member.get("votes", 1)),
        "weight": float(member.get("weight", 1.0)),
        "num_points": int(member.get("predict_num_points", defaults.predict_num_points)),
        "batch_size": int(member.get("predict_batch_size", member.get("batch_size", 32))),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    members_cfg = args.members
    if not members_cfg:
        raise ValueError("ensemble config must define at least one entry under `members`")

    labels = load_labels(args.labels)
    device = select_device(args.use_gpu)
    data_root = args.test_data_root

    member_specs = [resolve_member_settings(m, args) for m in members_cfg]
    for spec in member_specs:
        if not spec["checkpoint"].is_file():
            raise FileNotFoundError(f"checkpoint not found: {spec['checkpoint']}")

    print(f"ensemble: {len(member_specs)} models, data_root={data_root}, device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    members: list[tuple[dict, torch.nn.Module, DataLoader]] = []
    ref_dataset: ModelNetLikeDataset | None = None

    for spec in member_specs:
        model, arch = load_classifier_checkpoint(
            spec["checkpoint"],
            variant=spec["variant"],
            width=spec["width"],
            nsample=spec["nsample"],
            use_normals=spec["use_normals"],
            num_classes=len(labels),
            device=device,
        )
        dataset = ModelNetLikeDataset(
            data_root,
            split=args.test_split,
            labels=labels,
            num_points=spec["num_points"],
            use_normals=spec["use_normals"],
            train=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=spec["batch_size"],
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )
        print(
            f"  + {spec['name']}: {spec['checkpoint']} "
            f"variant={arch['variant']} width={arch['width']} points={spec['num_points']} "
            f"votes={spec['votes']} weight={spec['weight']}"
        )
        if ref_dataset is None:
            ref_dataset = dataset
        elif [item.sample_id for item in dataset.items] != [item.sample_id for item in ref_dataset.items]:
            raise ValueError("test sample order/id mismatch between ensemble members")
        members.append((spec, model, loader))

    assert ref_dataset is not None
    weight_sum = sum(spec["weight"] for spec, _, _ in members)
    rows: list[tuple[str, str]] = []

    with torch.no_grad():
        for batches in tqdm(zip(*(loader for _, _, loader in members)), desc="ensemble"):
            ensemble_probs = None
            ref_batch = batches[0]
            for (spec, model, _), batch in zip(members, batches):
                points = batch["points"].to(device)
                probs = forward_vote_probs(model, points, votes=spec["votes"])
                weighted = probs * spec["weight"]
                ensemble_probs = weighted if ensemble_probs is None else ensemble_probs + weighted

            ensemble_probs = ensemble_probs / weight_sum
            pred = ensemble_probs.argmax(dim=1).cpu().tolist()
            rows.extend(
                (sample_id, labels[class_idx])
                for sample_id, class_idx in zip(ref_batch["id"], pred)
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        for sample_id, label in rows:
            f.write(f"{sample_id},{label}\n")
    print(f"saved {len(rows)} ensemble predictions to {out_csv}")

    if args.eval_on_test:
        evaluate_predictions(rows, ref_dataset, labels)


if __name__ == "__main__":
    main()
