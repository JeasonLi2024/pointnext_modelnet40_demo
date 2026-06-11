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
        "predict_batch_size": None,
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
        "architecture": member.get("architecture"),
        "width": member.get("width"),
        "nsample": member.get("nsample"),
        "use_normals": member.get("use_normals", True),
        "votes": int(member.get("votes", 1)),
        "weight": float(member.get("weight", 1.0)),
        "num_points": int(member.get("predict_num_points", defaults.predict_num_points)),
    }


def resolve_ensemble_batch_size(args: argparse.Namespace, member_specs: list[dict]) -> int:
    if args.predict_batch_size is not None:
        return int(args.predict_batch_size)
    per_member = [
        int(m.get("predict_batch_size", m.get("batch_size", 32)))
        for m in args.members
    ]
    return min(per_member)


def members_share_inputs(member_specs: list[dict]) -> bool:
    ref = member_specs[0]
    return all(
        s["num_points"] == ref["num_points"] and s["use_normals"] == ref["use_normals"]
        for s in member_specs
    )


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

    ensemble_batch_size = resolve_ensemble_batch_size(args, member_specs)
    shared_inputs = members_share_inputs(member_specs)

    print(f"ensemble: {len(member_specs)} models, data_root={data_root}, device={device}")
    print(f"ensemble batch_size={ensemble_batch_size} (shared_inputs={shared_inputs})")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    models: list[tuple[dict, torch.nn.Module]] = []
    member_datasets: list[ModelNetLikeDataset] = []

    for spec in member_specs:
        model, arch = load_classifier_checkpoint(
            spec["checkpoint"],
            variant=spec["variant"],
            width=spec["width"],
            nsample=spec["nsample"],
            use_normals=spec["use_normals"],
            architecture=spec["architecture"],
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
        print(
            f"  + {spec['name']}: {spec['checkpoint']} "
            f"variant={arch['variant']} width={arch['width']} points={spec['num_points']} "
            f"architecture={arch['architecture']} votes={spec['votes']} weight={spec['weight']}"
        )
        if member_datasets and [item.sample_id for item in dataset.items] != [
            item.sample_id for item in member_datasets[0].items
        ]:
            raise ValueError("test sample order/id mismatch between ensemble members")
        models.append((spec, model))
        member_datasets.append(dataset)

    ref_dataset = member_datasets[0]
    if shared_inputs:
        loader = DataLoader(
            ref_dataset,
            batch_size=ensemble_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )
    else:
        loader = DataLoader(
            ref_dataset,
            batch_size=ensemble_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )
        member_id_maps = [
            {item.sample_id: i for i, item in enumerate(ds.items)} for ds in member_datasets
        ]

    weight_sum = sum(spec["weight"] for spec, _ in models)
    rows: list[tuple[str, str]] = []

    with torch.no_grad():
        for ref_batch in tqdm(loader, desc="ensemble"):
            ensemble_probs = None
            sample_ids = ref_batch["id"]
            for member_idx, (spec, model) in enumerate(models):
                if shared_inputs:
                    batch = ref_batch
                else:
                    ds = member_datasets[member_idx]
                    idx_map = member_id_maps[member_idx]
                    indices = [idx_map[sample_id] for sample_id in sample_ids]
                    batch = collate_batch([ds[i] for i in indices])

                points = batch["points"].to(device)
                if points.shape[-1] == 6 and not spec["use_normals"]:
                    points = points[:, :, :3]
                elif points.shape[-1] == 3 and spec["use_normals"]:
                    raise ValueError(
                        f"member {spec['name']} expects normals but batch has only xyz channels"
                    )

                probs = forward_vote_probs(model, points, votes=spec["votes"])
                weighted = probs * spec["weight"]
                ensemble_probs = weighted if ensemble_probs is None else ensemble_probs + weighted

            assert ensemble_probs is not None
            ensemble_probs = ensemble_probs / weight_sum
            pred = ensemble_probs.argmax(dim=1).cpu().tolist()
            rows.extend(
                (sample_id, labels[class_idx])
                for sample_id, class_idx in zip(sample_ids, pred)
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
