from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ModelNetLikeDataset, collate_batch
from .model import build_model
from .utils import load_config, load_labels, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ModelNet40 classes and export CSV.")
    parser.add_argument("--config", default="configs/pointnext_s_c64.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--variant", choices=["s", "b"], default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--nsample", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--use-normals", dest="use_normals", action="store_true", default=None)
    parser.add_argument("--no-normals", dest="use_normals", action="store_false")
    parser.add_argument("--votes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None)
    parser.add_argument("--cpu", dest="use_gpu", action="store_false")
    args = parser.parse_args()
    config = {
        "data_root": "modelnet40_train_data/modelnet40_normal_resampled",
        "split": "test",
        "labels": "labels/modelnet40.txt",
        "out_csv": "submit.csv",
        "votes": 10,
        "batch_size": 16,
        "num_workers": 0,
        "seed": 42,
        "use_gpu": True,
    }
    train_config = load_config(args.config)
    if "require_cuda" in train_config and "use_gpu" not in train_config:
        train_config["use_gpu"] = bool(train_config.pop("require_cuda"))
    else:
        train_config.pop("require_cuda", None)
    config.update({key: value for key, value in train_config.items() if key in config})
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value
    config["config"] = args.config
    return argparse.Namespace(**config)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    labels = ckpt.get("labels") or load_labels(args.labels)
    train_args = ckpt.get("args", {})
    variant = args.variant or train_args.get("variant", "s")
    width = args.width or int(train_args.get("width", 32))
    nsample = args.nsample or int(train_args.get("nsample", 32))
    num_points = args.num_points or int(train_args.get("num_points", 1024))
    use_normals = bool(train_args.get("use_normals", True)) if args.use_normals is None else args.use_normals

    dataset = ModelNetLikeDataset(
        args.data_root,
        split=args.split,
        labels=labels,
        num_points=num_points,
        use_normals=use_normals,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    device = select_device(args.use_gpu)
    model = build_model(
        variant,
        num_classes=len(labels),
        use_normals=use_normals,
        width=width,
        nsample=nsample,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows: list[tuple[str, str]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            points = batch["points"].to(device)
            probs = torch.zeros(points.shape[0], len(labels), device=device)
            for vote_idx in range(args.votes):
                vote_points = points.clone()
                if vote_idx > 0:
                    theta = torch.rand(points.shape[0], device=device) * 2 * np.pi
                    c, s = torch.cos(theta), torch.sin(theta)
                    x = vote_points[:, :, 0].clone()
                    z = vote_points[:, :, 2].clone()
                    vote_points[:, :, 0] = c[:, None] * x + s[:, None] * z
                    vote_points[:, :, 2] = -s[:, None] * x + c[:, None] * z
                    if vote_points.shape[-1] >= 6:
                        nx = vote_points[:, :, 3].clone()
                        nz = vote_points[:, :, 5].clone()
                        vote_points[:, :, 3] = c[:, None] * nx + s[:, None] * nz
                        vote_points[:, :, 5] = -s[:, None] * nx + c[:, None] * nz
                probs += F.softmax(model(vote_points), dim=1)
            pred = probs.argmax(dim=1).cpu().tolist()
            rows.extend((sample_id, labels[class_idx]) for sample_id, class_idx in zip(batch["id"], pred))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        for sample_id, label in rows:
            f.write(f"{sample_id},{label}\n")
    print(f"saved {len(rows)} predictions to {out_csv}")


if __name__ == "__main__":
    main()
