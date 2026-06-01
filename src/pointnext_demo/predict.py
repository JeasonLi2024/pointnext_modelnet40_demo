from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ModelNetLikeDataset, collate_batch
from .inference import forward_vote_probs, load_classifier_checkpoint
from .utils import load_config, load_labels, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ModelNet40 classes and export CSV.")
    parser.add_argument("--config", default="configs/pointnext_s_c64.yaml")
    parser.add_argument("--data-root", default=None, help="Override test_data_root or data_root from config.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--variant", choices=["s", "b"], default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--nsample", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None, help="Override predict_num_points / num_points.")
    parser.add_argument("--use-normals", dest="use_normals", action="store_true", default=None)
    parser.add_argument("--no-normals", dest="use_normals", action="store_false")
    parser.add_argument("--votes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override predict_batch_size / batch_size.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None)
    parser.add_argument("--cpu", dest="use_gpu", action="store_false")
    parser.add_argument("--eval", dest="eval_on_test", action="store_true", default=None)
    parser.add_argument("--no-eval", dest="eval_on_test", action="store_false")
    cli = parser.parse_args()
    cli_num_points = cli.num_points
    cli_batch_size = cli.batch_size

    config = {
        "data_root": "modelnet40_train_data/modelnet40_normal_resampled",
        "test_data_root": "modelnet40_test_data/modelnet40_normal_resampled",
        "test_split": "test",
        "labels": "labels/modelnet40.txt",
        "out_dir": "runs/pointnext_s_c64_normals",
        "checkpoint": None,
        "out_csv": "runs/pointnext_s_c64_normals/submit.csv",
        "variant": "s",
        "width": 64,
        "nsample": 32,
        "num_points": 1024,
        "predict_num_points": None,
        "use_normals": True,
        "votes": 10,
        "batch_size": 16,
        "predict_batch_size": None,
        "num_workers": 0,
        "seed": 42,
        "use_gpu": True,
        "eval_on_test": True,
    }
    file_config = load_config(cli.config)
    if "require_cuda" in file_config and "use_gpu" not in file_config:
        file_config["use_gpu"] = bool(file_config.pop("require_cuda"))
    else:
        file_config.pop("require_cuda", None)
    config.update(file_config)
    for key, value in vars(cli).items():
        if key != "config" and value is not None:
            config[key] = value
    if config.get("checkpoint") is None:
        config["checkpoint"] = str(Path(config["out_dir"]) / "best.pt")
    config["config"] = cli.config
    ns = argparse.Namespace(**config)
    ns._cli_num_points = cli_num_points
    ns._cli_batch_size = cli_batch_size
    return ns


def resolve_predict_settings(args: argparse.Namespace, train_args: dict) -> dict:
    """CLI/config first; checkpoint fills any remaining model fields."""
    variant = args.variant or train_args.get("variant", "s")
    width = args.width or int(train_args.get("width", 64))
    nsample = args.nsample or int(train_args.get("nsample", 32))
    if args._cli_num_points is not None:
        num_points = int(args._cli_num_points)
    elif args.predict_num_points is not None:
        num_points = int(args.predict_num_points)
    else:
        num_points = int(args.num_points or train_args.get("num_points", 1024))
    if args.use_normals is None:
        use_normals = bool(train_args.get("use_normals", True))
    else:
        use_normals = args.use_normals
    if args._cli_batch_size is not None:
        batch_size = int(args._cli_batch_size)
    else:
        batch_size = int(args.predict_batch_size or args.batch_size or train_args.get("batch_size", 16))
    return {
        "variant": variant,
        "width": width,
        "nsample": nsample,
        "num_points": int(num_points),
        "use_normals": use_normals,
        "batch_size": int(batch_size),
    }


def evaluate_predictions(rows: list[tuple[str, str]], dataset: ModelNetLikeDataset, labels: list[str]) -> None:
    gt = {item.sample_id: labels[item.label] for item in dataset.items if item.label is not None}
    if not gt:
        print("eval skipped: test split has no class labels")
        return
    preds = dict(rows)
    correct = 0
    class_correct: dict[str, int] = defaultdict(int)
    class_total: dict[str, int] = defaultdict(int)
    for sample_id, true_cls in gt.items():
        pred_cls = preds[sample_id]
        class_total[true_cls] += 1
        if pred_cls == true_cls:
            correct += 1
            class_correct[true_cls] += 1
    instance_acc = correct / len(gt)
    class_acc = sum(class_correct[c] / class_total[c] for c in class_total) / len(class_total)
    print(f"Test Instance Accuracy: {instance_acc:.4f} ({instance_acc * 100:.2f}%)")
    print(f"Class Accuracy: {class_acc:.4f} ({class_acc * 100:.2f}%)")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu")
    labels = ckpt.get("labels") or load_labels(args.labels)
    train_args = ckpt.get("args", {})
    settings = resolve_predict_settings(args, train_args)

    data_root = args.test_data_root or args.data_root
    dataset = ModelNetLikeDataset(
        data_root,
        split=args.test_split,
        labels=labels,
        num_points=settings["num_points"],
        use_normals=settings["use_normals"],
        train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    device = select_device(args.use_gpu)
    model, _ = load_classifier_checkpoint(
        checkpoint,
        variant=settings["variant"],
        width=settings["width"],
        nsample=settings["nsample"],
        use_normals=settings["use_normals"],
        num_classes=len(labels),
        device=device,
    )

    print(
        f"predict: checkpoint={checkpoint} data_root={data_root} split={args.test_split} "
        f"num_points={settings['num_points']} votes={args.votes} batch_size={settings['batch_size']} device={device}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    rows: list[tuple[str, str]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            points = batch["points"].to(device)
            probs = forward_vote_probs(model, points, args.votes)
            pred = probs.argmax(dim=1).cpu().tolist()
            rows.extend((sample_id, labels[class_idx]) for sample_id, class_idx in zip(batch["id"], pred))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        for sample_id, label in rows:
            f.write(f"{sample_id},{label}\n")
    print(f"saved {len(rows)} predictions to {out_csv}")

    if args.eval_on_test:
        evaluate_predictions(rows, dataset, labels)


if __name__ == "__main__":
    main()
