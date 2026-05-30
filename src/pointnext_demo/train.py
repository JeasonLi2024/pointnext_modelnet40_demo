from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import ModelNetLikeDataset, collate_batch
from .model import build_model
from .utils import AverageMeter, load_config, load_labels, save_checkpoint, save_json, select_device, set_seed


def default_train_config() -> dict:
    return {
        "data_root": "modelnet40_train_data/modelnet40_normal_resampled",
        "labels": "labels/modelnet40.txt",
        "out_dir": "runs/pointnext_s_c64_normals",
        "variant": "s",
        "width": 64,
        "nsample": 32,
        "num_points": 1024,
        "use_normals": True,
        "random_rotate": False,
        "augment_strength": "normal",
        "epochs": 600,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 5e-2,
        "label_smoothing": 0.2,
        "val_ratio": 0.15,
        "num_workers": 0,
        "seed": 42,
        "use_gpu": True,
        "resume_checkpoint": None,
        "use_class_weights": False,
        "class_weight_power": 0.5,
        "warmup_epochs": 0,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "early_stop_metric": "val_class_acc",
    }


def namespace_from_config(config_path: str, overrides: dict | None = None) -> argparse.Namespace:
    config = default_train_config()
    file_config = load_config(config_path)
    if "require_cuda" in file_config and "use_gpu" not in file_config:
        file_config["use_gpu"] = bool(file_config.pop("require_cuda"))
    else:
        file_config.pop("require_cuda", None)
    config.update(file_config)
    if overrides:
        config.update(overrides)
    config["config"] = config_path
    return argparse.Namespace(**config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNeXt-style classifier for ModelNet40.")
    parser.add_argument("--config", default="configs/pointnext_s_c64.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--variant", choices=["s", "b"], default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--nsample", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--use-normals", dest="use_normals", action="store_true", default=None)
    parser.add_argument("--no-normals", dest="use_normals", action="store_false")
    parser.add_argument("--random-rotate", dest="random_rotate", action="store_true", default=None)
    parser.add_argument("--no-random-rotate", dest="random_rotate", action="store_false")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None)
    parser.add_argument("--cpu", dest="use_gpu", action="store_false")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--use-class-weights", dest="use_class_weights", action="store_true", default=None)
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--class-weight-power", type=float, default=None)
    parser.add_argument("--augment-strength", choices=["normal", "strong"], default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--early-stop-metric", default=None)
    cli = parser.parse_args()
    overrides = {key: value for key, value in vars(cli).items() if key != "config" and value is not None}
    return namespace_from_config(cli.config, overrides)


def stratified_split(dataset: ModelNetLikeDataset, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    labeled = [i for i, item in enumerate(dataset.items) if item.label is not None and item.label >= 0]
    if val_ratio <= 0:
        return labeled, []

    by_label: dict[int, list[int]] = defaultdict(list)
    for idx in labeled:
        by_label[dataset.items[idx].label].append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_label.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in order]
        if len(shuffled) <= 1:
            train_indices.extend(shuffled)
            continue
        val_size = max(1, int(round(len(shuffled) * val_ratio)))
        val_size = min(val_size, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_size])
        train_indices.extend(shuffled[val_size:])
    return train_indices, val_indices


def compute_class_weights(dataset: ModelNetLikeDataset, num_classes: int, power: float) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for item in dataset.items:
        if item.label is not None and item.label >= 0:
            counts[item.label] += 1
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / (num_classes * counts)) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(args: argparse.Namespace, class_weights: torch.Tensor | None, device: torch.device) -> nn.Module:
    weight = class_weights.to(device) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    warmup = int(args.warmup_epochs)
    if warmup > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def metric_value(row: dict, metric: str) -> float:
    if metric == "val_acc":
        return row["val_acc"]
    if metric == "val_class_acc":
        return row["val_class_acc"]
    if metric == "train_class_acc":
        return row["train_class_acc"]
    if metric == "train_acc":
        return row["train_acc"]
    raise ValueError(f"unknown early_stop_metric: {metric}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool, num_classes: int) -> tuple[float, float, float]:
    if loader is None:
        return 0.0, 0.0, 0.0
    model.train(train)
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, leave=False, desc="train" if train else "valid")
        for batch in pbar:
            points = batch["points"].to(device)
            labels = batch["label"].to(device)
            logits = model(points)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
            batch_correct = (pred == labels).detach().cpu()
            batch_labels = labels.detach().cpu()
            for class_idx in range(num_classes):
                mask = batch_labels == class_idx
                class_total[class_idx] += mask.sum()
                class_correct[class_idx] += batch_correct[mask].sum()
            loss_meter.update(loss.item(), labels.numel())
            instance_acc = correct / max(total, 1)
            seen = class_total > 0
            class_acc = (class_correct[seen].float() / class_total[seen].float()).mean().item() if seen.any() else 0.0
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", inst=f"{instance_acc:.4f}", cls=f"{class_acc:.4f}")
    instance_acc = correct / max(total, 1)
    seen = class_total > 0
    class_acc = (class_correct[seen].float() / class_total[seen].float()).mean().item() if seen.any() else 0.0
    return loss_meter.avg, instance_acc, class_acc


def run_training(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    labels = load_labels(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_train = args.val_ratio <= 0
    if full_train:
        args.early_stop_metric = "train_class_acc"

    train_dataset = ModelNetLikeDataset(
        args.data_root,
        split="train",
        labels=labels,
        num_points=args.num_points,
        use_normals=args.use_normals,
        train=True,
        random_rotate=args.random_rotate,
        augment_strength=args.augment_strength,
    )
    val_dataset = ModelNetLikeDataset(
        args.data_root,
        split="train",
        labels=labels,
        num_points=args.num_points,
        use_normals=args.use_normals,
        train=False,
        random_rotate=False,
        augment_strength="normal",
    )
    train_indices, val_indices = stratified_split(train_dataset, args.val_ratio, args.seed)
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices) if val_indices else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        drop_last=len(train_set) > args.batch_size,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )
        if val_set is not None
        else None
    )

    device = select_device(args.use_gpu)
    model = build_model(
        args.variant,
        num_classes=len(labels),
        use_normals=args.use_normals,
        width=args.width,
        nsample=args.nsample,
    ).to(device)

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"loaded weights from {resume_path} (epoch {ckpt.get('epoch', '?')})")

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset, len(labels), args.class_weight_power)
        print(f"class weights enabled (power={args.class_weight_power})")

    criterion = build_criterion(args, class_weights, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)

    mode = "full-train" if full_train else f"train/val={len(train_set)}/{len(val_set)}"
    print(
        f"dataset={len(train_dataset)} {mode} channels={train_dataset.num_channels} "
        f"classes={len(labels)} variant={args.variant} device={device}"
    )
    print(
        f"aug: rotate={args.random_rotate} strength={args.augment_strength} "
        f"class_weights={args.use_class_weights} early_stop={args.early_stop_patience} "
        f"metric={args.early_stop_metric}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    best_metric = float("-inf")
    best_acc = 0.0
    best_class_acc = 0.0
    patience_counter = 0
    history = []
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_class_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, num_classes=len(labels)
        )
        if val_loader is not None:
            val_loss, val_acc, val_class_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False, num_classes=len(labels)
            )
        else:
            val_loss, val_acc, val_class_acc = 0.0, 0.0, 0.0

        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_class_acc": train_class_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_class_acc": val_class_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if val_loader is not None:
            print(
                f"epoch {epoch:03d}: train_loss={train_loss:.4f} "
                f"train_instance_acc={train_acc:.4f} train_class_acc={train_class_acc:.4f} "
                f"val_loss={val_loss:.4f} val_instance_acc={val_acc:.4f} val_class_acc={val_class_acc:.4f}"
            )
        else:
            print(
                f"epoch {epoch:03d}: train_loss={train_loss:.4f} "
                f"train_instance_acc={train_acc:.4f} train_class_acc={train_class_acc:.4f} [full train]"
            )

        current = metric_value(row, args.early_stop_metric)
        if current > best_metric + args.early_stop_min_delta:
            best_metric = current
            best_acc = val_acc if val_loader is not None else train_acc
            best_class_acc = val_class_acc if val_loader is not None else train_class_acc
            patience_counter = 0
            save_checkpoint(
                out_dir / "best.pt",
                {
                    "model": model.state_dict(),
                    "labels": labels,
                    "args": vars(args),
                    "num_classes": len(labels),
                    "num_channels": train_dataset.num_channels,
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "best_acc": best_acc,
                    "best_class_acc": best_class_acc,
                    "best_metric": best_metric,
                    "early_stop_metric": args.early_stop_metric,
                    "epoch": epoch,
                },
            )
        else:
            patience_counter += 1

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(
                f"early stopping at epoch {epoch}: no improvement in {args.early_stop_metric} "
                f"for {args.early_stop_patience} epochs (best={best_metric:.4f})"
            )
            stopped_early = True
            break

    summary = {
        "history": history,
        "best_acc": best_acc,
        "best_class_acc": best_class_acc,
        "best_metric": best_metric,
        "early_stop_metric": args.early_stop_metric,
        "stopped_early": stopped_early,
        "epochs_ran": len(history),
    }
    save_json(out_dir / "history.json", summary)
    print(f"best {args.early_stop_metric}: {best_metric:.4f}")
    if val_loader is not None:
        print(f"best validation instance accuracy: {best_acc:.4f}")
        print(f"best validation class accuracy: {best_class_acc:.4f}")
    else:
        print(f"best train instance accuracy (at best checkpoint): {best_acc:.4f}")
        print(f"best train class accuracy (at best checkpoint): {best_class_acc:.4f}")
    return summary


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
