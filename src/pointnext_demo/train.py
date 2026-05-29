from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import ModelNetLikeDataset, collate_batch
from .model import build_model
from .utils import AverageMeter, load_config, load_labels, save_checkpoint, save_json, set_seed


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
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=None)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    args = parser.parse_args()
    config = {
        "data_root": "modelnet40_train_data/modelnet40_normal_resampled",
        "labels": "labels/modelnet40.txt",
        "out_dir": "runs/pointnext_s_c64_normals",
        "variant": "s",
        "width": 64,
        "nsample": 32,
        "num_points": 1024,
        "use_normals": True,
        "random_rotate": False,
        "epochs": 600,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 5e-2,
        "label_smoothing": 0.2,
        "val_ratio": 0.15,
        "num_workers": 0,
        "seed": 42,
        "require_cuda": True,
    }
    config.update(load_config(args.config))
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value
    config["config"] = args.config
    return argparse.Namespace(**config)


def stratified_split(dataset: ModelNetLikeDataset, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, item in enumerate(dataset.items):
        if item.label is None or item.label < 0:
            continue
        by_label[item.label].append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_label.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in order]
        if len(shuffled) <= 1 or val_ratio <= 0:
            train_indices.extend(shuffled)
            continue
        val_size = max(1, int(round(len(shuffled) * val_ratio)))
        val_size = min(val_size, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_size])
        train_indices.extend(shuffled[val_size:])

    if not val_indices:
        val_indices = train_indices[-1:]
        train_indices = train_indices[:-1]
    return train_indices, val_indices


def run_epoch(model, loader, criterion, optimizer, device, train: bool, num_classes: int) -> tuple[float, float, float]:
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    labels = load_labels(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ModelNetLikeDataset(
        args.data_root,
        split="train",
        labels=labels,
        num_points=args.num_points,
        use_normals=args.use_normals,
        train=True,
        random_rotate=args.random_rotate,
    )
    val_dataset = ModelNetLikeDataset(
        args.data_root,
        split="train",
        labels=labels,
        num_points=args.num_points,
        use_normals=args.use_normals,
        train=False,
    )
    train_indices, val_indices = stratified_split(train_dataset, args.val_ratio, args.seed)
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        drop_last=len(train_set) > args.batch_size,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and check `nvidia-smi`, "
            "or remove --require-cuda to train on CPU."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.variant,
        num_classes=len(labels),
        use_normals=args.use_normals,
        width=args.width,
        nsample=args.nsample,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(
        f"dataset={len(train_dataset)} train={len(train_set)} val={len(val_set)} "
        f"channels={train_dataset.num_channels} classes={len(labels)} device={device}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    best_acc = 0.0
    best_class_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_class_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, num_classes=len(labels)
        )
        val_loss, val_acc, val_class_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False, num_classes=len(labels)
        )
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
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"train_instance_acc={train_acc:.4f} train_class_acc={train_class_acc:.4f} "
            f"val_loss={val_loss:.4f} val_instance_acc={val_acc:.4f} val_class_acc={val_class_acc:.4f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_class_acc = val_class_acc
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
                    "epoch": epoch,
                },
            )

    save_json(out_dir / "history.json", {"history": history, "best_acc": best_acc, "best_class_acc": best_class_acc})
    print(f"best validation instance accuracy: {best_acc:.4f}")
    print(f"best validation class accuracy: {best_class_acc:.4f}")


if __name__ == "__main__":
    main()
