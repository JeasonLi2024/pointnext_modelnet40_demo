from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import build_model


def forward_vote_probs(model: torch.nn.Module, points: torch.Tensor, votes: int) -> torch.Tensor:
    """Average softmax probabilities over `votes` forward passes (optional Y-rotation votes)."""
    probs: torch.Tensor | None = None
    for vote_idx in range(votes):
        vote_points = points.clone()
        if vote_idx > 0:
            theta = torch.rand(points.shape[0], device=points.device) * 2 * np.pi
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
        batch_probs = F.softmax(model(vote_points), dim=1)
        probs = batch_probs if probs is None else probs + batch_probs
    assert probs is not None
    return probs / votes


def load_classifier_checkpoint(
    checkpoint: str | Path,
    *,
    variant: str | None = None,
    width: int | None = None,
    nsample: int | None = None,
    use_normals: bool | None = None,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    settings = {
        "variant": variant or train_args.get("variant", "s"),
        "width": int(width or train_args.get("width", 64)),
        "nsample": int(nsample or train_args.get("nsample", 32)),
        "use_normals": bool(train_args.get("use_normals", True)) if use_normals is None else use_normals,
    }
    model = build_model(
        settings["variant"],
        num_classes=num_classes,
        use_normals=settings["use_normals"],
        width=settings["width"],
        nsample=settings["nsample"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, settings
