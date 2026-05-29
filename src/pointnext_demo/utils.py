from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_labels(labels_path: str | Path) -> list[str]:
    path = Path(labels_path)
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_config(config_path: str | Path | None) -> dict:
    if config_path is None:
        return {}
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} should contain a YAML mapping.")
    return data


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
