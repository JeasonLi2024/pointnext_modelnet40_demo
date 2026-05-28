from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


POINT_EXTS = {".txt", ".csv", ".npy", ".npz"}


@dataclass(frozen=True)
class PointCloudItem:
    path: Path
    sample_id: str
    label: int | None


def read_point_cloud(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        obj = np.load(path)
        key = "points" if "points" in obj else obj.files[0]
        arr = obj[key]
    else:
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        except ValueError:
            arr = np.loadtxt(path, dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path} should have shape N x 3 or N x 6, got {arr.shape}")
    if arr.shape[1] >= 6:
        arr = arr[:, :6]
    else:
        normals = np.zeros((arr.shape[0], 3), dtype=np.float32)
        arr = np.concatenate([arr[:, :3], normals], axis=1)
    return arr


def has_class_dirs(path: Path, label_to_idx: dict[str, int]) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(p.is_dir() and p.name in label_to_idx for p in path.iterdir())


def normalize_xyz(points: np.ndarray) -> np.ndarray:
    points = points.copy()
    xyz = points[:, :3]
    xyz -= xyz.mean(axis=0, keepdims=True)
    radius = np.sqrt((xyz**2).sum(axis=1)).max()
    if radius > 0:
        xyz /= radius
    points[:, :3] = xyz
    return points


def sample_points(points: np.ndarray, num_points: int, random_sample: bool) -> np.ndarray:
    n = points.shape[0]
    if n == num_points:
        return points
    replace = n < num_points
    if random_sample:
        idx = np.random.choice(n, num_points, replace=replace)
    else:
        if replace:
            extra = np.resize(np.arange(n), num_points - n)
            idx = np.concatenate([np.arange(n), extra])
        else:
            idx = np.linspace(0, n - 1, num_points).astype(np.int64)
    return points[idx]


def rotate_y(points: np.ndarray) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    points = points.copy()
    points[:, :3] = points[:, :3] @ rot.T
    points[:, 3:6] = points[:, 3:6] @ rot.T
    return points


def augment(points: np.ndarray, random_rotate: bool = False) -> np.ndarray:
    if random_rotate:
        points = rotate_y(points)
    else:
        points = points.copy()
    scale = np.random.uniform(0.8, 1.2)
    shift = np.random.uniform(-0.1, 0.1, size=(1, 3)).astype(np.float32)
    jitter = np.clip(0.01 * np.random.randn(*points[:, :3].shape), -0.05, 0.05).astype(np.float32)
    points[:, :3] = points[:, :3] * scale + shift + jitter
    keep_ratio = np.random.uniform(0.875, 1.0)
    keep = max(8, int(points.shape[0] * keep_ratio))
    if keep < points.shape[0]:
        idx = np.random.choice(points.shape[0], keep, replace=False)
        dropped = points[idx]
        fill = dropped[np.random.choice(keep, points.shape[0] - keep, replace=True)]
        points = np.concatenate([dropped, fill], axis=0)
    return points


class ModelNetLikeDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        labels: list[str],
        num_points: int = 1024,
        use_normals: bool = True,
        train: bool = False,
        random_rotate: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.labels = labels
        self.label_to_idx = {name: i for i, name in enumerate(labels)}
        self.num_points = num_points
        self.use_normals = use_normals
        self.train = train
        self.random_rotate = random_rotate
        self.items = self._discover_items()
        if not self.items:
            raise FileNotFoundError(f"No point cloud files found for split '{split}' under {self.data_root}")

    @property
    def num_channels(self) -> int:
        return 6 if self.use_normals else 3

    def _split_dir(self) -> Path:
        split_dir = self.data_root / self.split
        if split_dir.exists():
            return split_dir
        if self.split == "train" and has_class_dirs(self.data_root, self.label_to_idx):
            return self.data_root
        return split_dir

    def _discover_items(self) -> list[PointCloudItem]:
        split_dir = self._split_dir()
        items: list[PointCloudItem] = []

        manifest = self.data_root / f"{self.split}.csv"
        if manifest.exists():
            with manifest.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rel = row.get("path") or row.get("file")
                    label_name = row.get("label") or row.get("class")
                    if not rel:
                        continue
                    path = self.data_root / rel
                    label = self.label_to_idx.get(label_name, None) if label_name else None
                    items.append(PointCloudItem(path, path.stem, label))
            return items

        if not split_dir.exists():
            return items

        if self.split == "train":
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                label = self.label_to_idx.get(class_dir.name)
                if label is None:
                    continue
                for path in sorted(class_dir.rglob("*")):
                    if path.suffix.lower() in POINT_EXTS:
                        items.append(PointCloudItem(path, path.stem, label))
        else:
            for path in sorted(split_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in POINT_EXTS:
                    label = self.label_to_idx.get(path.parent.name)
                    items.append(PointCloudItem(path, path.stem, label))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        points = read_point_cloud(item.path)
        points = normalize_xyz(points)
        points = sample_points(points, self.num_points, random_sample=self.train)
        if self.train:
            points = augment(points, random_rotate=self.random_rotate)
            np.random.shuffle(points)
        if not self.use_normals:
            points = points[:, :3]

        return {
            "points": torch.from_numpy(points.astype(np.float32)),
            "label": -1 if item.label is None else item.label,
            "id": item.sample_id,
        }


def collate_batch(batch: list[dict]) -> dict:
    return {
        "points": torch.stack([x["points"] for x in batch], dim=0),
        "label": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "id": [x["id"] for x in batch],
    }
