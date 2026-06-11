from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return torch.cdist(src, dst, p=2) ** 2


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    b = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(b, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(b, n, device=xyz.device) * 1e10
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(b, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(-1)[1]
    return centroids


def knn_group(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int) -> torch.Tensor:
    dists = square_distance(new_xyz, xyz)
    k = min(k, xyz.shape[1])
    return dists.topk(k=k, dim=-1, largest=False, sorted=False)[1]


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ConvBNAct1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointMLPResidual(nn.Module):
    def __init__(self, channels: int, expansion: float = 1.0) -> None:
        super().__init__()
        hidden_channels = max(1, int(channels * expansion))
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class PointMLPLocalGrouper(nn.Module):
    """PointMLP local grouping with geometric affine normalization."""

    def __init__(self, channels: int, groups: int, kneighbors: int) -> None:
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = min(self.groups, xyz.shape[1])
        fps_idx = farthest_point_sample(xyz, groups)
        new_xyz = index_points(xyz, fps_idx)
        point_features = features.transpose(1, 2).contiguous()
        anchor_features = index_points(point_features, fps_idx)
        group_idx = knn_group(xyz, new_xyz, self.kneighbors)
        grouped_features = index_points(point_features, group_idx)

        anchor = anchor_features.unsqueeze(2)
        centered = grouped_features - anchor
        std = centered.reshape(centered.shape[0], -1).std(dim=-1, unbiased=False)
        std = std.view(-1, 1, 1, 1)
        normalized = centered / (std + 1e-5)
        normalized = normalized * self.affine_alpha + self.affine_beta
        local_features = torch.cat([normalized, anchor.expand_as(normalized)], dim=-1)
        return new_xyz, local_features


class PointMLPPreExtraction(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int, expansion: float) -> None:
        super().__init__()
        self.transfer = ConvBNAct1D(in_channels * 2, out_channels)
        self.blocks = nn.Sequential(
            *[PointMLPResidual(out_channels, expansion=expansion) for _ in range(blocks)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, groups, neighbors, channels = features.shape
        features = features.permute(0, 1, 3, 2).reshape(
            batch_size * groups, channels, neighbors
        )
        features = self.blocks(self.transfer(features))
        features = F.adaptive_max_pool1d(features, 1).reshape(batch_size, groups, -1)
        return features.transpose(1, 2).contiguous()


class PointMLPPostExtraction(nn.Module):
    def __init__(self, channels: int, blocks: int, expansion: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[PointMLPResidual(channels, expansion=expansion) for _ in range(blocks)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.blocks(features)


class PointMLPClassifier(nn.Module):
    """PointMLP classifier adapted to optional xyz+normal input channels."""

    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 6,
        num_points: int = 1024,
        width: int = 64,
        nsample: int = 24,
        elite: bool = False,
    ) -> None:
        super().__init__()
        if elite:
            dim_expansion = [2, 2, 2, 1]
            pre_blocks = [1, 1, 2, 1]
            post_blocks = [1, 1, 2, 1]
            residual_expansion = 0.25
        else:
            dim_expansion = [2, 2, 2, 2]
            pre_blocks = [2, 2, 2, 2]
            post_blocks = [2, 2, 2, 2]
            residual_expansion = 1.0

        self.architecture = "pointmlp_elite" if elite else "pointmlp"
        self.embedding = ConvBNAct1D(in_channels, width)
        self.groupers = nn.ModuleList()
        self.pre_extractors = nn.ModuleList()
        self.post_extractors = nn.ModuleList()

        channels = width
        groups = num_points
        for stage_idx, multiplier in enumerate(dim_expansion):
            groups = max(1, groups // 2)
            out_channels = channels * multiplier
            self.groupers.append(PointMLPLocalGrouper(channels, groups, kneighbors=nsample))
            self.pre_extractors.append(
                PointMLPPreExtraction(
                    channels,
                    out_channels,
                    blocks=pre_blocks[stage_idx],
                    expansion=residual_expansion,
                )
            )
            self.post_extractors.append(
                PointMLPPostExtraction(
                    out_channels,
                    blocks=post_blocks[stage_idx],
                    expansion=residual_expansion,
                )
            )
            channels = out_channels

        self.classifier = nn.Sequential(
            nn.Linear(channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points[:, :, :3].contiguous()
        features = self.embedding(points.transpose(1, 2).contiguous())
        for grouper, pre_extract, post_extract in zip(
            self.groupers, self.pre_extractors, self.post_extractors
        ):
            xyz, grouped_features = grouper(xyz, features)
            features = post_extract(pre_extract(grouped_features))
        global_features = F.adaptive_max_pool1d(features, 1).squeeze(-1)
        return self.classifier(global_features)


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, k: int, in_channels: int, out_channels: int, blocks: int) -> None:
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.local = nn.Sequential(
            ConvBNAct(in_channels + 3, out_channels),
            ConvBNAct(out_channels, out_channels),
        )
        self.res_blocks = nn.Sequential(*[ResidualMLP(out_channels) for _ in range(blocks)])

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        npoint = min(self.npoint, xyz.shape[1])
        fps_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        group_idx = knn_group(xyz, new_xyz, self.k)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_features = index_points(features.transpose(1, 2), group_idx)
        relative_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        local_input = torch.cat([relative_xyz, grouped_features], dim=-1)
        local_input = local_input.permute(0, 3, 1, 2).contiguous()
        new_features = self.local(local_input).max(dim=-1)[0]
        new_features = self.res_blocks(new_features)
        return new_xyz, new_features


class PointNeXtClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 6,
        variant: str = "s",
        width: int = 32,
        nsample: int = 32,
    ) -> None:
        super().__init__()
        if variant.lower() == "b":
            widths = [width, width * 2, width * 4, width * 8]
            blocks = [1, 2, 2, 2]
        else:
            widths = [width, width * 2, width * 4, width * 8]
            blocks = [1, 1, 1, 1]

        self.variant = variant.lower()
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            ResidualMLP(widths[0]),
        )
        self.sa1 = SetAbstraction(512, nsample, widths[0], widths[1], blocks[1])
        self.sa2 = SetAbstraction(128, nsample, widths[1], widths[2], blocks[2])
        self.sa3 = SetAbstraction(32, nsample, widths[2], widths[3], blocks[3])
        self.classifier = nn.Sequential(
            nn.Linear(widths[3] * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points[:, :, :3].contiguous()
        features = points.transpose(1, 2).contiguous()
        features = self.stem(features)
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        _, features = self.sa3(xyz, features)
        global_max = F.adaptive_max_pool1d(features, 1).squeeze(-1)
        global_avg = F.adaptive_avg_pool1d(features, 1).squeeze(-1)
        return self.classifier(torch.cat([global_max, global_avg], dim=1))


def build_model(
    variant: str,
    num_classes: int,
    use_normals: bool,
    width: int = 32,
    nsample: int = 32,
    architecture: str = "pointnext_legacy",
    num_points: int = 1024,
) -> nn.Module:
    in_channels = 6 if use_normals else 3
    architecture = architecture.lower()
    if architecture in {"pointmlp", "pointmlp_elite"}:
        return PointMLPClassifier(
            num_classes=num_classes,
            in_channels=in_channels,
            num_points=num_points,
            width=width,
            nsample=nsample,
            elite=architecture == "pointmlp_elite",
        )
    if architecture != "pointnext_legacy":
        raise ValueError(f"unknown architecture: {architecture}")
    return PointNeXtClassifier(
        num_classes=num_classes,
        in_channels=in_channels,
        variant=variant,
        width=width,
        nsample=nsample,
    )
