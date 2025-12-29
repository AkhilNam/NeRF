from typing import Tuple

import torch

from .data import CameraIntrinsics


def get_rays(
    height: int,
    width: int,
    intrinsics: CameraIntrinsics,
    c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate one ray per pixel for a pinhole camera."""

    device = c2w.device

    # Pixel grid (pixel centers)
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing="xy",
    )
    x = i + 0.5
    y = j + 0.5

    # Camera coordinates
    x_cam = (x - intrinsics.cx) / intrinsics.fx
    y_cam = (y - intrinsics.cy) / intrinsics.fy

    camera_dirs = torch.stack(
        [x_cam, y_cam, torch.ones_like(x_cam)],
        dim=-1,
    ).reshape(-1, 3)

    # Rotate into world space
    directions = camera_dirs @ c2w[:3, :3].T
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # Ray origins
    origins = c2w[:3, 3].expand_as(directions)

    return origins, directions
