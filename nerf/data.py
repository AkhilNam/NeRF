from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import numpy as np
import torch


@dataclass
class CameraIntrinsics:
	"""
	Camera intrinsics parameters.

	Args:
	- fx: focal length in x-direction
	- fy: focal length in y-direction
	- cx: principal point in x-direction
	- cy: principal point in y-direction
	"""
	fx: float
	fy: float
	cx: float
	cy: float	

	


def load_tiny_synthetic_dataset(
	num_images: int = 2, height: int = 64, width: int = 64
) -> Dict[str, object]:
	"""
	Load a tiny synthetic dataset suitable for educational NeRF training.

	Expected to return:
	- 'images': (N, H, W, 3) in [0, 1], float32
	- 'intrinsics': CameraIntrinsics
	- 'c2w_matrices': list of (4, 4) torch.FloatTensor camera-to-world

	Args:
	- num_images: number of views
	- height: image height
	- width: image width

	Returns:
	- dict with keys 'images', 'intrinsics', 'c2w_matrices'

	TODO: verify coordinate conventions (right-handed vs left-handed, forward axis)
	"""
	assert num_images >= 1
	H, W = height, width

	# Simple pinhole intrinsics: square pixels, principal point at image center.
	focal = 0.5 * W  # arbitrary but reasonable for a toy example
	intrinsics = CameraIntrinsics(
		fx=float(focal),
		fy=float(focal),
		cx=float(W / 2.0),
		cy=float(H / 2.0),
	)

	def _checkerboard(h: int, w: int, checks: int = 8) -> np.ndarray:
		"""
		Create an RGB checkerboard in [0, 1], shape (H, W, 3).
		"""
		y = np.linspace(0, 1, h, endpoint=False)
		x = np.linspace(0, 1, w, endpoint=False)
		xx, yy = np.meshgrid(x, y)
		grid = ((np.floor(xx * checks) + np.floor(yy * checks)) % 2).astype(np.float32)
		img = np.stack([grid, 1.0 - grid, 0.5 * np.ones_like(grid)], axis=-1)
		return img.astype(np.float32)

	def _solid(h: int, w: int, color: Tuple[float, float, float]) -> np.ndarray:
		"""
		Create a solid color RGB image in [0, 1], shape (H, W, 3).
		"""
		img = np.ones((h, w, 3), dtype=np.float32)
		img *= np.asarray(color, dtype=np.float32)[None, None, :]
		return img

	# Tiny set of synthetic images for toy training
	images: List[np.ndarray] = []
	base_colors = [
		(1.0, 0.0, 0.0),
		(0.0, 1.0, 0.0),
		(0.0, 0.0, 1.0),
		(1.0, 1.0, 0.0),
	]
	for i in range(num_images):
		if i % 2 == 0:
			images.append(_checkerboard(H, W, checks=8))
		else:
			images.append(_solid(H, W, base_colors[i % len(base_colors)]))
	images_np = np.stack(images, axis=0).astype(np.float32)  # (N, H, W, 3)

	# Camera poses on a small circle around origin, looking at origin
	# Convention (assumed, TODO verify):
	# - World axes: x right, y up, z forward
	# - Camera looks toward origin; forward vector points from cam to origin
	c2w_list: List[torch.Tensor] = []
	radius = 2.0
	for i in range(num_images):
		theta = (2.0 * math.pi * i) / max(1, num_images)
		cam_pos = np.array(
			[radius * math.cos(theta), 0.2 * radius, radius * math.sin(theta)],
			dtype=np.float32,
		)
		forward = (np.zeros(3, dtype=np.float32) - cam_pos)
		forward = forward / (np.linalg.norm(forward) + 1e-8)
		up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		right = np.cross(forward, up)
		right = right / (np.linalg.norm(right) + 1e-8)
		true_up = np.cross(right, forward)
		true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

		R = np.stack([right, true_up, forward], axis=1)  # columns are basis
		t = cam_pos.reshape(3, 1)
		c2w = np.eye(4, dtype=np.float32)
		c2w[:3, :3] = R
		c2w[:3, 3:4] = t
		c2w_list.append(torch.from_numpy(c2w))

	return {
		"images": images_np,
		"intrinsics": intrinsics,
		"c2w_matrices": c2w_list,
	}


