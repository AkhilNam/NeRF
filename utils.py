import imageio

import numpy as np
import torch

global_seed = 42

def get_device(force_cpu: bool = True) -> torch.device:

	"""
	Select device (placeholder).

	Args:
	- force_cpu: if True, always select CPU

	Returns:
	- device 

	TODO: device handling
	"""
	if force_cpu:
		device = torch.device("cpu")
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	return device


def set_seed(seed: int = 42) -> None:
	"""
	Set random seeds (placeholder).

	Args:
	- seed: integer seed

	Returns:
	- None
	"""
	global global_seed
	global_seed = seed
	torch.manual_seed(seed)
	np.random.seed(seed)


def to_numpy_image(x: torch.Tensor) -> np.ndarray:
	"""
	Convert tensor in [0, 1] to uint8 HxWx3 (placeholder).

	Args:
	- x: (H, W, 3) or (3, H, W)

	Returns:
	- img: (H, W, 3) uint8

	TODO: simple visualization helpers
	"""
	if x.dim() == 3 and x.shape[0] == 3:
		x = x.permute(1, 2, 0)

	x = x.detach().cpu().clamp(0, 1)
	numpy_image = (255 * x).numpy().astype(np.uint8)
	
	return numpy_image

def save_image(path: str, img: np.ndarray) -> None:
	"""
	Save image to disk (placeholder).

	Args:
	- path: output path
	- img: (H, W, 3) uint8

	Returns:
	- None
	"""

	imageio.imwrite(path, img.astype(np.uint8))




