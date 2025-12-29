from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import load_tiny_synthetic_dataset, CameraIntrinsics
from .model import NeRFMLP
from .rays import get_rays
from .render import sample_points_along_rays, volume_render
from .utils import get_device, set_seed, save_image, to_numpy_image

def train_minimal_nerf() -> None:
	"""
	Train a minimal NeRF on a tiny synthetic dataset (CPU-friendly).

	Steps:
	1. Sample random rays
	2. Run NeRF forward
	3. Volume render
	4. Compute MSE loss
	5. Backprop

	Design choices:
	- No positional encoding to keep it minimal (input_dim=3).
	- Overfit a single view for toy correctness.
	- CPU by default; adjust get_device if desired.

	TODO: add simple logging (PSNR helper)
	TODO: render a separate validation view
	"""
	set_seed(42)
	device = get_device(force_cpu=True)

	# Tiny data
	data = load_tiny_synthetic_dataset(num_images=2, height=64, width=64)
	images_np = data["images"]
	intrinsics: CameraIntrinsics = data["intrinsics"]
	c2w_list = data["c2w_matrices"]

	# Use first view for overfitting
	c2w = c2w_list[0].to(device)
	H, W = images_np.shape[1], images_np.shape[2]
	img0 = torch.from_numpy(images_np[0]).to(device=device, dtype=torch.float32)  # (H, W, 3)
	gt_rgb = img0.view(-1, 3).contiguous()  # (H*W, 3)

	# Rays for full image
	rays_o, rays_d = get_rays(H, W, intrinsics, c2w)  # (HW, 3), (HW, 3)
	num_rays = rays_o.shape[0]

	# Model: input xyz only (no encoding), so input_dim=3
	model = NeRFMLP(input_dim=3, hidden_dim=128, num_layers=4).to(device)
	optimizer = optim.Adam(model.parameters(), lr=5e-4)
	criterion = nn.MSELoss(reduction="mean")

	# Hyperparameters
	iters = 400
	batch_size = 1024
	num_samples = 32
	near, far = 1.0, 4.0

	for it in range(1, iters + 1):
		model.train()
		indices = torch.randint(0, num_rays, size=(batch_size,), device=device)
		o_batch = rays_o[indices]
		d_batch = rays_d[indices]
		gt_batch = gt_rgb[indices]

		points, _, deltas = sample_points_along_rays(o_batch, d_batch, near, far, num_samples)
		B, S, _ = points.shape

		# Flatten points, forward MLP
		points_flat = points.reshape(-1, 3)
		rgb_flat, sigma_flat = model(points_flat)
		rgb = rgb_flat.view(B, S, 3)
		sigma = sigma_flat.view(B, S, 1)

		# Volume render
		out = volume_render(rgb, sigma, deltas)
		rgb_comp = out["rgb"]

		loss = criterion(rgb_comp, gt_batch)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		if it % 25 == 0 or it == 1:
			mse = float(loss.item())
			psnr = -10.0 * np.log10(mse + 1e-10)
			print(f"[iter {it:04d}] loss={mse:.6f} psnr={psnr:.2f} dB")

	# Render full training view and save
	with torch.no_grad():
		model.eval()
		points, _, deltas = sample_points_along_rays(rays_o, rays_d, near, far, num_samples)
		N, S, _ = points.shape
		points_flat = points.reshape(-1, 3)
		rgb_flat, sigma_flat = model(points_flat)
		rgb = rgb_flat.view(N, S, 3)
		sigma = sigma_flat.view(N, S, 1)
		out = volume_render(rgb, sigma, deltas)
		img = out["rgb"].view(H, W, 3)
		save_image("nerf_train_output.png", to_numpy_image(img))
		print("Saved training view render to 'nerf_train_output.png'")


if __name__ == "__main__":
	train_minimal_nerf()


