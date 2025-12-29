from typing import Dict, Tuple

import torch


def sample_points_along_rays(
	origins: torch.Tensor,
	directions: torch.Tensor,
	near: float,
	far: float,
	num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Uniformly sample points along each ray between near and far bounds.

	Args:
	- origins: (N_rays, 3)
	- directions: (N_rays, 3)
	- near: near bound
	- far: far bound
	- num_samples: samples per ray

	Returns:
	- points: (N_rays, S, 3)
	- t_vals: (N_rays, S)
	- deltas: (N_rays, S)

	Notes:
	- Uses evenly spaced samples (no stratified jitter).
	- directions are assumed to be normalized.
	- deltas use forward differences; the last delta is repeated for shape match.

	TODO: uniform sampling along rays (done)
	TODO: add stratified jitter or hierarchical sampling (optional)
	"""
	assert far > near, "far must be greater than near"
	assert num_samples >= 1, "num_samples must be >= 1"
	N = origins.shape[0]
	assert directions.shape == (N, 3), "directions must be (N_rays, 3)"

	device = origins.device
	dtype = origins.dtype

	# Shared t samples across all rays
	t = torch.linspace(near, far, steps=num_samples, device=device, dtype=dtype)  # (S,)
	t_vals = t.view(1, -1).expand(N, num_samples).contiguous()  # (N, S)

	# Points p(t) = o + t * d
	points = origins[:, None, :] + t_vals[:, :, None] * directions[:, None, :]  # (N, S, 3)

	# Spacing between adjacent samples; repeat last delta
	if num_samples > 1:
		deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (N, S-1)
		last_delta = deltas[:, -1:]
		deltas = torch.cat([deltas, last_delta], dim=1)  # (N, S)
	else:
		# Single sample: use a coarse delta equal to the full interval
		deltas = torch.full((N, 1), fill_value=far - near, device=device, dtype=dtype)

	return points, t_vals, deltas
def volume_render(
	rgb: torch.Tensor, sigma: torch.Tensor, deltas: torch.Tensor, eps: float = 1e-10
) -> Dict[str, torch.Tensor]:
	"""
	Volume rendering via alpha compositing.

	Args:
	- rgb: (N_rays, S, 3)
	- sigma: (N_rays, S, 1)
	- deltas: (N_rays, S)
	- eps: small constant for stability

	Returns:
	- dict with 'rgb', 'weights', 'alpha', 'transmittance'

	Details:
	- alpha_i = 1 - exp(-sigma_i * delta_i)
	- T_i = prod_{j < i} (1 - alpha_j)
	- weight_i = alpha_i * T_i
	- C = sum_i weight_i * rgb_i

	TODO: convert density to alpha (done)
	TODO: alpha compositing (done)
	TODO: check numerical stability (use eps clamps)
	"""
	N, S, _ = rgb.shape
	assert sigma.shape == (N, S, 1), "sigma must be (N, S, 1)"
	assert deltas.shape == (N, S), "deltas must be (N, S)"

	# Convert density to opacity
	alpha = 1.0 - torch.exp(-torch.clamp(sigma.squeeze(-1), min=0.0) * deltas)  # (N, S)
	alpha = torch.clamp(alpha, 0.0, 1.0)

	# Cumulative transmittance (prepend 1.0, exclude current alpha)
	one_minus_alpha = (1.0 - alpha).clamp(min=eps)
	T = torch.cumprod(
		torch.cat([torch.ones((N, 1), device=rgb.device, dtype=rgb.dtype), one_minus_alpha[:, :-1]], dim=1),
		dim=1,
	)  # (N, S)

	# Weights and composite color
	weights = alpha * T  # (N, S)
	comp_rgb = torch.sum(weights[:, :, None] * rgb, dim=1)  # (N, 3)

	return {
		"rgb": comp_rgb,
		"weights": weights,
		"alpha": alpha,
		"transmittance": T,
	}


