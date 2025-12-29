import torch


def positional_encoding(x: torch.Tensor, num_freqs: int, include_input: bool = True) -> torch.Tensor:
	"""
	Apply NeRF-style positional encoding to input coordinates.

	Args:
	- x: (..., C) input tensor
	- num_freqs: number of frequency bands
	- include_input: include raw x in output

	Returns:
	- encoded tensor (..., C * (1 + 2 * num_freqs)) if include_input else (..., C * 2 * num_freqs)

	Details:
	- Uses frequencies 2^k for k in [0, num_freqs-1]
	- For each frequency f and channel, append sin(pi * f * x) and cos(pi * f * x)

	TODO: verify whether to include the pi factor (some implementations omit it)
	TODO: verify output dimensionality for your chosen include_input setting
	TODO: allow turning encoding off (num_freqs=0) (done)
	"""
	if num_freqs < 0:
		raise ValueError("num_freqs must be >= 0")

	# Edge case: no encoding
	if num_freqs == 0:
		return x if include_input else x.new_zeros((*x.shape[:-1], 0))

	device = x.device
	dtype = x.dtype

	# Frequencies: 2^k, k = 0..num_freqs-1
	freqs = (2.0 ** torch.arange(num_freqs, device=device, dtype=dtype))  # (F,)

	# Broadcast x over frequencies: (..., 1, C)
	x_expanded = x.unsqueeze(-2)
	# Angles: (..., F, C)
	angles = x_expanded * freqs.view(-1, 1) * torch.pi

	sin_components = torch.sin(angles)  # (..., F, C)
	cos_components = torch.cos(angles)  # (..., F, C)

	# Concatenate along frequency axis then flatten that axis
	encoded = torch.cat([sin_components, cos_components], dim=-2)  # (..., 2F, C)
	encoded = encoded.reshape(*x.shape[:-1], -1)  # (..., 2F*C)

	if include_input:
		return torch.cat([x, encoded], dim=-1)
	return encoded


