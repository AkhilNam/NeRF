from typing import Tuple

import torch
import torch.nn as nn


class NeRFMLP(nn.Module):
	"""
	Minimal NeRF MLP.

	Inputs:
	- encoded position: (N, C_in)
	Outputs:
	- rgb: (N, 3)
	- density (sigma): (N, 1)

	TODO: choose layer sizes
	TODO: split density and color heads
	TODO: apply appropriate activations
	"""

	def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 4) -> None:
		"""
		Args:
		- input_dim: encoded input dimension
		- hidden_dim: hidden width
		- num_layers: number of trunk layers
		"""
		super().__init__()
		assert num_layers >= 1, "num_layers must be >= 1"

		trunk = []
		prev = input_dim
		for _ in range(num_layers):
			trunk.append(nn.Linear(prev, hidden_dim))
			prev = hidden_dim
		self.trunk = nn.ModuleList(trunk)

		# Density head
		self.sigma_head = nn.Linear(hidden_dim, 1)

		# Color head (small MLP for clarity)
		self.color_head = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, 3),
		)

	def forward(self, x_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Forward pass through the MLP.

		Args:
		- x_encoded: (N, C_in)

		Returns:
		- rgb: (N, 3)
		- sigma: (N, 1)
		"""
		h = x_encoded
		for layer in self.trunk:
			h = layer(h)
			h = torch.relu(h)

		# Density (non-negative). ReLU for simplicity; Softplus is also common.
		sigma = torch.relu(self.sigma_head(h))

		# Color in [0, 1]
		rgb = torch.sigmoid(self.color_head(h))
		return rgb, sigma


