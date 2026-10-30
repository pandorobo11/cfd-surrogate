"""Load the official model at runtime; upstream source stays outside this package."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn


def select_device(requested=None):
    if requested is not None:
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_transolver(transolver_dir, n_hidden=128, n_layers=5, slice_num=32):
    if n_hidden < 8 or n_hidden % 8 or n_layers < 1 or slice_num < 1:
        raise ValueError("Hidden width must be a positive multiple of eight; layers/slices positive")
    path = Path(transolver_dir) / "models" / "Transolver.py"
    spec = importlib.util.spec_from_file_location("_surface_official_transolver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class SurfaceTransolver(module.Transolver):
        def get_grid(self, my_pos):
            # Same numpy linspace -> fp32 construction and reference extents as upstream.
            batch = my_pos.shape[0]
            gx = torch.tensor(np.linspace(-2, 4, self.ref), dtype=my_pos.dtype, device=my_pos.device)
            gy = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=my_pos.dtype, device=my_pos.device)
            gx = gx.reshape(1, self.ref, 1, 1).repeat(batch, 1, self.ref, 1)
            gy = gy.reshape(1, 1, self.ref, 1).repeat(batch, self.ref, 1, 1)
            grid = torch.cat((gx, gy), dim=-1).reshape(batch, self.ref ** 2, 2)
            return ((my_pos[:, :, None, :] - grid[:, None, :, :]) ** 2).sum(-1).sqrt().contiguous()

    return SurfaceTransolver(space_dim=6, fun_dim=0, out_dim=2, n_head=8,
                             mlp_ratio=2, unified_pos=1, n_hidden=n_hidden,
                             n_layers=n_layers, slice_num=slice_num)


MODELS = ("transolver", "mlp")


def build_model(name, transolver_dir, n_hidden, n_layers, slice_num):
    """Model named in a run config; slice_num is ignored by the point MLP."""
    if name == "transolver":
        return make_transolver(transolver_dir, n_hidden, n_layers, slice_num)
    if name == "mlp":
        return PointMLP(n_hidden, n_layers)
    raise ValueError(f"Unknown model {name!r}; expected one of {MODELS}")


class PointMLP(nn.Module):
    """Independent point predictions (SURF-05 to 08): no exchange between points."""
    def __init__(self, n_hidden=128, n_layers=5):
        super().__init__()
        if n_hidden < 1 or n_layers < 1:
            raise ValueError("Hidden width and hidden layer count must be positive")
        layers = []
        for i in range(n_layers):
            layers.extend((nn.Linear(6 if i == 0 else n_hidden, n_hidden), nn.GELU()))
        self.network = nn.Sequential(*layers, nn.Linear(n_hidden, 2))

    def forward(self, data):
        return self.network(data.x)
