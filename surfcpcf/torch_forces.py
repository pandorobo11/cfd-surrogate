"""Differentiable counterpart of geometry.coefficients_from_cp_cf (unit chord)."""

import torch

from .geometry import COEFFICIENTS


def coefficients_from_cp_cf(cp, cf, normal, lines, length, cell_normal, aoa_rad):
    """Integrate signed tangential shear and pressure; dynamic pressure cancels."""
    tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
    tau = cf[:, None] * tangent
    fp = (cp[lines].mean(dim=1)[:, None] * cell_normal * length[:, None]).sum(dim=0)
    fv = (tau[lines].mean(dim=1) * length[:, None]).sum(dim=0)
    angle = torch.as_tensor(aoa_rad, dtype=cp.dtype, device=cp.device)
    c, s = angle.cos(), angle.sin()
    basis = torch.stack((torch.stack((c, s)), torch.stack((-s, c))))
    pressure, viscous = basis @ fp, basis @ fv
    return dict(zip(COEFFICIENTS, (
        pressure[0] + viscous[0], pressure[0], viscous[0],
        pressure[1] + viscous[1], pressure[1], viscous[1],
    )))
