"""Numpy-only, density-divided surface force integration.

Normals are used exactly as supplied; they are neither flipped nor normalized.
AirfRANS force() uses point-averaged cell normals, which can differ from the
original VTP cell normals. Callers must select the intended convention.
"""

import numpy as np

COEFFICIENTS = ("cd", "cdp", "cdv", "cl", "clp", "clv")


def tangent(normal):
    normal = np.asarray(normal, dtype=np.float64)
    return np.stack((-normal[..., 1], normal[..., 0]), axis=-1)


def reconstruct(cp, cf, normal, q, cf_n=None):
    """Return p and tau, retaining the signed +90-degree tangent convention."""
    normal = np.asarray(normal, dtype=np.float64)
    tau = np.asarray(cf).reshape(-1, 1) * tangent(normal)
    if cf_n is not None:
        tau = tau + np.asarray(cf_n).reshape(-1, 1) * normal
    return np.asarray(cp).reshape(-1) * q, tau * q


def force_coefficients(p, tau, lines, length, cell_normal, aoa_rad, q):
    """Average endpoints, integrate p*n and tau, then rotate into drag/lift."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    tau = np.asarray(tau, dtype=np.float64)
    lines = np.asarray(lines, dtype=np.int64)
    length = np.asarray(length, dtype=np.float64).reshape(-1)
    cell_normal = np.asarray(cell_normal, dtype=np.float64)
    if q <= 0 or not np.isfinite(q):
        raise ValueError("q must be finite and positive")
    if lines.shape != (len(length), 2) or cell_normal.shape != (len(length), 2):
        raise ValueError("Expected two endpoints and a 2D normal per line")
    if tau.shape != (len(p), 2):
        raise ValueError("Expected a 2D shear vector per pressure point")
    fp = (p[lines].mean(axis=1)[:, None] * cell_normal * length[:, None]).sum(axis=0)
    fv = (tau[lines].mean(axis=1) * length[:, None]).sum(axis=0)
    c, s = np.cos(aoa_rad), np.sin(aoa_rad)
    basis = np.array([[c, s], [-s, c]])
    cp, cv = basis @ fp / q, basis @ fv / q
    return dict(zip(COEFFICIENTS, map(float, (
        cp[0] + cv[0], cp[0], cv[0], cp[1] + cv[1], cp[1], cv[1]
    ))))


def coefficients_from_cp_cf(cp, cf, normal, lines, length, cell_normal,
                            aoa_rad, q, cf_n=None):
    p, tau = reconstruct(cp, cf, normal, q, cf_n)
    return force_coefficients(p, tau, lines, length, cell_normal, aoa_rad, q)


def relative_error(pred, true):
    """Absolute relative error, with exact zero handled without an epsilon floor."""
    error, scale = np.abs(np.asarray(pred) - true), np.abs(true)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scale != 0, error / scale, np.where(error == 0, 0., np.inf))
