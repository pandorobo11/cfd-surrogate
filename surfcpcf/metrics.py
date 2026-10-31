"""Shared coefficient metrics and training-range extrapolation flags."""

import warnings

import numpy as np
from scipy.stats import spearmanr

from .geometry import relative_error

EXTRAPOLATION_KEYS = ("reynolds_extrapolation", "aoa_extrapolation", "unseen_geometry_parameters")
GEOMETRY_RULE = "case-name geometry parameter tuple absent from training; not a geometric distance test"
ZERO_POLICY = "Exact zero reference with nonzero error is infinity (CSV); nonfinite aggregate is null (JSON), with count."


def distribution(values):
    values = np.asarray(values)
    mean, median = float(values.mean()), float(np.median(values))
    return {"mean": mean if np.isfinite(mean) else None,
            "median": median if np.isfinite(median) else None,
            "nonfinite_count": int((~np.isfinite(values)).sum())}


def training_ranges(conditions):
    return {key: [min(c[key] for c in conditions), max(c[key] for c in conditions)]
            for key in ("uinf", "aoa_deg", "nu")}


class Extrapolation:
    def __init__(self, train_names, train_meta, training_range):
        self.re_range = [min(m["uinf"] / m["nu"] for m in train_meta),
                         max(m["uinf"] / m["nu"] for m in train_meta)]
        self.aoa_range = training_range["aoa_deg"]
        self.geometries = {tuple(name.split("_")[4:]) for name in train_names}

    def flags(self, name, meta):
        return dict(reynolds_extrapolation=not self.re_range[0] <= meta["uinf"] / meta["nu"] <= self.re_range[1],
                    aoa_extrapolation=not self.aoa_range[0] <= meta["aoa_deg"] <= self.aoa_range[1],
                    unseen_geometry_parameters=tuple(name.split("_")[4:]) not in self.geometries)


def coefficient_row(predicted, reference):
    row = {}
    for key, value in reference.items():
        row.update({f"pred_{key}": float(predicted[key]), f"true_{key}": float(value),
                    f"{key}_relative_error": float(relative_error(predicted[key], value))})
    return row


def coefficient_metrics(rows, coefficients):
    correlations = {}
    for key in ("cl", "cd"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation = spearmanr([r[f"pred_{key}"] for r in rows], [r[f"true_{key}"] for r in rows])
        correlations[key] = {"statistic": float(correlation.statistic) if np.isfinite(correlation.statistic) else None,
                             "pvalue": float(correlation.pvalue) if np.isfinite(correlation.pvalue) else None}
    return {"coefficient_relative_error": {
                key: distribution([r[f"{key}_relative_error"] for r in rows]) for key in coefficients},
            "spearman": correlations,
            "extrapolation_counts": {key: sum(r[key] for r in rows) for key in EXTRAPOLATION_KEYS},
            "geometry_extrapolation_rule": GEOMETRY_RULE,
            "relative_error_zero_policy": ZERO_POLICY}
