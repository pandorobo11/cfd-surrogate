"""Evaluate the official test split using surface field and integrated force errors."""

import argparse
import csv
import json
import os
from pathlib import Path
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

from .checks import integration_args
from .data import Normalization, SurfaceDataset, load_case, read_splits, to_device
from .geometry import COEFFICIENTS, coefficients_from_cp_cf, relative_error
from .model import build_model, select_device
from .train import synchronize


def relative_l2(pred, true):
    norm = np.linalg.norm(true)
    error = np.linalg.norm(pred - true)
    return float(error / norm) if norm else (0. if error == 0 else float("inf"))


def distribution(values):
    values = np.asarray(values)
    mean, median = float(values.mean()), float(np.median(values))
    return {"mean": mean if np.isfinite(mean) else None,
            "median": median if np.isfinite(median) else None,
            "nonfinite_count": int((~np.isfinite(values)).sum())}


def plot_cases(path, representatives, surface, predictions):
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for row, (label, result) in enumerate(representatives):
        name = result["case"]
        case = load_case(surface, name)
        x = case["pos"][:, 0]
        normal = case["normal"]
        outward = np.mean(np.sum(normal * (case["pos"] - case["pos"].mean(0)), axis=1)) > 0
        # A centroid-y cut mixes the two surfaces on cambered airfoils.
        upper = normal[:, 1] > 0 if outward else normal[:, 1] < 0
        for col, field in enumerate(("cp", "cf")):
            ax = axes[row, col]
            for mask, side, color in ((upper, "upper", "tab:blue"), (~upper, "lower", "tab:orange")):
                ids = np.flatnonzero(mask)
                ids = ids[np.argsort(x[ids])]
                ax.plot(x[ids], case[field][ids], color=color, label=f"CFD {side}", linewidth=1.4)
                ax.plot(x[ids], predictions[name][ids, col], "--", color=color, label=f"prediction {side}", linewidth=1.2)
            ax.set(xlabel="x / chord", ylabel=field.capitalize() if field == "cp" else "Cf")
            ax.set_title(f"{label}; Cp relative L2={result['cp_relative_l2']:.3g}\n{name}", fontsize=8)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            if field == "cp":
                ax.invert_yaxis()
    fig.suptitle("Surface only Cp/Cf surrogate: official test cases (not validated flow solutions)", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/surface/SURF-01"))
    parser.add_argument("--surface", type=Path, default=Path("data/airfrans_surface"))
    parser.add_argument("--transolver-dir", type=Path, default=Path("external/Transolver/Airfoil-Design-AirfRANS"))
    parser.add_argument("--device", choices=("mps", "cpu"))
    args = parser.parse_args(argv)
    config = json.loads((args.run / "config.json").read_text())
    splits = read_splits(args.surface, config["task"])
    if splits != config["splits"]:
        raise ValueError("Manifest splits differ from the saved training run")
    names = splits["test"]
    if not names:
        raise ValueError("Empty test split")
    norm = Normalization.load(args.run / "norm.npz")
    dataset = SurfaceDataset(args.surface, names, norm)
    device = select_device(args.device)
    # Runs saved before --model existed are Transolver runs.
    model = build_model(config.get("model", "transolver"), args.transolver_dir,
                        config["n_hidden"], config["n_layers"], config["slice_num"]).to(device)
    model.load_state_dict(torch.load(args.run / "model.pt", map_location=device, weights_only=True))
    model.eval()
    synchronize(device)
    start = time.perf_counter()
    rows, predictions = [], {}
    train_meta = [json.loads(str(load_case(args.surface, name)["metadata"])) for name in splits["train"]]
    re_range = [min(m["uinf"] / m["nu"] for m in train_meta), max(m["uinf"] / m["nu"] for m in train_meta)]
    train_geometries = {tuple(name.split("_")[4:]) for name in splits["train"]}
    with torch.no_grad():
        for name, (data, _) in zip(names, dataset):
            case = load_case(args.surface, name)
            prediction = norm.inverse_targets(model(to_device(data, device)).cpu().numpy())
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Nonfinite prediction: {name}")
            predictions[name] = prediction
            coefficients = coefficients_from_cp_cf(prediction[:, 0], prediction[:, 1], case["normal"], **integration_args(case))
            meta = json.loads(str(case["metadata"]))
            row = {"case": name, "cp_relative_l2": relative_l2(prediction[:, 0], case["cp"]),
                   "cf_relative_l2": relative_l2(prediction[:, 1], case["cf"]),
                   "reynolds_extrapolation": not re_range[0] <= meta["uinf"] / meta["nu"] <= re_range[1],
                   "aoa_extrapolation": not config["training_range"]["aoa_deg"][0] <= meta["aoa_deg"] <= config["training_range"]["aoa_deg"][1],
                   "unseen_geometry_parameters": tuple(name.split("_")[4:]) not in train_geometries}
            for key, reference in zip(COEFFICIENTS, case["reference"]):
                row.update({f"pred_{key}": coefficients[key], f"true_{key}": float(reference),
                            f"{key}_relative_error": float(relative_error(coefficients[key], reference))})
            rows.append(row)
    synchronize(device)
    elapsed = time.perf_counter() - start
    out = args.run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    correlations = {}
    for key in ("cl", "cd"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation = spearmanr([r[f"pred_{key}"] for r in rows], [r[f"true_{key}"] for r in rows])
        correlations[key] = {"statistic": float(correlation.statistic) if np.isfinite(correlation.statistic) else None,
                             "pvalue": float(correlation.pvalue) if np.isfinite(correlation.pvalue) else None}
    ordered = sorted(rows, key=lambda r: r["cp_relative_l2"])
    representatives = list(zip(("minimum", "median (upper middle)", "maximum"),
                               (ordered[0], ordered[len(ordered) // 2], ordered[-1])))
    metrics = {"task": config["task"], "official_test_split": "full_test" if config["task"] == "scarce" else config["task"] + "_test",
               "cases": len(rows), "epochs": config["epochs"], "model": dict(name=config.get("model", "transolver"), **{k: config[k] for k in ("n_hidden", "n_layers", "slice_num")}),
               "setting": config["setting"], "point_subsampling": False,
               "device": str(device), "torch_version": torch.__version__,
               "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
               "inference_and_integration_seconds": elapsed,
               "training_range": dict(config["training_range"], reynolds_unit_chord=re_range),
               "geometry_extrapolation_rule": "case-name geometry parameter tuple absent from training; not a geometric distance test",
               "extrapolation_counts": {key: sum(r[key] for r in rows) for key in ("reynolds_extrapolation", "aoa_extrapolation", "unseen_geometry_parameters")},
               "field_relative_l2": {field: distribution([r[f"{field}_relative_l2"] for r in rows]) for field in ("cp", "cf")},
               "coefficient_relative_error": {key: distribution([r[f"{key}_relative_error"] for r in rows]) for key in COEFFICIENTS},
               "spearman": correlations, "representatives": {label: r["case"] for label, r in representatives},
               "relative_error_zero_policy": "Exact zero reference with nonzero error is infinity (CSV); nonfinite aggregate is null (JSON), with count."}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    plot_cases(out / "cases.png", representatives, args.surface, predictions)
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
