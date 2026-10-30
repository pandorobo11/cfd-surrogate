"""Run all five pre-training checks on every preprocessed surface case."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .data import load_case
from .geometry import COEFFICIENTS, coefficients_from_cp_cf, force_coefficients, relative_error
from .model import make_transolver, select_device


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "mean": float(values.mean()),
            "median": float(np.median(values)), "max": float(values.max())}


def integration_args(case):
    return {key: case[key] for key in ("lines", "length", "cell_normal", "aoa_rad", "q")}


def run_checks(surface, dataset, transolver_dir, device=None):
    names = sorted(path.stem for path in Path(surface).glob("*.npz"))
    if not names:
        raise ValueError("No preprocessed cases")
    manifest = json.loads((Path(dataset) / "manifest.json").read_text())
    known = {name for split in manifest.values() for name in split}
    if set(names) - known:
        raise ValueError("Surface files contain cases absent from the source manifest")
    absolute, relative, tangent_abs, tangent_rel = [], [], [], []
    counts, times, ratios = [], [], []
    outward = inward = neutral = upper_positive = upper_count = 0
    per_case = []
    for name in names:
        case = load_case(surface, name)
        meta = json.loads(str(case["metadata"]))
        counts.append(len(case["pos"]))
        times.append(meta["preprocessing_seconds"])
        args = integration_args(case)
        integrated = force_coefficients(case["p"], case["tau"], **args)
        pred = np.array([integrated[key] for key in COEFFICIENTS])
        reference = case["reference"]
        absolute.append(np.abs(pred - reference))
        relative.append(relative_error(pred, reference))
        reconstructed = coefficients_from_cp_cf(case["cp"], case["cf"], case["normal"], **args)
        pred_t = np.array([reconstructed[key] for key in COEFFICIENTS])
        tangent_abs.append(np.abs(pred_t - reference))
        tangent_rel.append(relative_error(pred_t, reference))
        normal_norm, tangent_norm = np.linalg.norm(case["cf_n"]), np.linalg.norm(case["cf"])
        ratio = float(normal_norm / tangent_norm) if tangent_norm else (0. if normal_norm == 0 else float("inf"))
        ratios.append(ratio)
        pos, normal = case["pos"], case["normal"]
        center = pos.mean(axis=0)
        dots = (normal * (pos - center)).sum(axis=1)
        outward += int((dots > 0).sum())
        inward += int((dots < 0).sum())
        neutral += int((dots == 0).sum())
        # Omit the first 5% of chord to avoid the leading-edge stagnation region.
        upper = (pos[:, 1] > center[1]) & (pos[:, 0] > pos[:, 0].min() + 0.05 * np.ptp(pos[:, 0]))
        upper_count += int(upper.sum())
        upper_positive += int((case["cf"][upper] > 0).sum())
        per_case.append({"case": name, "normal_to_tangent_l2": ratio,
                         "upper_positive_cf_fraction": float((case["cf"][upper] > 0).mean()) if upper.any() else None})
    max_relative = np.max(relative, axis=0)
    passed = bool(np.all(np.isfinite(max_relative)) and np.all(max_relative <= 1e-6))
    total = outward + inward + neutral
    direction = "outward" if outward > inward else "inward" if inward > outward else "undetermined"
    fraction = upper_positive / upper_count if upper_count else None
    result = {
        "cases": len(names),
        "check1": {"points": summary(counts), "preprocessing_seconds": summary(times)},
        "check2": {"passed": passed, "relative_tolerance": 1e-6,
                   "full_vector": {key: {"max_absolute": float(np.max(absolute, axis=0)[i]),
                                         "max_relative": float(max_relative[i])} for i, key in enumerate(COEFFICIENTS)},
                   "tangent_only": {key: {"max_absolute": float(np.max(tangent_abs, axis=0)[i]),
                                          "max_relative": float(np.max(tangent_rel, axis=0)[i])} for i, key in enumerate(COEFFICIENTS)}},
        "check3": {"normal_to_tangent_l2": summary(ratios)},
        "check4": {"outward_fraction": outward / total, "inward_fraction": inward / total,
                   "zero_fraction": neutral / total, "direction": direction,
                   "upper_positive_cf_fraction": fraction, "upper_point_count": upper_count,
                   "upper_selection": "y > mean(y), x > min(x) + 0.05 * chord",
                   "interpretation": f"Normals are predominantly {direction}. The +90 degree tangent defines signed Cf; the upper-surface positive fraction is {fraction}. This geometric test alone does not diagnose separation.",
                   "per_case": per_case},
    }
    selected = select_device(device)
    model_check = {"device": str(selected), "mps_available": torch.backends.mps.is_available(),
                   "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                   "torch_version": torch.__version__, "n_points": 1000,
                   "space_dim": 6, "n_hidden": 128, "n_layers": 5, "slice_num": 32,
                   "passed": False, "error": None}
    try:
        torch.manual_seed(0)
        model = make_transolver(transolver_dir).to(selected)
        dummy = SimpleNamespace(x=torch.randn(1000, 6, device=selected),
                                pos=torch.rand(1000, 2, device=selected))
        prediction = model(dummy)
        if prediction.shape != (1000, 2) or not torch.isfinite(prediction).all():
            raise RuntimeError("Invalid model output")
        prediction.square().mean().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("Missing or nonfinite gradients")
        if selected.type == "mps":
            torch.mps.synchronize()
        model_check.update(passed=True, output_shape=list(prediction.shape))
    except Exception as exc:
        model_check["error"] = f"{type(exc).__name__}: {exc}"
    result["check5"] = model_check
    result["passed"] = passed and model_check["passed"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, default=Path("data/airfrans_surface"))
    parser.add_argument("--dataset", type=Path, default=Path("data/airfrans/Dataset"))
    parser.add_argument("--transolver-dir", type=Path, default=Path("external/Transolver/Airfoil-Design-AirfRANS"))
    parser.add_argument("--out", type=Path, default=Path("results/surface/checks.json"))
    parser.add_argument("--device", choices=("mps", "cpu"))
    args = parser.parse_args(argv)
    result = run_checks(args.surface, args.dataset, args.transolver_dir, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "check4"}, indent=2))
    print(json.dumps({"check4": {k: v for k, v in result["check4"].items() if k != "per_case"}}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
