"""Trailing-edge separation on the upper surface: CFD vs a surfcpcf run, by angle of attack.

Upper surface = points whose (inward) normal has negative y. With the +90 degree tangent convention,
attached upper-surface flow has Cf > 0, so separation is Cf < 0 (excluding the first 5% of chord).
Usage: python high_aoa_separation.py <airfrans_surface> <transolver_dir> <run_dir> <out_dir>
"""
import csv, json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from surfcpcf.data import Normalization, SurfaceDataset, load_case, read_splits, to_device
from surfcpcf.model import build_model, select_device
from surfcpcf.checks import integration_args
from surfcpcf.geometry import force_coefficients, coefficients_from_cp_cf

surface, tdir, run, out = sys.argv[1:5]
run, out = Path(run), Path(out); out.mkdir(parents=True, exist_ok=True)
config = json.loads((run / "config.json").read_text())
splits = read_splits(surface, config["task"])
lo, hi = config["training_range"]["aoa_deg"]


def upper_sep(case, cf):
    x = case["pos"][:, 0]; chord = np.ptp(x)
    upper = (case["normal"][:, 1] < 0) & (x > x.min() + 0.05 * chord)
    xs, sep = x[upper], cf[upper] < 0
    frac = float(sep.mean())
    onset = float((xs[sep].min() - x.min()) / chord) if sep.any() else None
    return frac, onset


rows = []
for split in ("train", "test"):
    for name in splits[split]:
        case = load_case(surface, name); meta = json.loads(str(case["metadata"]))
        frac, onset = upper_sep(case, case["cf"])
        rows.append(dict(split=split, case=name, aoa=meta["aoa_deg"], sep_frac_true=frac, onset_true=onset))
by_name = {r["case"]: r for r in rows}

norm = Normalization.load(run / "norm.npz"); names = splits["test"]
ds = SurfaceDataset(surface, names, norm); dev = select_device(None)
model = build_model(config.get("model", "transolver"), tdir, config["n_hidden"], config["n_layers"], config["slice_num"]).to(dev)
model.load_state_dict(torch.load(run / "model.pt", map_location=dev, weights_only=True)); model.eval()
preds = {}
with torch.no_grad():
    for name, (d, _) in zip(names, ds):
        case = load_case(surface, name)
        pr = norm.inverse_targets(model(to_device(d, dev)).cpu().numpy()); preds[name] = pr
        r = by_name[name]
        r["sep_frac_pred"], r["onset_pred"] = upper_sep(case, pr[:, 1])
        ref = dict(zip(("cd", "cdp", "cdv", "cl", "clp", "clv"), case["reference"]))
        c = coefficients_from_cp_cf(pr[:, 0], pr[:, 1], case["normal"], **integration_args(case))
        r.update(cd_true=ref["cd"], cdp_true=ref["cdp"], cdv_true=ref["cdv"], cd_pred=c["cd"], cdp_pred=c["cdp"], cdv_pred=c["cdv"])

with (out / "separation_per_case.csv").open("w", newline="") as f:
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def summ(rs, label):
    if not rs: return
    sep_t = np.array([r["sep_frac_true"] for r in rs]); line = f"{label}: n={len(rs)} cases with CFD upper separation (>1% of upper points) {np.mean(sep_t > .01):.2f}, median separated fraction {np.median(sep_t):.3f}"
    if "sep_frac_pred" in rs[0]:
        sep_p = np.array([r["sep_frac_pred"] for r in rs])
        line += f" | predicted: cases {np.mean(sep_p > .01):.2f}, median fraction {np.median(sep_p):.3f}"
        for k in ("cd", "cdp", "cdv"):
            e = np.array([r[k + "_pred"] - r[k + "_true"] for r in rs]); line += f" | {k} err median {np.median(e):+.5f} (|.| {np.median(np.abs(e)):.5f})"
    print(line)

train = [r for r in rows if r["split"] == "train"]; test = [r for r in rows if r["split"] == "test"]
summ([r for r in train if r["aoa"] >= 10], "train aoa>=10")
summ([r for r in train if r["aoa"] >= hi - 1], f"train aoa>={hi-1:.2f}")
summ([r for r in test if r["aoa"] > hi], "test aoa>hi")
summ([r for r in test if r["aoa"] < lo], "test aoa<lo")

# figure: four high-AoA test cases with the largest |Cd error|
high = sorted([r for r in test if r["aoa"] > hi], key=lambda r: -abs(r["cd_pred"] - r["cd_true"]))[:4]
fig, axes = plt.subplots(len(high), 2, figsize=(13, 3.2 * len(high)), constrained_layout=True)
for row, r in enumerate(high):
    case = load_case(surface, r["case"]); pr = preds[r["case"]]; x = case["pos"][:, 0]
    up = case["normal"][:, 1] < 0
    for col, (j, field) in enumerate(((0, "cp"), (1, "cf"))):
        ax = axes[row, col]
        for mask, side, color in ((up, "upper", "tab:blue"), (~up, "lower", "tab:orange")):
            ids = np.flatnonzero(mask); ids = ids[np.argsort(x[ids])]
            ax.plot(x[ids], case[field][ids], color=color, lw=1.4, label=f"CFD {side}")
            ax.plot(x[ids], pr[ids, j], "--", color=color, lw=1.2, label=f"prediction {side}")
        if field == "cp": ax.invert_yaxis()
        else: ax.axhline(0, color="k", lw=0.6)
        ax.set(xlabel="x / chord", ylabel="Cp" if field == "cp" else "Cf"); ax.grid(alpha=.25); ax.legend(fontsize=7)
        ax.set_title(f"AoA {r['aoa']:.2f} deg (training max {hi:.2f}); Cd CFD {r['cd_true']:.4f}, pred {r['cd_pred']:.4f}\n{r['case']}", fontsize=8)
fig.suptitle("SURF-04 angle-of-attack extrapolation, largest Cd errors (surrogate, not validated flow solutions)", fontsize=10)
fig.savefig(out / "high_aoa_cases.png", dpi=130); plt.close(fig)
print("figure", out / "high_aoa_cases.png")
