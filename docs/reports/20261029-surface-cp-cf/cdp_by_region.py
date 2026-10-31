"""Pressure-drag contribution by chordwise region (leading edge vs rest), CFD vs a surfcpcf run.

Usage: python cdp_by_region.py <airfrans_surface> <transolver_dir> <run_dir> <aoa_min>
Only test cases with angle of attack above <aoa_min> are used.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, ".")
from surfcpcf.data import Normalization, SurfaceDataset, load_case, read_splits, to_device
from surfcpcf.model import build_model, select_device

surface, tdir, run, aoa_min = sys.argv[1], sys.argv[2], Path(sys.argv[3]), float(sys.argv[4])
c = json.loads((run / "config.json").read_text())
names = [n for n in read_splits(surface, c["task"])["test"]
         if json.loads(str(load_case(surface, n)["metadata"]))["aoa_deg"] > aoa_min]
norm = Normalization.load(run / "norm.npz"); ds = SurfaceDataset(surface, names, norm); dev = select_device(None)
m = build_model(c.get("model", "transolver"), tdir, c["n_hidden"], c["n_layers"], c["slice_num"]).to(dev)
m.load_state_dict(torch.load(run / "model.pt", map_location=dev, weights_only=True)); m.eval()


def cdp_cells(case, cp):
    """Per-cell pressure-drag coefficient (same integration as geometry.force_coefficients)."""
    a = case["aoa_rad"]; pe = cp[case["lines"]].mean(1)
    f = pe[:, None] * case["cell_normal"] * case["length"][:, None]
    return f[:, 0] * np.cos(a) + f[:, 1] * np.sin(a)

regions = {"x<2%": (0, .02), "2-10%": (.02, .10), "10-50%": (.10, .50), "50-100%": (.50, 1.01)}
err = {k: [] for k in regions}; tot = []; peak = []
with torch.no_grad():
    for name, (d, _) in zip(names, ds):
        case = load_case(surface, name)
        pr = norm.inverse_targets(m(to_device(d, dev)).cpu().numpy())
        x = case["pos"][:, 0]; xc = (x[case["lines"]].mean(1) - x.min()) / np.ptp(x)
        dt, dp = cdp_cells(case, case["cp"]), cdp_cells(case, pr[:, 0])
        for k, (lo, hi) in regions.items():
            sel = (xc >= lo) & (xc < hi); err[k].append((dp[sel] - dt[sel]).sum())
        tot.append(dp.sum() - dt.sum())
        peak.append(pr[:, 0].min() - case["cp"].min())  # >0: predicted suction peak weaker
tot = np.array(tot)
print(f"n={len(names)} total Cdp error median {np.median(tot):+.5f}")
for k in regions:
    e = np.array(err[k]); print(f"  {k:8s} median {np.median(e):+.5f}  mean {np.mean(e):+.5f}  share of mean total {np.mean(e) / np.mean(tot):.2f}")
peak = np.array(peak)
print(f"  suction peak (min Cp) pred - CFD: median {np.median(peak):+.3f}; weaker in {np.mean(peak > 0):.2f} of cases;"
      f" Spearman with total Cdp error {spearmanr(peak, tot)[0]:.2f}")
