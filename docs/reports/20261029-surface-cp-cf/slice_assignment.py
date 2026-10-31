"""Physics-Attention slice assignment of airfoil surface points for a trained surfcpcf Transolver run.

For every block and head, slice weights are softmax(in_project_slice(x_mid) / temperature), as in the
official Physics_Attention_Irregular_Mesh. Each point is assigned to its highest-weight slice.
Usage (repository root): python slice_assignment.py <airfrans_surface> <transolver_dir> <run_dir> <out_dir> [case ...]
"""
import json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from surfcpcf.data import Normalization, SurfaceDataset, load_case, read_splits, to_device
from surfcpcf.model import build_model, select_device

surface, tdir, run, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
show = sys.argv[5:]
out.mkdir(parents=True, exist_ok=True)
c = json.loads((run / "config.json").read_text())
names = read_splits(surface, c["task"])["test"]
norm = Normalization.load(run / "norm.npz"); ds = SurfaceDataset(surface, names, norm); dev = select_device(None)
model = build_model("transolver", tdir, c["n_hidden"], c["n_layers"], c["slice_num"]).to(dev)
model.load_state_dict(torch.load(run / "model.pt", map_location=dev, weights_only=True)); model.eval()

captured = {}
for i, block in enumerate(model.blocks):
    attn = block.Attn
    def hook(module, inputs, output, i=i, attn=attn):
        captured[i] = torch.softmax(output / attn.temperature, dim=-1)[0].detach().cpu().numpy()  # H N G
    attn.in_project_slice.register_forward_hook(hook)

stats = []  # per case: layer x head -> (used slices, mean max weight, LE-specific slices)
assignments = {}
with torch.no_grad():
    for name, (d, _) in zip(names, ds):
        model(to_device(d, dev))
        case = load_case(surface, name); x = case["pos"][:, 0]
        le = (x - x.min()) / np.ptp(x) < 0.02
        per = np.zeros((len(captured), captured[0].shape[0], 3))
        for layer, w in captured.items():
            arg = w.argmax(-1)  # H N
            for h in range(w.shape[0]):
                used = np.unique(arg[h])
                le_specific = sum(1 for g in used if le[arg[h] == g].mean() >= 0.8)
                per[layer, h] = (len(used), w[h].max(-1).mean(), le_specific)
        stats.append(per)
        if name in show:
            assignments[name] = {layer: w.argmax(-1) for layer, w in captured.items()}
stats = np.array(stats)  # cases x layers x heads x 3
summary = {"cases": len(names), "slice_num": c["slice_num"],
           "per_layer": [{"layer": l + 1,
                          "used_slices_median_over_heads_and_cases": float(np.median(stats[:, l, :, 0])),
                          "mean_max_weight": float(stats[:, l, :, 1].mean()),
                          "le_specific_slices_mean_per_head": float(stats[:, l, :, 2].mean()),
                          "heads_with_le_specific_slice_fraction": float((stats[:, l, :, 2] > 0).mean())}
                         for l in range(stats.shape[1])]}
(out / "slice_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

cmap = plt.get_cmap("hsv", c["slice_num"])
for name, layers in assignments.items():
    case = load_case(surface, name); pos = case["pos"]; cp = case["cp"]
    L, H = len(layers), layers[0].shape[0]
    fig, axes = plt.subplots(L + 1, H, figsize=(2.2 * H, 1.25 * (L + 1)), constrained_layout=True)
    for h in range(H):
        ax = axes[0, h]
        ax.scatter(pos[:, 0], pos[:, 1] * 3, c=cp, cmap="coolwarm_r", s=2, vmin=-2, vmax=1)
        ax.set_axis_off(); ax.set_title("CFD Cp" if h == 0 else "", fontsize=7)
        for l in range(L):
            ax = axes[l + 1, h]
            ax.scatter(pos[:, 0], pos[:, 1] * 3, c=layers[l][h], cmap=cmap, vmin=-.5, vmax=c["slice_num"] - .5, s=2)
            ax.set_axis_off()
            if l == 0: ax.set_title(f"head {h + 1}", fontsize=7)
        axes[1 + 0, 0].text(-0.15, 0, "", fontsize=6)
    for l in range(L):
        axes[l + 1, 0].text(-0.08, 0.5, f"layer {l + 1}", transform=axes[l + 1, 0].transAxes, rotation=90,
                            va="center", ha="right", fontsize=7)
    meta = json.loads(str(case["metadata"]))
    fig.suptitle(f"Highest-weight slice of each surface point (colour = slice index, y stretched x3)\n"
                 f"{name}  U={meta['uinf']:.1f} m/s  AoA={meta['aoa_deg']:.2f} deg", fontsize=8)
    fig.savefig(out / f"slices_{name}.png", dpi=140); plt.close(fig)
    print("figure", out / f"slices_{name}.png")
