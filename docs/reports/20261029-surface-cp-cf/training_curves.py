"""Plot training loss and validation Cp/Cf MSE of surface runs against epoch.

Usage: python training_curves.py OUT.png LABEL=RUN_DIR [LABEL=RUN_DIR ...]
Each RUN_DIR holds the log.csv and config.json written by surfcpcf.train.
The dotted line marks each run's OneCycleLR peak (30% of the scheduler steps).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

out, specs = Path(sys.argv[1]), [s.split("=", 1) for s in sys.argv[2:]]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
for (label, run), color in zip(specs, COLORS):
    rows = list(csv.DictReader(open(Path(run) / "log.csv")))
    cfg = json.load(open(Path(run) / "config.json"))
    n_train = len(cfg["splits"]["train"])
    peak_epoch = 0.3 * (n_train + 1) * cfg["epochs"] / n_train
    name = f"{label} (hidden {cfg['n_hidden']}, {cfg['n_layers']} layers, lr {cfg['lr']:g})"
    epochs = [int(r["epoch"]) for r in rows]
    axes[0].plot(epochs, [float(r["train_loss"]) for r in rows], color=color, lw=1.2, label=name)
    val = [r for r in rows if r["val_cp_mse"]]
    for ax, key in zip(axes[1:], ("val_cp_mse", "val_cf_mse")):
        ax.plot([int(r["epoch"]) for r in val], [float(r[key]) for r in val], color=color, lw=1.6, marker="o", ms=3)
    for ax in axes:
        ax.axvline(peak_epoch, color=color, ls=":", lw=1)
titles = ("training loss (normalized field MSE)", "validation Cp MSE (normalized, 80 cases)",
          "validation Cf MSE (normalized, 80 cases)")
for ax, title in zip(axes, titles):
    ax.set(yscale="log", xlabel="epoch", title=title)
    ax.title.set_fontsize(9)
    ax.grid(alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.suptitle("Surface-only Cp/Cf Transolver on AirfRANS full (dotted: peak learning rate of OneCycleLR)", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=130)
