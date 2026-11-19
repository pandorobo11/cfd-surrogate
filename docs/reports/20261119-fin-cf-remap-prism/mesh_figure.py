"""基準格子（舵角 0）の対称面 (y=0) の格子を図にする。付け根の断面で、プリズム層と外側の四面体の境界面が見える。

使い方（リポジトリの直下で）:
  python3 docs/reports/20261119-fin-cf-remap-prism/mesh_figure.py <実行フォルダ>/meshes/ref/mesh.su2 <出力.png>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402


def read(path):
    with open(path) as f:
        f.readline()
        nelem = int(f.readline().split("=")[1])
        elems = [f.readline() for _ in range(nelem)]
        npoin = int(f.readline().split("=")[1])
        xyz = np.loadtxt([f.readline() for _ in range(npoin)], usecols=(0, 1, 2))
        nmark = int(f.readline().split("=")[1])
        markers = {}
        for _ in range(nmark):
            name = f.readline().split("=")[1].strip()
            n = int(f.readline().split("=")[1])
            markers[name] = [list(map(int, f.readline().split()))[1:] for _ in range(n)]
    prisms = np.array([list(map(int, e.split()[1:7])) for e in elems if e.startswith("13 ")])
    return xyz, prisms, markers


def main(mesh, out):
    xyz, prisms, markers = read(mesh)
    sym = markers["sym"]
    lines = []
    for e in sym:
        q = xyz[e][:, [0, 2]]
        lines += [np.stack([q[i], q[(i + 1) % len(q)]]) for i in range(len(q))]
    lines = np.array(lines)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [("Root section (symmetry plane y = 0)", (-0.05, 1.15), (-0.12, 0.12)),
             ("Leading edge", (-0.012, 0.03), (-0.016, 0.016)),
             ("Trailing edge", (0.96, 1.015), (-0.02, 0.02))]
    for ax, (title, xl, zl) in zip(axes, views):
        ax.add_collection(LineCollection(lines, colors="0.2", linewidths=0.3))
        ax.set_xlim(*xl)
        ax.set_ylim(*zl)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"対称面の要素 {len(sym)} 個（四角形 {sum(len(e) == 4 for e in sym)}）、プリズム {len(prisms)} 個")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
