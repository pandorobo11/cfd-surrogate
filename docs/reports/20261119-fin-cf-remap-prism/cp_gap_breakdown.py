"""補間した Cp と、基準格子で別に解いた Cp の法線力の差を、翼面の場所ごとに分ける。

analyze が書く analysis/remapped-<ケース>.vtu（基準格子の翼面。Cp_mapped と Cp_reference を持つ）を読む。
三角形ごとに、3 頂点の平均の Cp 差 × 面積ベクトルの z 成分（翼固定軸の法線力）を足し、上面・下面、
翼弦方向の位置 xi、翼幅方向の位置 eta で分ける。値は基準の解の法線力に対する % で出す。

外向きの法線は、三角形の向きによらず、翼の中心面 z=0 から離れる向き（翼端面は +y 向き）に取る。

使い方（リポジトリの直下で）:
  python3 docs/reports/20261119-fin-cf-remap-prism/cp_gap_breakdown.py <ラベル>=<実行フォルダ>... > cp_gap_breakdown.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from finremap.mesh import planform, ref_area  # noqa: E402
from finremap.vtu import read_vtu  # noqa: E402

XI_BINS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)]
ETA_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def breakdown(vtu, geom):
    m = read_vtu(vtu)
    f = lambda k: m.point_data[k].astype(float)
    return breakdown_fields(m.points, m.triangles()[0], f("Cp_mapped"), f("Cp_reference"), geom)


def breakdown_fields(points, tri, cp_a, cp_b, geom):
    """翼面の節点値 cp_a − cp_b の法線力を場所ごとに分ける（cp_b の法線力に対する %）。"""
    pt = points[tri]
    c = pt.mean(axis=1)
    a = 0.5 * np.cross(pt[:, 1] - pt[:, 0], pt[:, 2] - pt[:, 0])
    side = np.abs(a[:, 2]) > np.abs(a[:, 1])
    s = np.where(side, np.sign(a[:, 2] * c[:, 2]), np.sign(a[:, 1] * (c[:, 1] - 0.5 * geom["span"])))
    a = a * s[:, None]
    cpa, cpb = cp_a[tri].mean(axis=1), cp_b[tri].mean(axis=1)
    dcp = cpa - cpb
    d_cn = -dcp * a[:, 2] / ref_area(geom)
    cn = float(-(cpb * a[:, 2]).sum() / ref_area(geom))
    x_le, chord = planform(geom, c[:, 1])
    xi, eta = (c[:, 0] - x_le) / chord, c[:, 1] / geom["span"]
    up = c[:, 2] > 0
    pct = lambda k: round(100.0 * float(d_cn[k].sum()) / cn, 3)
    within = lambda v, lo, hi: (v >= lo) & ((v < hi) | (hi == 1.0))  # 1.0 を含める（後縁・翼端）
    w = np.linalg.norm(a, axis=1)
    mid = within(xi, 0.3, 0.7)
    return dict(
        CN_reference_triangle=round(cn, 5),
        total_pct=pct(np.ones_like(up)),
        upper_pct=pct(up), lower_pct=pct(~up),
        xi_pct={f"{lo:.1f}-{hi:.1f}": pct(within(xi, lo, hi)) for lo, hi in XI_BINS},
        eta_pct={f"{lo:.1f}-{hi:.1f}": pct(within(eta, lo, hi)) for lo, hi in ETA_BINS},
        mean_dcp_xi_0p3_0p7=dict(upper=round(float(np.average(dcp[mid & up], weights=w[mid & up])), 5),
                                 lower=round(float(np.average(dcp[mid & ~up], weights=w[mid & ~up])), 5)),
    )


if __name__ == "__main__":
    out = []
    for arg in sys.argv[1:]:
        label, run = arg.split("=", 1)
        run = Path(run)
        plan = json.loads((run / "plan.json").read_text())
        for f in sorted((run / "analysis").glob("remapped-*.vtu")):
            out.append(dict(run=label, source=f.stem.removeprefix("remapped-"), **breakdown(f, plan["geometry"])))
    json.dump(dict(rows=out), sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
