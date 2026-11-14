"""SU2 の surface_flow.vtu を読み、舵角を戻して基準格子へ Cp を補間し、別に解いた解と比べる。

手順:
  1. 舵角 delta の格子の表面節点を、舵軸まわりに -delta 回して翼固定座標に戻す。
  2. 基準格子（舵角 0）の各節点から、元格子の三角形面上の最近接点を求める（裏面の三角形は除く）。
  3. 最近接点の重心座標で、三角形の 3 頂点の Cp を平均する。

検証:
  翼単体では「舵角 delta, 迎角 0」と「舵角 0, 迎角 delta」は同じ流れになる。
  補間した Cp を、基準格子で迎角 delta を解いた結果と比べる。補間そのものの質は、補間前後の力の変化で見る。
"""
import csv
import json
from pathlib import Path

import numpy as np

from . import remap
from .mesh import planform, ref_area
from .vtu import read_vtu, triangle_mesh, write_vtu

CP_NAME = "Pressure_Coefficient"
METHODS = ["A 回転を戻さず最近傍節点", "B 回転を戻して最近傍節点", "C 投影＋重心座標", "D C＋法線チェック"]


def load_surface(case_dir, delta_deg, geom):
    """表面の VTU を読み、座標・翼固定座標・外向きにそろえた三角形・Cp を返す。"""
    mesh = read_vtu(case_dir / "surface_flow.vtu")
    if CP_NAME not in mesh.point_data:
        raise ValueError(f"{case_dir.name}: surface_flow.vtu に {CP_NAME} がない（ある配列: {list(mesh.point_data)}）")
    tri, _ = mesh.triangles()
    origin = [geom["hinge_x"], 0.0, 0.0]
    fin_xyz = remap.rotate(mesh.points, origin, [0, 1, 0], -delta_deg)
    # 向きの基準にする内部点: 翼端面は面の内側、それ以外は同じ翼幅位置の翼弦線上の点
    cen = fin_xyz[tri].mean(axis=1)
    x_le, chord = planform(geom, cen[:, 1])
    interior = np.stack([x_le + 0.4 * chord, cen[:, 1], np.zeros(len(cen))], axis=1)
    tip = np.all(fin_xyz[tri][:, :, 1] > geom["span"] * (1 - 1e-6), axis=1)
    interior[tip] = cen[tip] - [0.0, 1.0, 0.0]
    tri = remap.orient_outward(fin_xyz, tri, interior)
    return dict(xyz=mesh.points, fin_xyz=fin_xyz, tri=tri, cp=mesh.point_data[CP_NAME].astype(float),
                run=json.loads((case_dir / "run.json").read_text()))


def chord_coords(geom, xyz):
    x_le, chord = planform(geom, xyz[:, 1])
    return (xyz[:, 0] - x_le) / chord, xyz[:, 1] / geom["span"]


def analyze_pair(run, plan, pair, plt):
    geom = plan["geometry"]
    area, origin = ref_area(geom), [geom["hinge_x"], 0.0, 0.0]
    delta = pair["delta_deg"]
    src = load_surface(run / "cases" / pair["source"], delta, geom)
    ref = load_surface(run / "cases" / pair["reference"], 0.0, geom)
    q, qn = ref["xyz"], remap.node_normals(ref["xyz"], ref["tri"])
    xi, eta = chord_coords(geom, q)
    zone = geom["edge_zone"]
    edge = ((xi < zone) | (xi > 1.0 - zone)) & (q[:, 1] < geom["span"] * (1 - 1e-6))
    label = f"{pair['source']} -> {pair['reference']}"

    # SU2 の力と、表面 Cp を積分した力の照合（読み込みと向きの確認）
    f, m = remap.integrate_forces(src["xyz"], src["tri"], src["cp"], area, origin)
    h = src["run"]["final"]
    print(f"\n=== {label} ===")
    print(f"元格子 節点 {len(src['xyz'])}, 三角形 {len(src['tri'])} / 基準格子 節点 {len(q)}, 三角形 {len(ref['tri'])}"
          f"（縁から {zone * 100:.0f}% 翼弦以内の節点 {int(edge.sum())}）")
    print(f"状態: {pair['source']} {src['run']['outcome']} (log10 rms[Rho] {h['rms_rho']:.2f}), "
          f"{pair['reference']} {ref['run']['outcome']} ({ref['run']['final']['rms_rho']:.2f})")
    print(f"元格子の積分の照合  CFz: SU2 {h['CFz']:.5f} / VTU の積分 {f[2]:.5f},  "
          f"CMy: SU2 {h['CMy']:.5f} / VTU の積分 {m[1]:.5f}")

    maps = {METHODS[0]: remap.map_nearest(src["xyz"], src["cp"], q),
            METHODS[1]: remap.map_nearest(src["fin_xyz"], src["cp"], q)}
    maps[METHODS[2]], _ = remap.map_barycentric(src["fin_xyz"], src["tri"], src["cp"], q)
    maps[METHODS[3]], dist = remap.map_barycentric(src["fin_xyz"], src["tri"], src["cp"], q, qn)
    found = np.isfinite(dist)
    print(f"投影距離: 中央値 {np.median(dist[found]):.1e}, 最大 {dist[found].max():.1e}, "
          f"対応する面が見つからない節点 {int((~found).sum())}")

    def coeffs(xyz, tri, cp):
        ff, mm = remap.integrate_forces(xyz, tri, cp, area, origin)
        return float(ff[2]), float(ff[0]), float(mm[1])  # 翼固定軸の CN, CA, CMy

    cn_s, ca_s, cm_s = coeffs(src["fin_xyz"], src["tri"], src["cp"])
    cn_r, ca_r, cm_r = coeffs(q, ref["tri"], ref["cp"])
    rms = lambda v: float(np.sqrt(np.mean(v**2)))
    rows = [dict(pair=label, delta_deg=delta, method="元格子の解（補間前）", CN=cn_s, CA=ca_s, CMy=cm_s,
                 source_nodes=len(src["xyz"]), reference_nodes=len(q), proj_dist_median=float(np.median(dist[found])),
                 proj_dist_max=float(dist[found].max()), unmatched_nodes=int((~found).sum()))]
    print(f"{'':<24}{'CN':>9}{'CA':>9}{'CMy':>10}{'Cp差RMS':>9}{'縁RMS':>9}{'縁最大':>9}{'他RMS':>9}{'他最大':>9}")
    print(f"{'元格子の解（補間前）':<24}{cn_s:9.5f}{ca_s:9.5f}{cm_s:10.5f}")
    for name, cp in maps.items():
        cn, ca, cm = coeffs(q, ref["tri"], cp)
        d = cp - ref["cp"]
        row = dict(pair=label, delta_deg=delta, method=name, CN=cn, CA=ca, CMy=cm,
                   CN_change_pct=(cn / cn_s - 1) * 100, CA_change_pct=(ca / ca_s - 1) * 100,
                   CMy_change_pct=(cm / cm_s - 1) * 100, cp_diff_rms=rms(d), cp_diff_rms_edge=rms(d[edge]),
                   cp_diff_max_edge=float(np.abs(d[edge]).max()), cp_diff_rms_rest=rms(d[~edge]),
                   cp_diff_max_rest=float(np.abs(d[~edge]).max()))
        rows.append(row)
        print(f"{name:<24}{cn:9.5f}{ca:9.5f}{cm:10.5f}{row['cp_diff_rms']:9.4f}{row['cp_diff_rms_edge']:9.4f}"
              f"{row['cp_diff_max_edge']:9.4f}{row['cp_diff_rms_rest']:9.4f}{row['cp_diff_max_rest']:9.4f}")
    print(f"{'基準格子で迎角を付けた解':<24}{cn_r:9.5f}{ca_r:9.5f}{cm_r:10.5f}")
    rows.append(dict(pair=label, delta_deg=delta, method="基準格子で迎角を付けた解", CN=cn_r, CA=ca_r, CMy=cm_r))

    mapped = maps[METHODS[3]]
    out = run / "analysis"
    write_vtu(out / f"remapped-{pair['source']}.vtu", triangle_mesh(q, ref["tri"], dict(
        Cp_mapped=mapped, Cp_reference=ref["cp"], Cp_difference=mapped - ref["cp"],
        Cp_nearest=maps[METHODS[1]], ProjectionDistance=np.where(found, dist, -1.0))))
    if plt is not None:
        figure(plt, out / f"remap-{pair['source']}.png", geom, plan, pair, src, ref, maps, xi, eta)
    return rows


def figure(plt, path, geom, plan, pair, src, ref, maps, xi, eta):
    from matplotlib.tri import Triangulation

    q, delta = ref["xyz"], pair["delta_deg"]
    mapped, nearest = maps[METHODS[3]], maps[METHODS[1]]

    def side(m, xyz, sign):
        n, _ = remap.face_normals_areas(xyz, m["tri"])
        return Triangulation(xyz[:, 0], xyz[:, 1], m["tri"][sign * n[:, 2] > 0.3])

    tr_s, tr_r, tr_rl = side(src, src["fin_xyz"], 1), side(ref, q, 1), side(ref, q, -1)
    upper = np.unique(tr_r.triangles)
    upper = upper[(xi[upper] > geom["edge_zone"]) & (xi[upper] < 1.0 - geom["edge_zone"])]  # 縁の細かい格子を除く
    lim = dict(zip(("vmin", "vmax"), np.percentile(ref["cp"][upper], [1, 99])))
    strip = np.abs(eta - 0.5) < 0.5 * plan["meshes"]["ref"]["h_fin"] / geom["span"]
    zoom = geom["edge_zone"] * 4.0 / 3.0
    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5))

    def wrap_plot(ax, mask, coord, title, xlabel):
        o = np.argsort(coord)
        ax.plot(coord[o], ref["cp"][mask][o], "k-", lw=1.8, label=f"reference mesh, AoA {delta:g}")
        ax.plot(coord[o], nearest[mask][o], "o", color="tab:orange", ms=4, mfc="none", label="nearest node B")
        ax.plot(coord[o], mapped[mask][o], "o", color="tab:green", ms=2.5, label="projection D")
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlabel(xlabel); ax.set_ylabel("Cp"); ax.set_title(title)

    m = strip & (xi > 1.0 - zoom)
    wrap_plot(axes[0, 0], m, np.sign(q[m, 2]) * np.sqrt(np.clip(1.0 - xi[m], 0, None)),
              f"(1) Around the trailing edge (x/c > {1 - zoom:.2f})", "sign(z) * sqrt(1 - x/c)   (lower < 0 < upper)")
    for ax, tr, val, title in [
        (axes[0, 1], tr_s, src["cp"], f"(2) SU2 Cp, {delta:g} deg mesh (upper side)"),
        (axes[0, 2], tr_r, mapped, "(3) Mapped onto reference mesh (D)"),
        (axes[0, 3], tr_r, ref["cp"], f"(4) Separate run: reference, AoA {delta:g}"),
    ]:
        im = ax.tripcolor(tr, val, shading="gouraud", cmap="viridis", **lim)
        fig.colorbar(im, ax=ax, label="Cp"); ax.set_title(title)
    for ax, tr, title in [(axes[1, 0], tr_r, "(5) Difference (3) - (4), upper side"),
                          (axes[1, 1], tr_rl, "(6) Same difference, lower side")]:
        im = ax.tripcolor(tr, mapped - ref["cp"], shading="gouraud", cmap="RdBu_r", vmin=-0.03, vmax=0.03)
        fig.colorbar(im, ax=ax, label="Cp difference"); ax.set_title(title)
    for ax in list(axes[0, 1:]) + [axes[1, 0], axes[1, 1]]:
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")

    ax = axes[1, 2]
    for mask, mk, lab in [(strip & (q[:, 2] >= 0), "o", "upper"), (strip & (q[:, 2] < 0), "s", "lower")]:
        o = np.argsort(xi[mask])
        ax.plot(xi[mask][o], ref["cp"][mask][o], "k-", lw=1.8, label=f"reference mesh, AoA {delta:g} ({lab})")
        ax.plot(xi[mask][o], nearest[mask][o], mk, color="tab:orange", ms=3, mfc="none", label=f"nearest node B ({lab})")
        ax.plot(xi[mask][o], mapped[mask][o], mk, color="tab:green", ms=2, label=f"projection D ({lab})")
    ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_xlabel("chordwise position x/c"); ax.set_ylabel("Cp")
    ax.set_title("(7) Reference-mesh nodes in a strip at mid span")
    m = strip & (xi < zoom)
    wrap_plot(axes[1, 3], m, np.sign(q[m, 2]) * np.sqrt(np.clip(xi[m], 0, None)),
              f"(8) Around the leading edge (x/c < {zoom:.2f})", "sign(z) * sqrt(x/c)   (lower < 0 < upper)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analyze(run, figures=True):
    run = Path(run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    plt = None
    if figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    rows, skipped = [], []
    for pair in plan["pairs"]:
        outcomes = []
        for cid in (pair["source"], pair["reference"]):
            p = run / "cases" / cid / "run.json"
            outcomes.append(json.loads(p.read_text()).get("outcome") if p.exists() else "未実行")
        if not all(o in ("converged", "unconverged") for o in outcomes):
            skipped.append(dict(pair=f"{pair['source']} -> {pair['reference']}", outcomes=outcomes))
            print(f"\n{pair['source']} -> {pair['reference']}: 使える解がないので飛ばす（{outcomes}）")
            continue
        rows += analyze_pair(run, plan, pair, plt)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(rows=rows, skipped=skipped), indent=1, ensure_ascii=False) + "\n")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n出力: {out}（summary.json, results.csv, remap-*.png, remapped-*.vtu）")
    return rows
