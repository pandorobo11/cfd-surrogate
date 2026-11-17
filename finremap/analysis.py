"""SU2 の surface_flow.vtu を読み、舵角を戻して基準格子へ Cp を補間し、別に解いた解と比べる。

手順:
  1. 舵角 delta の格子の表面節点を、舵軸まわりに -delta 回して翼固定座標に戻す。
  2. 基準格子（舵角 0）の各節点から、元格子の三角形面上の最近接点を求める（裏面の三角形は除く）。
  3. 最近接点の重心座標で、三角形の 3 頂点の Cp を平均する。

検証:
  翼単体では「舵角 delta, 迎角 0」と「舵角 0, 迎角 delta」は同じ流れになる。
  補間した Cp を、基準格子で迎角 delta を解いた結果と比べる。補間そのものの質は、補間前後の力の変化で見る。

粘性計算（設定の flow.reynolds）では、摩擦係数ベクトル Cf も同じ対応で基準格子へ運ぶ:
  4. Cf の成分を、座標と同じ回転で翼固定座標に戻す。
  5. 3 成分をそれぞれ重心座標で補間し、基準格子の節点の法線方向の成分を除く（接平面へ射影）。
"""
import csv
import json
from pathlib import Path

import numpy as np

from . import remap
from .mesh import planform, ref_area
from .vtu import read_vtu, triangle_mesh, write_vtu

CP_NAME = "Pressure_Coefficient"
CF_NAME = "Skin_Friction_Coefficient"
METHODS = ["A 回転を戻さず最近傍節点", "B 回転を戻して最近傍節点", "C 投影＋重心座標", "D C＋法線チェック"]
# Cf の扱いの比較。位置の対応はどれも D（回転を戻して投影＋重心座標＋法線チェック）
CF_METHODS = ["E 成分を回さず補間", "F 成分を回して補間", "G F＋接平面へ射影"]
HINGE_AXIS = [0, 1, 0]


def load_surface(case_dir, delta_deg, geom, viscous=False):
    """表面の VTU を読み、座標・翼固定座標・外向きにそろえた三角形・Cp を返す。

    viscous なら、摩擦係数ベクトル cf（格子の座標系の成分）と cf_fin（翼固定座標の成分）も返す。
    """
    mesh = read_vtu(case_dir / "surface_flow.vtu")
    for name in [CP_NAME] + ([CF_NAME] if viscous else []):
        if name not in mesh.point_data:
            raise ValueError(f"{case_dir.name}: surface_flow.vtu に {name} がない（ある配列: {list(mesh.point_data)}）")
    tri, _ = mesh.triangles()
    origin = [geom["hinge_x"], 0.0, 0.0]
    fin_xyz = remap.rotate(mesh.points, origin, HINGE_AXIS, -delta_deg)
    # 向きの基準にする内部点: 翼端面は面の内側、それ以外は同じ翼幅位置の翼弦線上の点
    cen = fin_xyz[tri].mean(axis=1)
    x_le, chord = planform(geom, cen[:, 1])
    interior = np.stack([x_le + 0.4 * chord, cen[:, 1], np.zeros(len(cen))], axis=1)
    tip = np.all(fin_xyz[tri][:, :, 1] > geom["span"] * (1 - 1e-6), axis=1)
    interior[tip] = cen[tip] - [0.0, 1.0, 0.0]
    tri = remap.orient_outward(fin_xyz, tri, interior)
    out = dict(xyz=mesh.points, fin_xyz=fin_xyz, tri=tri, cp=mesh.point_data[CP_NAME].astype(float),
               run=json.loads((case_dir / "run.json").read_text()))
    if viscous:
        out["cf"] = mesh.point_data[CF_NAME].astype(float)
        if out["cf"].shape != (len(mesh.points), 3):
            raise ValueError(f"{case_dir.name}: {CF_NAME} が 3 成分のベクトルでない（形 {out['cf'].shape}）")
        out["cf_fin"] = remap.rotate_vectors(out["cf"], HINGE_AXIS, -delta_deg)
    return out


def chord_coords(geom, xyz):
    x_le, chord = planform(geom, xyz[:, 1])
    return (xyz[:, 0] - x_le) / chord, xyz[:, 1] / geom["span"]


def edge_triangles(geom, xyz, tri):
    """重心が縁（前縁・後縁から edge_zone 以内）にある三角形の印。翼端面は縁に含めない。"""
    cen = xyz[tri].mean(axis=1)
    xi, _ = chord_coords(geom, cen)
    zone = geom["edge_zone"]
    return ((xi < zone) | (xi > 1.0 - zone)) & (cen[:, 1] < geom["span"] * (1 - 1e-6))


def analyze_friction(geom, pair, src, ref, qn, edge, label):
    """Cf を基準格子へ運び、補間前後の摩擦力と、別に解いた Cf との差を表にする。(行の一覧, 手法ごとの Cf) を返す。"""
    area, origin = ref_area(geom), [geom["hinge_x"], 0.0, 0.0]
    q, delta = ref["xyz"], pair["delta_deg"]

    # SU2 の力の内訳と、VTU の Cp・Cf を積分した力の照合（格子の座標系。読み込みと符号の確認）
    fp, mp = remap.integrate_forces(src["xyz"], src["tri"], src["cp"], area, origin)
    fv, mv = remap.integrate_friction(src["xyz"], src["tri"], src["cf"], area, origin)
    bd = src["run"].get("forces_breakdown") or {}
    print("元格子の力の内訳の照合（SU2 / VTU の積分）")
    for name, vp, vv in [("CFx", fp[0], fv[0]), ("CFz", fp[2], fv[2]), ("CMy", mp[1], mv[1])]:
        b = bd.get(name)
        su2 = (f"{b['pressure']:9.5f}", f"{b['friction']:9.5f}") if b else ("    なし", "    なし")
        print(f"  {name}: 圧力分 {su2[0]} / {vp:9.5f},  摩擦分 {su2[1]} / {vv:9.5f}")

    both, _ = remap.map_barycentric(src["fin_xyz"], src["tri"], np.hstack([src["cf_fin"], src["cf"]]), q, qn)
    maps = {CF_METHODS[0]: both[:, 3:], CF_METHODS[1]: both[:, :3]}
    maps[CF_METHODS[2]] = remap.tangent_project(maps[CF_METHODS[1]], qn)

    edge_s, edge_r = edge_triangles(geom, src["fin_xyz"], src["tri"]), edge_triangles(geom, q, ref["tri"])

    def coeffs(xyz, tri, cf, edge_tri):
        ff, mm = remap.integrate_friction(xyz, tri, cf, area, origin)
        fe, _ = remap.integrate_friction(xyz, tri[edge_tri], cf, area, origin)
        return dict(CA_f=float(ff[0]), CN_f=float(ff[2]), CMy_f=float(mm[1]), CA_f_edge=float(fe[0]),
                    CA_f_rest=float(ff[0] - fe[0]))  # 翼固定軸。縁とそれ以外に分けた軸力も出す

    c_src = coeffs(src["fin_xyz"], src["tri"], src["cf_fin"], edge_s)
    c_ref = coeffs(q, ref["tri"], ref["cf"], edge_r)
    rms = lambda v: float(np.sqrt(np.nanmean(v**2)))
    scale = rms(np.linalg.norm(ref["cf"], axis=1))
    line = lambda name, c: f"{name:<24}{c['CA_f']:10.6f}{c['CA_f_edge']:10.6f}{c['CA_f_rest']:10.6f}{c['CN_f']:10.6f}{c['CMy_f']:10.6f}"
    print(f"摩擦力（翼固定軸）と、基準格子で解いた Cf との差（|Cf| の RMS は {scale:.5f}）")
    print(f"{'':<24}{'CA':>10}{'CA 縁':>10}{'CA 他':>10}{'CN':>10}{'CMy':>10}{'差RMS':>9}{'縁RMS':>9}{'縁最大':>9}{'他RMS':>9}{'他最大':>9}{'法線RMS':>9}")
    print(line("元格子の解（補間前）", c_src))
    rows = [dict(pair=label, delta_deg=delta, method="元格子の解（補間前）", **c_src)]
    for name, cf in maps.items():
        c = coeffs(q, ref["tri"], cf, edge_r)
        d = np.linalg.norm(cf - ref["cf"], axis=1)
        row = dict(pair=label, delta_deg=delta, method=name, **c,
                   CA_f_change_pct=(c["CA_f"] / c_src["CA_f"] - 1) * 100,
                   CA_f_edge_change_pct=(c["CA_f_edge"] / c_src["CA_f_edge"] - 1) * 100,
                   CA_f_rest_change_pct=(c["CA_f_rest"] / c_src["CA_f_rest"] - 1) * 100,
                   CN_f_change=c["CN_f"] - c_src["CN_f"], CMy_f_change=c["CMy_f"] - c_src["CMy_f"],
                   cf_diff_rms=rms(d), cf_diff_rms_edge=rms(d[edge]), cf_diff_max_edge=float(np.nanmax(d[edge])),
                   cf_diff_rms_rest=rms(d[~edge]), cf_diff_max_rest=float(np.nanmax(d[~edge])),
                   cf_normal_rms=rms(np.sum(cf * qn, axis=1)), cf_reference_rms=scale)
        rows.append(row)
        print(line(name, c) + f"{row['cf_diff_rms']:9.5f}{row['cf_diff_rms_edge']:9.5f}{row['cf_diff_max_edge']:9.5f}"
              f"{row['cf_diff_rms_rest']:9.5f}{row['cf_diff_max_rest']:9.5f}{row['cf_normal_rms']:9.5f}")
    print(line("基準格子で迎角を付けた解", c_ref))
    rows.append(dict(pair=label, delta_deg=delta, method="基準格子で迎角を付けた解", **c_ref))
    return rows, maps


def analyze_pair(run, plan, pair, plt):
    geom = plan["geometry"]
    area, origin = ref_area(geom), [geom["hinge_x"], 0.0, 0.0]
    delta = pair["delta_deg"]
    viscous = plan["flow"].get("reynolds") is not None
    src = load_surface(run / "cases" / pair["source"], delta, geom, viscous)
    ref = load_surface(run / "cases" / pair["reference"], 0.0, geom, viscous)
    q, qn = ref["xyz"], remap.node_normals(ref["xyz"], ref["tri"])
    xi, eta = chord_coords(geom, q)
    zone = geom["edge_zone"]
    edge = ((xi < zone) | (xi > 1.0 - zone)) & (q[:, 1] < geom["span"] * (1 - 1e-6))
    label = f"{pair['source']} -> {pair['reference']}"

    # SU2 の力と、表面 Cp（粘性計算では Cp と Cf）を積分した力の照合（読み込みと向きの確認）
    f, m = remap.integrate_forces(src["xyz"], src["tri"], src["cp"], area, origin)
    if viscous:
        fv, mv = remap.integrate_friction(src["xyz"], src["tri"], src["cf"], area, origin)
        f, m = f + fv, m + mv
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
    rms = lambda v: float(np.sqrt(np.nanmean(v**2)))  # 対応する面が見つからない節点（NaN）は除く
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
                   cp_diff_max_edge=float(np.nanmax(np.abs(d[edge]))), cp_diff_rms_rest=rms(d[~edge]),
                   cp_diff_max_rest=float(np.nanmax(np.abs(d[~edge]))))
        rows.append(row)
        print(f"{name:<24}{cn:9.5f}{ca:9.5f}{cm:10.5f}{row['cp_diff_rms']:9.4f}{row['cp_diff_rms_edge']:9.4f}"
              f"{row['cp_diff_max_edge']:9.4f}{row['cp_diff_rms_rest']:9.4f}{row['cp_diff_max_rest']:9.4f}")
    print(f"{'基準格子で迎角を付けた解':<24}{cn_r:9.5f}{ca_r:9.5f}{cm_r:10.5f}")
    rows.append(dict(pair=label, delta_deg=delta, method="基準格子で迎角を付けた解", CN=cn_r, CA=ca_r, CMy=cm_r))

    mapped = maps[METHODS[3]]
    out = run / "analysis"
    data = dict(Cp_mapped=mapped, Cp_reference=ref["cp"], Cp_difference=mapped - ref["cp"],
                Cp_nearest=maps[METHODS[1]], ProjectionDistance=np.where(found, dist, -1.0))
    if viscous:
        cf_rows, cf_maps = analyze_friction(geom, pair, src, ref, qn, edge, label)
        rows += cf_rows
        cf = cf_maps[CF_METHODS[2]]
        data.update(Cf_mapped=cf, Cf_reference=ref["cf"], Cf_difference=cf - ref["cf"],
                    Cf_unprojected=cf_maps[CF_METHODS[1]])
    write_vtu(out / f"remapped-{pair['source']}.vtu", triangle_mesh(q, ref["tri"], data))
    if plt is not None:
        figure(plt, out / f"remap-{pair['source']}.png", geom, plan, pair, src, ref, maps, xi, eta)
        if viscous:
            figure_friction(plt, out / f"remap-cf-{pair['source']}.png", geom, plan, pair, src, ref, cf_maps, xi, eta)
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
        tr.set_mask(~np.isfinite(val)[tr.triangles].all(axis=1))  # NaN の節点を含む三角形は描かない
        im = ax.tripcolor(tr, val, shading="gouraud", cmap="viridis", **lim)
        fig.colorbar(im, ax=ax, label="Cp"); ax.set_title(title)
    for ax, tr, title in [(axes[1, 0], tr_r, "(5) Difference (3) - (4), upper side"),
                          (axes[1, 1], tr_rl, "(6) Same difference, lower side")]:
        tr.set_mask(~np.isfinite(mapped)[tr.triangles].all(axis=1))
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


def figure_friction(plt, path, geom, plan, pair, src, ref, maps, xi, eta):
    from matplotlib.tri import Triangulation

    q, delta = ref["xyz"], pair["delta_deg"]
    raw, rotated, mapped = (maps[k] for k in CF_METHODS)
    n, _ = remap.face_normals_areas(q, ref["tri"])
    upper, lower = (Triangulation(q[:, 0], q[:, 1], ref["tri"][s * n[:, 2] > 0.3]) for s in (1, -1))
    nodes = np.unique(upper.triangles)
    nodes = nodes[(xi[nodes] > geom["edge_zone"]) & (xi[nodes] < 1.0 - geom["edge_zone"])]  # 縁の細かい格子を除く
    lim = dict(zip(("vmin", "vmax"), np.percentile(ref["cf"][nodes, 0], [1, 99])))
    dlim = float(np.percentile(np.linalg.norm(ref["cf"][nodes], axis=1), 50)) * 0.2
    diff = np.linalg.norm(mapped - ref["cf"], axis=1)
    strip = np.abs(eta - 0.5) < 0.5 * plan["meshes"]["ref"]["h_fin"] / geom["span"]
    zoom = geom["edge_zone"] * 4.0 / 3.0
    mag = lambda v: np.linalg.norm(v, axis=1)
    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5))

    def wrap_plot(ax, mask, coord, title, xlabel):
        o = np.argsort(coord)
        ax.plot(coord[o], mag(ref["cf"])[mask][o], "k-", lw=1.8, label=f"reference mesh, AoA {delta:g}")
        ax.plot(coord[o], mag(rotated)[mask][o], "o", color="tab:orange", ms=4, mfc="none", label="rotated, not projected F")
        ax.plot(coord[o], mag(mapped)[mask][o], "o", color="tab:green", ms=2.5, label="rotated + projected G")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlabel(xlabel); ax.set_ylabel("|Cf|"); ax.set_title(title)

    m = strip & (xi > 1.0 - zoom)
    wrap_plot(axes[0, 0], m, np.sign(q[m, 2]) * np.sqrt(np.clip(1.0 - xi[m], 0, None)),
              f"(1) Around the trailing edge (x/c > {1 - zoom:.2f})", "sign(z) * sqrt(1 - x/c)   (lower < 0 < upper)")
    for ax, val, title in [(axes[0, 1], mapped[:, 0], "(2) Cf_x mapped (G), upper side"),
                           (axes[0, 2], ref["cf"][:, 0], f"(3) Cf_x, reference run, AoA {delta:g}")]:
        upper.set_mask(~np.isfinite(val)[upper.triangles].all(axis=1))  # NaN の節点を含む三角形は描かない
        im = ax.tripcolor(upper, val, shading="gouraud", cmap="viridis", **lim)
        fig.colorbar(im, ax=ax, label="Cf_x"); ax.set_title(title)
    for ax, tr, title in [(axes[0, 3], upper, "(4) |Cf difference|, upper side"),
                          (axes[1, 0], lower, "(5) Same difference, lower side")]:
        tr.set_mask(~np.isfinite(diff)[tr.triangles].all(axis=1))
        im = ax.tripcolor(tr, diff, shading="gouraud", cmap="magma_r", vmin=0.0, vmax=dlim)
        fig.colorbar(im, ax=ax, label="|Cf difference|"); ax.set_title(title)
    for ax in [axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 0]]:
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")

    for ax, comp, name in [(axes[1, 1], 0, "Cf_x"), (axes[1, 2], 2, "Cf_z")]:
        for mask, mk, lab in [(strip & (q[:, 2] >= 0), "o", "upper"), (strip & (q[:, 2] < 0), "s", "lower")]:
            o = np.argsort(xi[mask])
            ax.plot(xi[mask][o], ref["cf"][mask, comp][o], "k-", lw=1.8, label=f"reference mesh, AoA {delta:g} ({lab})")
            ax.plot(xi[mask][o], raw[mask, comp][o], mk, color="tab:red", ms=3, mfc="none", label=f"components not rotated E ({lab})")
            ax.plot(xi[mask][o], mapped[mask, comp][o], mk, color="tab:green", ms=2, label=f"rotated + projected G ({lab})")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_xlabel("chordwise position x/c"); ax.set_ylabel(name)
        ax.set_title(f"({6 + comp // 2}) {name} in a strip at mid span")
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
