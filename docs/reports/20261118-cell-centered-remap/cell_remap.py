"""SU2 の翼面の Cp をセル中心の値に変え、つながりから作ったセル中心の三角形格子で基準格子へ補間する。

手順（舵角 delta の格子 → 舵角 0 の基準格子）:
  1. 節点の Cp を三角形ごとに 3 頂点で平均し、セル値にする（元格子も基準格子も）。
  2. 元格子を舵軸まわりに -delta 回し、節点の環からセル中心の三角形格子を作る（remap.centroid_mesh）。
  3. 基準格子の三角形の重心を、その格子に投影する。基準格子の三角形の法線との角度が 60° を超える三角形は除く。
  4. 投影した点の重心座標で、3 つのセル値を平均する。

比べる手法（どれも基準格子の三角形ごとの値を作る）:
  RING   上の手順（つながりから作ったセル中心の三角形格子）。三角形が覆わない縁の帯の点は、最も近い縁の辺の上の値になる
  RING-X RING と同じ。ただし縁の帯の点は、最も近い三角形の平面で重心座標を延長する（線形の外挿）
  FACE   投影した点を含む元の三角形のセル値をそのまま使う
  NEAR3  法線の角度が 60° 以内のセル中心から最も近い 3 点を選び、その平面に射影して重心座標で補間する
  NODE   節点値のまま補間し（finremap の手法 D）、基準格子の 3 頂点で平均する。節点値がある場合の参考

使い方（リポジトリの直下で）:
  python3 docs/reports/20261118-cell-centered-remap/cell_remap.py results/finremap/fin-round-m2-20261114 \
      --arrays results/finremap/cell-remap-20261118 > docs/reports/20261118-cell-centered-remap/summary.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from finremap import analysis, remap  # noqa: E402
from finremap.mesh import ref_area  # noqa: E402

FEATURE_ANGLE_DEG = 45.0
MIN_COS = 0.5  # 法線の角度 60°
METHODS = ["RING", "RING-X", "FACE", "NEAR3", "NODE"]

# 三角形上の 7 点の求積（5 次まで厳密）
_A1, _B1, _A2, _B2 = 0.0597158717, 0.4701420641, 0.7974269853, 0.1012865073
_QB = np.array([[1 / 3, 1 / 3, 1 / 3], [_A1, _B1, _B1], [_B1, _A1, _B1], [_B1, _B1, _A1],
                [_A2, _B2, _B2], [_B2, _A2, _B2], [_B2, _B2, _A2]])
_QW = np.array([0.225] + [0.1323941527] * 3 + [0.1259391805] * 3)


def analytic(p):
    """解析的な場。z の項で上面と下面の符号が変わり、丸い前縁のまわりで急に変わる。"""
    return np.sin(5.0 * p[:, 0]) * np.cos(3.0 * p[:, 1]) + 8.0 * p[:, 2]


def face_mean(func, points, tri):
    q = np.einsum("qj,fjd->fqd", _QB, points[tri])
    return np.einsum("q,fq->f", _QW, func(q.reshape(-1, 3)).reshape(len(tri), -1))


def nearest3(cen, cn, values, q, qn, k=16):
    """法線の角度が 60° 以内のセル中心から最も近い 3 点で、重心座標（外挿を含む）により補間する。"""
    _, i = cKDTree(cen).query(q, k=k)
    ok = np.einsum("nkd,nd->nk", cn[i], qn) >= MIN_COS
    order = np.argsort(~ok, axis=1, kind="stable")[:, :3]
    sel = np.take_along_axis(i, order, 1)
    enough = np.take_along_axis(ok, order, 1).all(1)
    w = remap.plane_barycentric(q, cen[sel[:, 0]], cen[sel[:, 1]], cen[sel[:, 2]])
    val = np.sum(w * values[sel], 1)
    val[~enough] = np.nan
    return val, w, sel


def stats(d, masks):
    out = {}
    for name, m in masks.items():
        x = d[m]
        x = x[np.isfinite(x)]
        out[name] = dict(rms=float(np.sqrt(np.mean(x**2))), max=float(np.max(np.abs(x))), n=int(len(x)))
    return out


def run_pair(run, plan, pair, arrays_dir):
    geom = plan["geometry"]
    area, origin = ref_area(geom), [geom["hinge_x"], 0.0, 0.0]
    src = analysis.load_surface(run / "cases" / pair["source"], pair["delta_deg"], geom)
    ref = analysis.load_surface(run / "cases" / pair["reference"], 0.0, geom)
    sp, st, rp, rt = src["fin_xyz"], src["tri"], ref["xyz"], ref["tri"]

    # 1. セル値
    src_cell, ref_cell = remap.cell_values(st, src["cp"]), remap.cell_values(rt, ref["cp"])
    q = rp[rt].mean(axis=1)
    qn, _ = remap.face_normals_areas(rp, rt)
    sn, _ = remap.face_normals_areas(sp, st)

    # 2. セル中心の三角形格子
    t0 = time.perf_counter()
    cen, ctri, info = remap.centroid_mesh(sp, st, FEATURE_ANGLE_DEG)
    t_mesh = time.perf_counter() - t0

    # 3-4. 補間
    timing, maps, extra = {"centroid_mesh": t_mesh}, {}, {}
    t0 = time.perf_counter()
    maps["RING"], dist, idx, w = remap.map_barycentric(cen, ctri, src_cell, q, qn, min_cos=MIN_COS, return_index=True)
    timing["RING"] = time.perf_counter() - t0 + t_mesh
    uncovered = remap.boundary_foot(ctri, idx, w, len(st))
    t_ring = ctri[np.maximum(idx, 0)]
    wx = w.copy()
    wx[uncovered] = remap.plane_barycentric(q[uncovered], *(cen[t_ring[uncovered, j]] for j in range(3)))
    maps["RING-X"] = np.where(idx >= 0, np.einsum("nj,nj->n", wx, src_cell[t_ring]), np.nan)
    timing["RING-X"] = timing["RING"]
    # 同じ手順を 1 つの関数にまとめた remap.map_cell_values と一致することを確かめる
    lib, _, lib_band = remap.map_cell_values(sp, st, src_cell, q, qn, FEATURE_ANGLE_DEG, min_cos=MIN_COS)
    assert np.array_equal(lib_band, uncovered) and np.allclose(lib, maps["RING-X"], equal_nan=True)
    _, _, idx_nn, _ = remap.map_barycentric(cen, ctri, src_cell, q, return_index=True)  # 法線を見ない場合
    t0 = time.perf_counter()
    _, _, fidx, _ = remap.map_barycentric(sp, st, src_cell, q, qn, min_cos=MIN_COS, return_index=True)
    maps["FACE"] = np.where(fidx >= 0, src_cell[np.maximum(fidx, 0)], np.nan)
    timing["FACE"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    maps["NEAR3"], w3, sel3 = nearest3(cen, sn, src_cell, q, qn)
    timing["NEAR3"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    node_mapped, _ = remap.map_barycentric(sp, st, src["cp"], rp, remap.node_normals(rp, rt))
    maps["NODE"] = remap.cell_values(rt, node_mapped)
    timing["NODE"] = time.perf_counter() - t0

    # 解析的な場（同じ対応で、厳密な面平均と比べる。CFD の格子の違いを含まない）
    src_a, ref_a = face_mean(analytic, sp, st), face_mean(analytic, rp, rt)
    amaps = {"RING": np.where(idx >= 0, np.einsum("nj,nj->n", w, src_a[ctri[np.maximum(idx, 0)]]), np.nan),
             "RING-X": np.where(idx >= 0, np.einsum("nj,nj->n", wx, src_a[t_ring]), np.nan),
             "FACE": np.where(fidx >= 0, src_a[np.maximum(fidx, 0)], np.nan),
             "NEAR3": nearest3(cen, sn, src_a, q, qn)[0],
             "NODE": remap.cell_values(rt, remap.map_barycentric(sp, st, analytic(sp), rp,
                                                                  remap.node_normals(rp, rt))[0])}

    # 領域: 縁（前縁・後縁から 3% 翼弦以内）、翼端面、その他。縁の帯に入った点は別に数える
    edge = analysis.edge_triangles(geom, rp, rt)
    tip = np.all(rp[rt][:, :, 1] > geom["span"] * (1 - 1e-6), axis=1)
    rest = ~edge & ~tip
    masks = dict(all=np.ones(len(rt), bool), edge=edge, tip_face=tip, rest=rest,
                 uncovered=uncovered, covered=~uncovered)

    def coeffs(points, tri, cell):
        f, m = remap.integrate_forces_cells(points, tri, cell, area, origin)
        return dict(CN=float(f[2]), CA=float(f[0]), CMy=float(m[1]))

    c_src, c_ref = coeffs(sp, st, src_cell), coeffs(rp, rt, ref_cell)
    rows = []
    for name in METHODS:
        c = coeffs(rp, rt, maps[name])
        rows.append(dict(method=name, **c, **{f"{k}_change_pct": (c[k] / c_src[k] - 1) * 100 for k in c},
                         nan_faces=int(np.isnan(maps[name]).sum()), time_s=timing[name],
                         cp_diff=stats(maps[name] - ref_cell, masks),
                         analytic_err=stats(amaps[name] - ref_a, masks)))
    y = q[:, 1]
    where = lambda m: dict(root=int(np.sum(m & (y < 0.02))), tip=int(np.sum(m & (y > geom["span"] - 0.02))),
                           other=int(np.sum(m & (y >= 0.02) & (y <= geom["span"] - 0.02))))
    out = dict(pair=f"{pair['source']} -> {pair['reference']}", delta_deg=pair["delta_deg"],
               source_nodes=len(sp), source_faces=len(st), reference_faces=len(rt),
               centroid_mesh={k: v for k, v in info.items() if k not in ("ring_vertex", "failed_vertices")},
               uncovered_faces=int(uncovered.sum()), uncovered_where=where(uncovered),
               ring_proj_dist_median=float(np.median(dist)), ring_proj_dist_max=float(dist.max()),
               ring_changed_without_normal_check=int(np.sum(idx_nn != idx)),
               ring_x_weight_range=[float(wx[uncovered].min()), float(wx[uncovered].max())] if uncovered.any() else None,
               near3_outside_faces=int(np.sum(np.min(w3, 1) < -1e-9)),
               near3_max_abs_weight=float(np.nanmax(np.abs(w3))),
               source=c_src, reference=c_ref, methods=rows)
    if arrays_dir is not None:
        arrays_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_dir / f"{pair['source']}.npz", sp=sp, st=st, src_cp=src["cp"], src_cell=src_cell,
                            cen=cen, ctri=ctri, ring_vertex=info["ring_vertex"], rp=rp, rt=rt, ref_cell=ref_cell,
                            q=q, qn=qn, idx=idx, w=w, wx=wx, idx_nn=idx_nn, uncovered=uncovered, sel3=sel3, w3=w3,
                            **{f"map_{k}": v for k, v in maps.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--arrays", type=Path, default=None, help="図に使う配列の保存先（Git の管理外に置く）")
    a = ap.parse_args()
    plan = json.loads((a.run / "plan.json").read_text())
    res = dict(run=a.run.name, feature_angle_deg=FEATURE_ANGLE_DEG, normal_angle_limit_deg=60.0,
               pairs=[run_pair(a.run, plan, p, a.arrays) for p in plan["pairs"]])
    json.dump(res, sys.stdout, ensure_ascii=False, indent=1)
    print()


if __name__ == "__main__":
    main()
