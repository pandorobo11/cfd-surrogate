"""SU2 の力の内訳の摩擦分を、restart と格子から再現し、接線分と法線分に分ける。

2026-11-17 の記録の friction_check.py を、プリズム（種類 13）を含む格子も読めるようにしたもの。

壁面節点の速度勾配を SU2 と同じ重み付き最小二乗（重み 1/|d|^2、辺でつながる点）で作り、
粘性応力 tau から tau·n を求める。n は、まわりの三角形の面積ベクトルの 1/3 の和を正規化したもの。

求める値（格子の座標系、基準面積と動圧で割った係数）:
  su2_breakdown     SU2 の forces_breakdown.dat の摩擦分
  tangent_nodal     接線成分 × 節点面積 |Σ a_t/3|（SU2 の求積で、法線成分を除いたもの）
  normal_nodal      法線成分 tau_nn n × 節点面積
  total_nodal       上の 2 つの和（SU2 の内訳を再現するはず）
  tangent_triangle  出力の Cf を三角形ごとに 3 頂点で平均し、面積を掛けた和（finremap の積分）

使い方（リポジトリの直下で）:
  python3 docs/reports/20261119-fin-cf-remap-prism/friction_check.py <ケースのフォルダ>... > friction_check.json
ケースのフォルダには case.cfg、mesh.su2、restart_flow.dat（バイナリ）、surface_flow.vtu、forces_breakdown.dat が要る。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from finremap.vtu import read_vtu  # noqa: E402

WALL = "fin"


def read_cfg(path):
    cfg = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("%"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def read_restart(path):
    """SU2 のバイナリ restart。先頭に int32 が 5 個、変数名が 33 バイトずつ、続いて点ごとの値。"""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(20), dtype=np.int32)
        nvar, npt = int(hdr[1]), int(hdr[2])
        names = [f.read(33).split(b"\0")[0].decode() for _ in range(nvar)]
        data = np.frombuffer(f.read(8 * nvar * npt), dtype=np.float64).reshape(npt, nvar)
    return dict(zip(names, data.T))


# 要素の辺（SU2 と同じ: 四面体は 6 本、プリズムは底面・上面の 3 本ずつと縦の 3 本）
EDGES = {10: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
         13: [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]}
NODES = {10: 4, 13: 6}


def read_mesh(path):
    """要素の辺（節点の組 (E, 2)、重複あり）と、翼面の三角形を返す。四面体とプリズムを読む。"""
    lines = Path(path).read_text().split("\n")
    nelem = int(lines[1].split("=")[1])
    by_type = {}
    for l in lines[2:2 + nelem]:
        z = l.split()
        by_type.setdefault(int(z[0]), []).append(z[1:1 + NODES.get(int(z[0]), 0)])
    if set(by_type) - set(EDGES):
        raise ValueError(f"読めない要素の種類がある: {sorted(set(by_type) - set(EDGES))}")
    pairs = []
    for t, rows in by_type.items():
        conn = np.array(rows, dtype=np.int64)
        pairs += [conn[:, [a, b]] for a, b in EDGES[t]]
    i = next(k for k, l in enumerate(lines) if l.strip() == f"MARKER_TAG= {WALL}")
    nm = int(lines[i + 1].split("=")[1])
    tri = np.array([l.split()[1:4] for l in lines[i + 2:i + 2 + nm]], dtype=np.int64)
    return np.concatenate(pairs), tri


def su2_friction(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"Total (CF[xyz]|CM[xyz]):.*Friction \([^)]*\):\s*(\S+)", line)
        if m and m.group(1) not in out:
            out[m.group(1)] = float(m.group(2))
    return out


def coeffs(xyz_nodes, f_nodes, origin):
    """節点ごとの力 (N, 3) から CFx, CFz, CMy。"""
    F = f_nodes.sum(axis=0)
    M = np.cross(xyz_nodes - origin, f_nodes).sum(axis=0)
    return {"CFx": F[0], "CFy": F[1], "CFz": F[2], "CMy": M[1]}


def check(case):
    case = Path(case)
    cfg = read_cfg(case / "case.cfg")
    gamma, R = float(cfg["GAMMA_VALUE"]), float(cfg["GAS_CONSTANT"])
    p, T, mach = float(cfg["FREESTREAM_PRESSURE"]), float(cfg["FREESTREAM_TEMPERATURE"]), float(cfg["MACH_NUMBER"])
    mu, aref, lref = float(cfg["MU_CONSTANT"]), float(cfg["REF_AREA"]), float(cfg["REF_LENGTH"])
    origin = np.array([float(cfg[f"REF_ORIGIN_MOMENT_{c}"]) for c in "XYZ"])
    rho = p / (R * T)
    q = 0.5 * rho * mach**2 * gamma * R * T
    scale = 1.0 / (q * aref)

    sol = read_restart(case / "restart_flow.dat")
    xyz = np.stack([sol["x"], sol["y"], sol["z"]], axis=1)
    vel = np.stack([sol[f"Momentum_{c}"] for c in "xyz"], axis=1) / sol["Density"][:, None]
    edge_pairs, wtri = read_mesh(case / "mesh.su2")
    wall = np.unique(wtri)
    npt = len(xyz)

    # 壁面節点に接する辺（両向き）
    iswall = np.zeros(npt, bool)
    iswall[wall] = True
    pairs = edge_pairs[iswall[edge_pairs[:, 0]] | iswall[edge_pairs[:, 1]]]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    e = np.concatenate([pairs[iswall[pairs[:, 0]]], pairs[iswall[pairs[:, 1]]][:, ::-1]])
    k = np.searchsorted(wall, e[:, 0])
    d = xyz[e[:, 1]] - xyz[e[:, 0]]
    w = 1.0 / np.einsum("ij,ij->i", d, d)
    du = vel[e[:, 1]] - vel[e[:, 0]]
    S = np.zeros((len(wall), 3, 3))
    B = np.zeros((len(wall), 3, 3))
    np.add.at(S, k, w[:, None, None] * d[:, :, None] * d[:, None, :])
    np.add.at(B, k, w[:, None, None] * d[:, :, None] * du[:, None, :])
    G = np.linalg.solve(S, B).transpose(0, 2, 1)  # G[n, i, j] = du_i/dx_j

    # 節点面積: SU2 は |Σ a_t/3|、三角形の積分に相当するのは Σ |a_t|/3
    pt = xyz[wtri]
    a = 0.5 * np.cross(pt[:, 1] - pt[:, 0], pt[:, 2] - pt[:, 0])
    A = np.zeros((npt, 3))
    np.add.at(A, wtri.ravel(), np.repeat(a / 3, 3, axis=0))
    A = A[wall]
    area = np.linalg.norm(A, axis=1)
    asum = np.zeros(npt)
    np.add.at(asum, wtri.ravel(), np.repeat(np.linalg.norm(a, axis=1) / 3, 3))
    asum = asum[wall]
    n = A / area[:, None]

    div = np.trace(G, axis1=1, axis2=2)
    tau = mu * (G + G.transpose(0, 2, 1)) - (2.0 / 3.0) * mu * div[:, None, None] * np.eye(3)
    te = np.einsum("nij,nj->ni", tau, n)
    tn = np.einsum("ni,ni->n", te, n)
    tt = te - tn[:, None] * n

    # 出力の Cf（物体が受ける向き）と向きをそろえる。n の向きが逆なら te と tt の符号が変わる
    srf = read_vtu(case / "surface_flow.vtu")
    dist, j = cKDTree(srf.points).query(xyz[wall])
    cf_out = srf.point_data["Skin_Friction_Coefficient"].astype(float)[j]
    s = np.sign(np.einsum("ni,ni->", tt / q, cf_out))
    te, tt = s * te, s * tt
    rel = np.sqrt(((tt / q - cf_out) ** 2).sum(1).mean() / (cf_out**2).sum(1).mean())

    xw = xyz[wall]
    f_tan = tt * area[:, None] * scale
    f_nrm = (te - tt) * area[:, None] * scale
    # 三角形ごとの積分（出力の Cf をそのまま使う）
    cf_mesh = np.full((npt, 3), np.nan)
    cf_mesh[wall] = cf_out
    ftri = cf_mesh[wtri].mean(axis=1) * np.linalg.norm(a, axis=1)[:, None] / aref
    ctri = coeffs(pt.mean(axis=1), ftri, origin)

    r6 = lambda dct: {kk: round(float(v), 6) for kk, v in dct.items()}
    return {
        "case": case.name,
        "wall_nodes": int(len(wall)),
        "max_vtu_match_distance": float(dist.max()),
        "max_wall_speed": float(np.abs(vel[wall]).max()),
        "cf_recomputed_vs_output_rel_rms": round(float(rel), 5),
        "tau_nn_over_tau_t_rms": round(float(np.sqrt((tn**2).mean() / (np.linalg.norm(tt, axis=1) ** 2).mean())), 4),
        "nodal_area_ratio_min": round(float((area / asum).min()), 4),
        "su2_breakdown": r6(su2_friction(case / "forces_breakdown.dat")),
        "tangent_nodal": r6(coeffs(xw, f_tan, origin)),
        "normal_nodal": r6(coeffs(xw, f_nrm, origin)),
        "total_nodal": r6(coeffs(xw, f_tan + f_nrm, origin)),
        "tangent_nodal_area_sum": r6(coeffs(xw, tt * asum[:, None] * scale, origin)),
        "tangent_triangle": r6(ctri),
    }


if __name__ == "__main__":
    json.dump({"cases": [check(c) for c in sys.argv[1:]]}, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
