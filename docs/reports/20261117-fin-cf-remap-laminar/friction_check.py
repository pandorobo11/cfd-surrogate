"""SU2 の力の内訳の摩擦分を、restart と格子から再現し、接線分と法線分に分ける。

壁面節点の速度勾配を SU2 と同じ重み付き最小二乗（重み 1/|d|^2、辺でつながる点）で作り、
粘性応力 tau から tau·n を求める。n は、まわりの三角形の面積ベクトルの 1/3 の和を正規化したもの。

求める値（格子の座標系、基準面積と動圧で割った係数）:
  su2_breakdown     SU2 の forces_breakdown.dat の摩擦分
  tangent_nodal     接線成分 × 節点面積 |Σ a_t/3|（SU2 の求積で、法線成分を除いたもの）
  normal_nodal      法線成分 tau_nn n × 節点面積
  total_nodal       上の 2 つの和（SU2 の内訳を再現するはず）
  tangent_triangle  出力の Cf を三角形ごとに 3 頂点で平均し、面積を掛けた和（finremap の積分）

使い方（リポジトリの直下で）:
  python3 docs/reports/20261117-fin-cf-remap-laminar/friction_check.py <ケースのフォルダ>... > friction_check.json
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


def read_mesh(path):
    lines = Path(path).read_text().split("\n")
    nelem = int(lines[1].split("=")[1])
    elem = [l.split() for l in lines[2:2 + nelem]]
    if any(e[0] != "10" for e in elem):
        raise ValueError("四面体（種類 10）以外の要素がある")
    tet = np.array([e[1:5] for e in elem], dtype=np.int64)
    i = next(k for k, l in enumerate(lines) if l.strip() == f"MARKER_TAG= {WALL}")
    nm = int(lines[i + 1].split("=")[1])
    tri = np.array([l.split()[1:4] for l in lines[i + 2:i + 2 + nm]], dtype=np.int64)
    return tet, tri


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
    tet, wtri = read_mesh(case / "mesh.su2")
    wall = np.unique(wtri)
    npt = len(xyz)

    # 壁面節点に接する辺（両向き）
    iswall = np.zeros(npt, bool)
    iswall[wall] = True
    pairs = np.unique(np.sort(np.concatenate([tet[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), axis=1), axis=0)
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
