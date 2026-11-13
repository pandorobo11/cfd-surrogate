"""SU2 の Euler 解析（前後縁を丸めた全遊動翼単体, Mach 2）の表面 Cp を、舵角を戻して基準格子へ補間する。

手順:
  1. 舵角 delta の格子の表面節点を、舵軸まわりに -delta 回して翼固定座標に戻す。
  2. 基準格子（舵角 0）の各節点から、元格子の三角形面上の最近接点を求める。
     裏面の三角形（法線が逆向き）は候補から外す。
  3. 最近接点の重心座標で、三角形の 3 頂点の Cp を平均する。

検証:
  翼単体では「舵角 delta, 迎角 0」と「舵角 0, 迎角 delta」は同じ流れになる。
  そこで、補間した Cp を、基準格子で迎角 delta を解いた結果と比べる。
  補間そのものの質は、補間前後の力とモーメントの変化で見る。

実行: python apply_su2.py --results <SU2 の results/fin-remap-20261114>   （numpy, scipy, matplotlib を使用）
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

HERE = Path(__file__).parent
SPAN, S_REF, HINGE_X = 0.8, 0.6, 0.45  # 翼幅、基準面積、舵軸の x 位置（舵軸は y 軸に平行）
XI_NOSE = 0.03  # 丸めた前縁・後縁のまわりとして別に集計する範囲（翼弦比）
PAIRS = [("r4-round2-d05-a00", "r4-round2-ref-a05", 5.0), ("r4-round2-d10-a00", "r4-round2-ref-a10", 10.0)]


# ---------- 幾何と補間 ----------
def rotate_about_hinge(xyz, deg):
    """舵軸（y 軸に平行）まわりの回転。"""
    a = np.radians(deg)
    c, sn = np.cos(a), np.sin(a)
    out = xyz.copy()
    dx, dz = xyz[:, 0] - HINGE_X, xyz[:, 2]
    out[:, 0] = HINGE_X + c * dx + sn * dz
    out[:, 2] = -sn * dx + c * dz
    return out


def face_normals_areas(xyz, tri):
    n = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / (2.0 * area[:, None]), area


def node_normals(xyz, tri):
    fn, area = face_normals_areas(xyz, tri)
    nn = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(nn, tri[:, k], fn * area[:, None])
    return nn / np.linalg.norm(nn, axis=1, keepdims=True)


def closest_point_barycentric(p, a, b, c):
    """点 p から三角形 abc 上の最近接点の重心座標 (wa, wb, wc)。Ericson の領域分けによる。"""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.sum(ab * ap, -1), np.sum(ac * ap, -1)
    bp = p - b
    d3, d4 = np.sum(ab * bp, -1), np.sum(ac * bp, -1)
    cp_ = p - c
    d5, d6 = np.sum(ab * cp_, -1), np.sum(ac * cp_, -1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        v, w = vb / den, vc / den  # 三角形の内側
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    wb, wc = v.copy(), w.copy()
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)  # 辺 AB
    wb[m], wc[m] = t_ab[m], 0.0
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)  # 辺 AC
    wb[m], wc[m] = 0.0, t_ac[m]
    m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)  # 辺 BC
    wb[m], wc[m] = 1.0 - t_bc[m], t_bc[m]
    m = (d1 <= 0) & (d2 <= 0)  # 頂点 A
    wb[m], wc[m] = 0.0, 0.0
    m = (d3 >= 0) & (d4 <= d3)  # 頂点 B
    wb[m], wc[m] = 1.0, 0.0
    m = (d6 >= 0) & (d5 <= d6)  # 頂点 C
    wb[m], wc[m] = 0.0, 1.0
    return np.stack([1.0 - wb - wc, wb, wc], axis=-1)


def map_nearest(src_xyz, src_cp, q):
    return src_cp[cKDTree(src_xyz).query(q)[1]]


def map_barycentric(src_xyz, src_tri, src_cp, q, q_normal=None, k=24):
    """最近接点を三角形面上に投影し、重心座標で補間する。q_normal を渡すと裏面の三角形を除外する。"""
    centroids = src_xyz[src_tri].mean(axis=1)
    cand = cKDTree(centroids).query(q, k=k)[1]  # (N, k) 候補三角形
    v = src_xyz[src_tri[cand]]  # (N, k, 3, 3)
    w = closest_point_barycentric(q[:, None, :], v[:, :, 0], v[:, :, 1], v[:, :, 2])
    closest = np.einsum("nkj,nkjd->nkd", w, v)
    dist = np.linalg.norm(closest - q[:, None, :], axis=-1)
    if q_normal is not None:
        fn, _ = face_normals_areas(src_xyz, src_tri)
        back = np.einsum("nkd,nd->nk", fn[cand], q_normal) < 0.0
        dist = np.where(back, np.inf, dist)
    best = np.argmin(dist, axis=1)
    rows = np.arange(len(q))
    cp = np.sum(w[rows, best] * src_cp[src_tri[cand[rows, best]]], axis=1)
    return cp, dist[rows, best]


# ---------- SU2 出力の読み込みと力の積分 ----------
def read_marker_triangles(mesh_path, tag="fin"):
    tris, grab, left = [], False, 0
    with open(mesh_path) as f:
        for line in f:
            if line.startswith("MARKER_TAG="):
                grab = line.split("=")[1].strip() == tag
            elif line.startswith("MARKER_ELEMS=") and grab:
                left = int(line.split("=")[1])
            elif grab and left > 0:
                p = line.split()
                assert p[0] == "5", "三角形以外の境界要素"
                tris.append((int(p[1]), int(p[2]), int(p[3])))
                left -= 1
                grab = left > 0
    return np.array(tris)


def forces(xyz, tri, cp):
    """Cp の面積分。(CFx, CFz, CMy)。モーメントは舵軸まわり。"""
    n, area = face_normals_areas(xyz, tri)
    f = -cp[tri].mean(axis=1)[:, None] * n * area[:, None] / S_REF
    r = xyz[tri].mean(axis=1) - np.array([HINGE_X, 0.0, 0.0])
    my = r[:, 2] * f[:, 0] - r[:, 0] * f[:, 2]
    return float(f[:, 0].sum()), float(f[:, 2].sum()), float(my.sum())


def to_fin_axes(cfx, cfz, delta_deg):
    a = np.radians(delta_deg)
    return cfx * np.cos(a) - cfz * np.sin(a), cfx * np.sin(a) + cfz * np.cos(a)


def chord_coords(xyz):
    eta = xyz[:, 1] / SPAN
    return (xyz[:, 0] - 0.35 * eta) / (1.0 - 0.5 * eta), eta


def load_case(case_dir, delta_deg):
    raw = np.genfromtxt(case_dir / "surface_flow.csv", delimiter=",", names=True)
    names = [n.strip('"') for n in raw.dtype.names]
    col = lambda k: raw[raw.dtype.names[names.index(k)]]
    pid = col("PointID").astype(int)
    xyz = np.stack([col("x"), col("y"), col("z")], axis=1)
    tri_g = read_marker_triangles(case_dir / "mesh.su2")
    lut = -np.ones(tri_g.max() + 1, dtype=int)
    lut[pid] = np.arange(len(pid))
    tri = lut[tri_g]
    assert tri.min() >= 0
    fin_xyz = rotate_about_hinge(xyz, -delta_deg)
    # 向きを外向きにそろえる。翼端面は +y、それ以外は同じ翼幅位置の翼弦線上の内部点から離れる向き
    n, _ = face_normals_areas(fin_xyz, tri)
    cen = fin_xyz[tri].mean(axis=1)
    eta = cen[:, 1] / SPAN
    inner = np.stack([0.35 * eta + 0.4 * (1.0 - 0.5 * eta), cen[:, 1], np.zeros(len(cen))], axis=1)
    tip = np.all(fin_xyz[tri][:, :, 1] > SPAN - 1e-6, axis=1)
    flip = np.where(tip, n[:, 1] < 0, np.sum(n * (cen - inner), axis=1) < 0)
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return dict(xyz=xyz, fin_xyz=fin_xyz, tri=tri, cp=col("Pressure_Coefficient"),
                run=json.loads((case_dir / "run.json").read_text()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    args = ap.parse_args()
    summary = []
    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5))

    for src_id, ref_id, delta in PAIRS:
        src, ref = load_case(args.results / src_id, delta), load_case(args.results / ref_id, 0.0)
        q, qn = ref["xyz"], node_normals(ref["xyz"], ref["tri"])
        xi, eta = chord_coords(q)
        nose = ((xi < XI_NOSE) | (xi > 1.0 - XI_NOSE)) & (q[:, 1] < SPAN - 1e-6)
        fx, fz, my = forces(src["xyz"], src["tri"], src["cp"])
        h = src["run"]["final"]
        print(f"\n=== {src_id} → 基準格子（比較相手 {ref_id}）===")
        print(f"元格子 節点 {len(src['xyz'])}, 三角形 {len(src['tri'])} / 基準格子 節点 {len(q)}, 三角形 {len(ref['tri'])}"
              f"（うち丸めた縁のまわり {XI_NOSE*100:.0f}% 翼弦以内の節点 {int(nose.sum())}）")
        print(f"状態: {src_id} {src['run']['state']} (log10 rms[Rho] {h['rms_rho']:.2f}), "
              f"{ref_id} {ref['run']['state']} ({ref['run']['final']['rms_rho']:.2f})")
        print(f"元格子の積分確認  CFz: SU2 {h['CFz']:.5f} / 自前 {fz:.5f},  CMy: SU2 {h['CMy']:.5f} / 自前 {my:.5f}")

        maps = {"A 回転を戻さず最近傍節点": map_nearest(src["xyz"], src["cp"], q),
                "B 回転を戻して最近傍節点": map_nearest(src["fin_xyz"], src["cp"], q)}
        maps["C 投影＋重心座標"], _ = map_barycentric(src["fin_xyz"], src["tri"], src["cp"], q)
        maps["D C＋法線チェック"], dist = map_barycentric(src["fin_xyz"], src["tri"], src["cp"], q, qn)
        print(f"投影距離（翼弦長 1）: 中央値 {np.median(dist):.1e}, 最大 {dist.max():.1e}, 縁まわりの最大 {dist[nose].max():.1e}")

        ca_s, cn_s = to_fin_axes(fx, fz, delta)
        _, _, cm_s = forces(src["fin_xyz"], src["tri"], src["cp"])
        ca_r, cn_r, cm_r = forces(q, ref["tri"], ref["cp"])
        print(f"{'':<24}{'CN':>9}{'CA':>9}{'CMy(舵軸)':>11}{'Cp差RMS':>9}{'縁RMS':>9}{'縁最大':>9}{'他RMS':>9}{'他最大':>9}")
        print(f"{'元格子の解（補間前）':<24}{cn_s:9.5f}{ca_s:9.5f}{cm_s:11.5f}")
        rms = lambda v: float(np.sqrt(np.mean(v**2)))
        for name, cp in maps.items():
            ca, cn, cm = forces(q, ref["tri"], cp)
            d = cp - ref["cp"]
            print(f"{name:<24}{cn:9.5f}{ca:9.5f}{cm:11.5f}{rms(d):9.4f}{rms(d[nose]):9.4f}{np.abs(d[nose]).max():9.4f}"
                  f"{rms(d[~nose]):9.4f}{np.abs(d[~nose]).max():9.4f}")
            summary.append(dict(pair=f"{src_id}->{ref_id}", method=name, CN=cn, CA=ca, CMy=cm,
                                CN_change_pct=(cn / cn_s - 1) * 100, CA_change_pct=(ca / ca_s - 1) * 100,
                                CMy_change_pct=(cm / cm_s - 1) * 100, cp_diff_rms=rms(d),
                                cp_diff_rms_nose=rms(d[nose]), cp_diff_max_nose=float(np.abs(d[nose]).max()),
                                cp_diff_rms_rest=rms(d[~nose]), cp_diff_max_rest=float(np.abs(d[~nose]).max())))
        print(f"{'基準格子で迎角を付けた解':<24}{cn_r:9.5f}{ca_r:9.5f}{cm_r:11.5f}")
        summary.append(dict(pair=f"{src_id}->{ref_id}", method="source", CN=cn_s, CA=ca_s, CMy=cm_s,
                            src_nodes=len(src["xyz"]), ref_nodes=len(q), proj_dist_median=float(np.median(dist)),
                            proj_dist_max=float(dist.max())))
        summary.append(dict(pair=f"{src_id}->{ref_id}", method="reference_solution", CN=cn_r, CA=ca_r, CMy=cm_r))

        if delta != 10.0:
            continue

        def side(m, xyz, sign):
            n, _ = face_normals_areas(xyz, m["tri"])
            return Triangulation(xyz[:, 0], xyz[:, 1], m["tri"][sign * n[:, 2] > 0.3])

        mapped = maps["D C＋法線チェック"]
        tr_s, tr_r, tr_rl = side(src, src["fin_xyz"], 1), side(ref, ref["xyz"], 1), side(ref, ref["xyz"], -1)
        strip = np.abs(eta - 0.5) < 0.006
        ax = axes[0, 0]
        m = strip & (xi > 0.96)
        s_te = np.sign(q[m, 2]) * np.sqrt(np.clip(1.0 - xi[m], 0, None))  # 後縁を回り込む座標
        o = np.argsort(s_te)
        ax.plot(s_te[o], ref["cp"][m][o], "k-", lw=1.8, label="reference mesh, AoA 10")
        ax.plot(s_te[o], maps["B 回転を戻して最近傍節点"][m][o], "o", color="tab:orange", ms=4, mfc="none", label="nearest node B")
        ax.plot(s_te[o], mapped[m][o], "o", color="tab:green", ms=2.5, label="projection D")
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlabel("sign(z) * sqrt(1 - x/c)   (lower side < 0 < upper side)"); ax.set_ylabel("Cp")
        ax.set_title("(1) Around the rounded trailing edge (x/c > 0.96)")
        lim = dict(vmin=-0.25, vmax=0.05)
        for ax, tr, val, title in [
            (axes[0, 1], tr_s, src["cp"], "(2) SU2 Cp, 10 deg mesh (upper side)"),
            (axes[0, 2], tr_r, mapped, "(3) Mapped onto reference mesh (D)"),
            (axes[0, 3], tr_r, ref["cp"], "(4) Separate run: reference, AoA 10"),
        ]:
            im = ax.tripcolor(tr, val, shading="gouraud", cmap="viridis", **lim)
            fig.colorbar(im, ax=ax, label="Cp"); ax.set_title(title)
        for ax, tr, title in [(axes[1, 0], tr_r, "(5) Difference (3) - (4), upper side"),
                              (axes[1, 1], tr_rl, "(6) Same difference, lower side")]:
            im = ax.tripcolor(tr, mapped - ref["cp"], shading="gouraud", cmap="RdBu_r", vmin=-0.03, vmax=0.03)
            fig.colorbar(im, ax=ax, label="Cp difference"); ax.set_title(title)
        for ax in list(axes[0, 1:]) + [axes[1, 0], axes[1, 1]]:
            ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")

        # 翼幅中央の細い帯にある基準格子の節点で比較する
        up = strip & (q[:, 2] >= 0)
        lo = strip & (q[:, 2] < 0)
        ax = axes[1, 2]
        for m, mk, lab in [(up, "o", "upper"), (lo, "s", "lower")]:
            o = np.argsort(xi[m])
            ax.plot(xi[m][o], ref["cp"][m][o], "k-", lw=1.8, label=f"reference mesh, AoA 10 ({lab})")
            ax.plot(xi[m][o], maps["B 回転を戻して最近傍節点"][m][o], mk, color="tab:orange", ms=3, mfc="none",
                    label=f"10 deg mesh, nearest node B ({lab})")
            ax.plot(xi[m][o], mapped[m][o], mk, color="tab:green", ms=2, label=f"10 deg mesh, projection D ({lab})")
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_xlabel("chordwise position x/c"); ax.set_ylabel("Cp"); ax.set_ylim(0.6, -0.4)
        ax.set_title("(7) Reference-mesh nodes in a strip at mid span")

        ax = axes[1, 3]
        m = strip & (xi < 0.04)
        s = np.sign(q[m, 2]) * np.sqrt(np.clip(xi[m], 0, None))  # 前縁を回り込む座標（下面が負）
        o = np.argsort(s)
        ax.plot(s[o], ref["cp"][m][o], "k-", lw=1.8, label="reference mesh, AoA 10")
        ax.plot(s[o], maps["B 回転を戻して最近傍節点"][m][o], "o", color="tab:orange", ms=4, mfc="none", label="nearest node B")
        ax.plot(s[o], mapped[m][o], "o", color="tab:green", ms=2.5, label="projection D")
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlabel("sign(z) * sqrt(x/c)   (lower side < 0 < upper side)"); ax.set_ylabel("Cp")
        ax.set_title("(8) Around the rounded leading edge (x/c < 0.04)")

    fig.tight_layout()
    fig.savefig(HERE / "fin_remap_su2.png", dpi=130)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n図: {HERE / 'fin_remap_su2.png'}")


if __name__ == "__main__":
    main()
