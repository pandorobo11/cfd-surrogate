"""cell_remap.py が保存した配列から、各ステップの図を作る（舵角 10° の組）。

使い方（リポジトリの直下で）:
  python3 docs/reports/20261118-cell-centered-remap/figures.py results/finremap/cell-remap-20261118/d10-a00.npz \
      results/finremap/fin-round-m2-20261114/plan.json docs/reports/20261118-cell-centered-remap
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from finremap import remap  # noqa: E402
from finremap.mesh import half_thickness, planform, thickness_coeffs  # noqa: E402

GRAY, PURPLE, ORANGE, RED, BLUE = "#888780", "#534AB7", "#D85A30", "#A32D2D", "#185FA5"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})


def unwrap(geom, p):
    """前縁まわりを開いた座標 (翼幅 y, 前縁からの近似弧長 s)。s = sign(z) * a0 * c * sqrt(xi) で、前縁の近くでは z に等しい。"""
    x_le, c = planform(geom, p[:, 1])
    xi = np.clip((p[:, 0] - x_le) / c, 0, None)
    a0, _ = thickness_coeffs(geom)
    return np.stack([p[:, 1], np.sign(p[:, 2]) * a0 * c * np.sqrt(xi)], 1)


def polys(P2, tri):
    return P2[tri]


def draw_mesh(ax, P2, tri, **kw):
    ax.add_collection(PolyCollection(polys(P2, tri), facecolors="none", **kw))


def window(P2, box):
    (x0, x1), (y0, y1) = box
    return (P2[:, 0] > x0) & (P2[:, 0] < x1) & (P2[:, 1] > y0) & (P2[:, 1] < y1)


def tri_in(P2, tri, box, pad=0.0):
    (x0, x1), (y0, y1) = box
    c = P2[tri].mean(1)
    return (c[:, 0] > x0 - pad) & (c[:, 0] < x1 + pad) & (c[:, 1] > y0 - pad) & (c[:, 1] < y1 + pad)


def setbox(ax, box, xlabel="span y", ylabel="arc length from LE s (upper +)"):
    ax.set_xlim(*box[0])
    ax.set_ylim(*box[1])
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main(npz, plan_path, out):
    d = dict(np.load(npz))
    geom = json.loads(Path(plan_path).read_text())["geometry"]
    out = Path(out)
    sp, st, cen, ctri = d["sp"], d["st"], d["cen"], d["ctri"]
    rp, rt, q = d["rp"], d["rt"], d["q"]
    src_cell, ref_cell = d["src_cell"], d["ref_cell"]
    S2, C2, Q2 = unwrap(geom, sp), unwrap(geom, cen), unwrap(geom, q)
    near_le = ((sp[:, 0] - planform(geom, sp[:, 1])[0]) / planform(geom, sp[:, 1])[1]) < 0.05
    box = ((0.394, 0.406), (-0.0055, 0.0055))
    le_tri = near_le[st].all(1) & tri_in(S2, st, box, 0.002)
    le_ct = le_tri[ctri].all(1)
    vmin, vmax = np.percentile(src_cell[le_tri], [1, 99])

    # --- ステップ 1: 節点値からセル値へ
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    ax = axes[0]
    tp = ax.tripcolor(S2[:, 0], S2[:, 1], st[le_tri], d["src_cp"], shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    draw_mesh(ax, S2, st[le_tri], edgecolors="w", linewidths=0.4)
    m = window(S2, box) & near_le
    ax.plot(S2[m, 0], S2[m, 1], "o", ms=2.5, color="k")
    setbox(ax, box)
    ax.set_title("(a) SU2 output: Cp at nodes (black dots), linear in each triangle")
    ax = axes[1]
    ax.add_collection(PolyCollection(polys(S2, st[le_tri]), array=src_cell[le_tri], cmap="viridis", clim=(vmin, vmax),
                                     edgecolors="w", linewidths=0.4))
    m = window(C2, box) & le_tri
    ax.plot(C2[m, 0], C2[m, 1], "o", ms=2.5, color=PURPLE)
    setbox(ax, box)
    ax.set_title("(b) cell values: mean of the 3 nodes, held at the centroid (purple)")
    fig.colorbar(tp, ax=axes, shrink=0.8, label="Cp")
    fig.suptitle("Step 1  Node values to cell values (d10 mesh rotated back, leading edge near mid-span, unwrapped)")
    fig.savefig(out / "step1-cell-values.png", dpi=150)
    plt.close(fig)

    # --- ステップ 2: 節点の環
    ctr = np.array([0.4, 0.0012])
    cand = np.nonzero(window(S2, ((ctr[0] - 0.0015, ctr[0] + 0.0015), (ctr[1] - 0.0008, ctr[1] + 0.0008))) & near_le)[0]
    rv = d["ring_vertex"]
    cand = [v for v in cand if np.sum(rv == v) >= 4]
    v0 = int(cand[np.argmin(np.linalg.norm(S2[cand] - ctr, axis=1))])
    faces = np.nonzero((st == v0).any(1))[0]
    fn, area = remap.face_normals_areas(sp, st)
    n = np.sum(fn[faces] * area[faces, None], 0)
    n /= np.linalg.norm(n)
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    to2 = lambda X: np.stack([(X - sp[v0]) @ e1, (X - sp[v0]) @ e2], -1) * 1e3  # mm 相当（基準長 1 の 1/1000）
    # 環の順（辺でたどる）: 三角形 (v0, a, b) の次は辺 (v0, b) の向こう
    order, f = [], faces[0]
    for _ in range(len(faces)):
        order.append(f)
        t = list(st[f])
        b = t[(t.index(v0) + 2) % 3]
        f = [g for g in faces if g != f and b in st[g]][0]
    ring_tris = ctri[rv == v0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    for k, ax in enumerate(axes[:2]):
        for i, f in enumerate(order):
            P = to2(sp[st[f]])
            ax.add_patch(plt.Polygon(P, fc="#FAEEDA" if k == 0 else "none", ec=GRAY, lw=0.8))
            c = to2(cen[f])
            if k == 0:
                ax.annotate(f"{i + 1}", c, xytext=(4, 4), textcoords="offset points", color=PURPLE, fontsize=10)
        R = to2(cen[order])
        ax.plot(*np.vstack([R, R[:1]]).T, "-", color=PURPLE, lw=2.0)
        ax.plot(*R.T, "o", color=PURPLE, ms=6)
        ax.plot(0, 0, "s", color="#BA7517", ms=7)
        ax.set_aspect("equal")
        ax.set_xlabel("tangent-plane coordinate [x 1e-3]")
    axes[0].set_title(f"(a) faces sharing node v (square), walked across shared edges;\n"
                      f"centroids joined in that order give the ring ({len(order)} corners)")
    loc = {f: i for i, f in enumerate(order)}
    cm = plt.get_cmap("Pastel1")
    for i, t in enumerate(ring_tris):
        P = to2(cen[t])
        mins = []
        for j in range(3):
            u, w_ = P[(j + 1) % 3] - P[j], P[(j + 2) % 3] - P[j]
            mins.append(np.degrees(np.arccos(np.dot(u, w_) / np.linalg.norm(u) / np.linalg.norm(w_))))
        axes[1].add_patch(plt.Polygon(P, fc=cm(i % 9), ec=PURPLE, lw=1.2, alpha=0.9))
        axes[1].annotate(f"cut {i + 1}\nmin {min(mins):.0f}°", P.mean(0), ha="center", va="center", fontsize=8)
    axes[1].set_title(f"(b) ear clipping: {len(ring_tris)} triangles; each step cuts the ear\n"
                      "with the largest minimum angle (numbers = cut order)")
    ax = axes[2]
    zb = ((ctr[0] - 0.0035, ctr[0] + 0.0035), (ctr[1] - 0.0028, ctr[1] + 0.0028))
    near_v = np.unique(st[tri_in(S2, st, zb, 0.0)])
    near_v = [v for v in near_v if near_le[v]]
    sel = np.isin(rv, near_v) & le_ct
    cols = plt.get_cmap("tab20")
    for i, v in enumerate(np.unique(rv[sel])):
        T = ctri[(rv == v) & sel]
        ax.add_collection(PolyCollection(C2[T], facecolors=cols(i % 20), edgecolors="w", linewidths=0.6, alpha=0.85))
    draw_mesh(ax, S2, st[le_tri], edgecolors=GRAY, linewidths=0.5)
    m = window(C2, zb) & le_tri
    ax.plot(C2[m, 0], C2[m, 1], "o", ms=2.5, color=PURPLE)
    ax.plot(*S2[v0], "s", color="#BA7517", ms=6)
    setbox(ax, zb)
    ax.set_title("(c) rings of neighbouring nodes (one colour per node)\ntile the surface without gaps or overlaps")
    fig.suptitle("Step 2  Ring of a node: order the faces around it, join their centroids, split into triangles")
    fig.savefig(out / "step2-vertex-ring.png", dpi=150)
    plt.close(fig)

    # --- ステップ 3: 特徴線と境界
    e = np.sort(np.concatenate([st[:, [0, 1]], st[:, [1, 2]], st[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(len(st)), 3)
    o = np.lexsort((e[:, 1], e[:, 0]))
    e, fid = e[o], fid[o]
    same = np.all(e[1:] == e[:-1], axis=1)
    pair_e = e[:-1][same]
    ang = np.degrees(np.arccos(np.clip(np.sum(fn[fid[:-1][same]] * fn[fid[1:][same]], 1), -1, 1)))
    key = e[:, 0] * len(sp) + e[:, 1]
    uk, cnt = np.unique(key, return_counts=True)
    bnd = np.stack([uk[cnt == 1] // len(sp), uk[cnt == 1] % len(sp)], 1)
    feat = pair_e[ang > 45.0]
    fig = plt.figure(figsize=(16, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.9, 1.3])
    ax = fig.add_subplot(gs[0])
    upper = fn[:, 2] > 0.3
    ax.add_collection(PolyCollection(sp[st[upper]][:, :, :2], facecolors="#F1EFE8", edgecolors="none"))
    ax.add_collection(LineCollection(sp[feat][:, :, :2], colors=RED, linewidths=1.5, label=f"feature edges ({len(feat)})"))
    ax.add_collection(LineCollection(sp[bnd][:, :, :2], colors=BLUE, linewidths=2.5, label=f"boundary edges ({len(bnd)})"))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right")
    ax.set_title("(a) where rings are cut: tip edges (dihedral > 45°)\nand the root on the symmetry plane")
    ax = fig.add_subplot(gs[1])
    ax.hist(ang, bins=np.arange(0, 92, 2), color=GRAY, log=True)
    ax.axvline(45, color=RED, ls="--")
    ax.text(46, ax.get_ylim()[1] * 0.3, "threshold 45°", color=RED)
    ax.set_xlabel("angle between normals of the two faces of an edge [deg]")
    ax.set_ylabel("number of edges")
    ax.set_title(f"(b) dihedral angles: none between {np.max(ang[ang < 45]):.0f}° and {np.min(ang[ang > 45]):.0f}°,\n"
                 "so any threshold in that gap gives the same cuts")
    # (c) 翼端の上面側の縁を開いて見る: 側面は y - span（負）、翼端面は縁からの距離（正）
    ax = fig.add_subplot(gs[2])
    span = geom["span"]
    x_le, c_tip = planform(geom, np.array([span]))
    xc = float(x_le[0] + 0.3 * c_tip[0])

    def z_up(x):
        return half_thickness(geom, np.clip((x - x_le[0]) / c_tip[0], 0, 1)) * c_tip[0]

    def unf(p):
        tipf = p[:, 1] > span - 1e-6
        return np.stack([p[:, 0] - xc, np.where(tipf, z_up(p[:, 0]) - p[:, 2], p[:, 1] - span)], 1)

    tipface = np.all(sp[st][:, :, 1] > span - 1e-6, 1)
    side_up = (fn[:, 2] > 0.3) & ~tipface
    near = (np.abs(sp[st].mean(1)[:, 0] - xc) < 0.045) & (np.abs(sp[st].mean(1)[:, 1] - span) < 0.04)
    T_side, T_tip = st[near & side_up], st[near & tipface & (sp[st].mean(1)[:, 2] > 0)]
    U = unf(sp)
    for T, col in [(T_side, "#E6F1FB"), (T_tip, "#FAECE7")]:
        ax.add_collection(PolyCollection(U[T], facecolors=col, edgecolors=GRAY, linewidths=0.5))
    Uc = unf(cen)
    f_side, f_tip = np.nonzero(near & side_up)[0], np.nonzero(near & tipface & (sp[st].mean(1)[:, 2] > 0))[0]
    for F, col in [(f_side, BLUE), (f_tip, ORANGE)]:
        T = ctri[np.isin(ctri, F).all(1)]
        ax.add_collection(PolyCollection(Uc[T], facecolors="none", edgecolors=col, linewidths=1.0))
        ax.plot(Uc[F, 0], Uc[F, 1], "o", ms=3, color=col)
    ax.axhline(0, color=RED, lw=1.5)
    qtip = np.all(rp[rt][:, :, 1] > span - 1e-6, 1)
    unc = d["uncovered"] & (np.abs(q[:, 0] - xc) < 0.04) & (np.abs(q[:, 1] - span) < 0.03) & (q[:, 2] > 0)
    Uq = unf(q)
    ax.plot(Uq[unc, 0], Uq[unc, 1], "x", color="k", ms=6, mew=1.5, label="target centroids in the uncovered band")
    ax.set_xlim(-0.035, 0.035)
    ax.set_ylim(-0.022, 0.016)
    ax.set_aspect("equal")
    ax.set_xlabel("x - x(30% chord of tip)")
    ax.set_ylabel("side: y - span   |   tip face: distance from the edge")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("(c) upper tip edge (red line) unfolded flat: the side (blue) and\n"
                 "the tip face (orange) get separate centroid triangles")
    del qtip
    fig.suptitle("Step 3  Cut the rings at feature edges and boundaries")
    fig.savefig(out / "step3-feature-edges.png", dpi=150)
    plt.close(fig)

    # --- ステップ 4: できたセル中心の三角形格子と、覆われない帯
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[0])
    box4 = ((0.3975, 0.4025), (-0.0022, 0.0022))
    draw_mesh(ax, S2, st[le_tri], edgecolors=GRAY, linewidths=0.5)
    ax.add_collection(PolyCollection(C2[ctri[le_ct]], facecolors="none", edgecolors=PURPLE, linewidths=0.6))
    m = window(C2, box4) & le_tri
    ax.plot(C2[m, 0], C2[m, 1], "o", ms=3, color=PURPLE)
    setbox(ax, box4)
    ax.set_title(f"(a) source mesh (gray) and centroid mesh (purple)\n"
                 f"{len(st):,} faces -> {len(ctri):,} centroid triangles")
    ax = fig.add_subplot(gs[1])
    tip_r = np.all(rp[rt][:, :, 1] > span - 1e-6, 1)
    un = d["uncovered"]
    rn, _ = remap.face_normals_areas(rp, rt)
    for msk, col, lab in [((rn[:, 2] > 0) & ~tip_r, "#F1EFE8", None)]:
        ax.add_collection(PolyCollection(rp[rt[msk]][:, :, :2], facecolors=col, edgecolors="none"))
    for msk, col, lab in [(un & (q[:, 2] >= 0) & ~tip_r, RED, "upper side"), (un & (q[:, 2] < 0) & ~tip_r, BLUE, "lower side"),
                          (un & tip_r, ORANGE, "tip face")]:
        ax.plot(q[msk, 0], q[msk, 1], ".", color=col, ms=5, label=f"{lab} ({msk.sum()})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.04, 0.84)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="center", title=f"uncovered target faces: {un.sum()} of {len(rt):,}")
    ax.set_title("(b) target faces whose centroid falls in the uncovered band\n(only along the root and the tip edges)")
    fig.suptitle("Step 4  The centroid mesh is an ordinary triangle surface mesh with values at its vertices")
    fig.savefig(out / "step4-centroid-mesh.png", dpi=150)
    plt.close(fig)

    # --- ステップ 5: 投影と重心座標
    idx, w = d["idx"], d["w"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    ax = axes[0]
    zb = ((0.3975, 0.4025), (-0.0012, 0.0028))
    ax.add_collection(PolyCollection(C2[ctri[le_ct]], facecolors="none", edgecolors=PURPLE, linewidths=0.6))
    m = window(C2, zb) & le_tri
    ax.plot(C2[m, 0], C2[m, 1], "o", ms=3, color=PURPLE, label="source centroids (values)")
    mq = window(Q2, zb) & (((q[:, 0] - planform(geom, q[:, 1])[0]) / planform(geom, q[:, 1])[1]) < 0.05)
    ax.plot(Q2[mq, 0], Q2[mq, 1], "^", ms=4, color=ORANGE, label="target centroids (reference mesh)")
    j = np.nonzero(mq)[0]
    j = int(j[np.argmin(np.linalg.norm(Q2[j] - [0.4, 0.0008], axis=1))])
    T = ctri[idx[j]]
    ax.add_patch(plt.Polygon(C2[T], fc=ORANGE, alpha=0.2, ec=ORANGE, lw=1.5))
    for k in range(3):
        ax.plot(*np.stack([Q2[j], C2[T[k]]]).T, "-", color=ORANGE, lw=0.8 + 4 * w[j, k])
        off = (C2[T[k]] - C2[T].mean(0))
        off = 28 * off / np.linalg.norm(off)
        ax.annotate(f"w={w[j, k]:.2f}\nCp={src_cell[T[k]]:.3f}", C2[T[k]], xytext=(off[0] - 14, off[1] - 6),
                    textcoords="offset points", fontsize=8, color=PURPLE, bbox=dict(fc="w", ec="none", alpha=0.8))
    val = float(np.dot(w[j], src_cell[T]))
    ax.plot(*Q2[j], "^", ms=9, color=ORANGE, mec="k")
    ax.annotate(f"interpolated {val:.3f}\nreference {ref_cell[j]:.3f}", Q2[j], xytext=(-170, -70), textcoords="offset points",
                fontsize=9, bbox=dict(fc="w", ec="k", lw=0.5), arrowprops=dict(arrowstyle="-", color="k", lw=0.5))
    setbox(ax, zb)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("(a) each target centroid is projected onto the centroid mesh;\n"
                 "the barycentric weights of the foot point average the 3 cell values")
    # (b) 後縁付近の断面: 投影の線と法線
    ax = axes[1]
    y0 = 0.4
    x_le0, c0 = planform(geom, np.array([y0]))

    def section(P, T):
        a, b, c_ = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        segs = []
        for u, v in [(a, b), (b, c_), (c_, a)]:
            s = (y0 - u[:, 1]) / np.where(v[:, 1] != u[:, 1], v[:, 1] - u[:, 1], np.nan)
            segs.append(np.where(((s >= 0) & (s <= 1))[:, None], u + s[:, None] * (v - u), np.nan))
        segs = np.stack(segs, 1)
        out_ = []
        for sgm in segs:
            ok = np.isfinite(sgm[:, 0])
            if ok.sum() >= 2:
                out_.append(sgm[ok][:2][:, [0, 2]])
        return out_

    te = (sp[st].mean(1)[:, 0] - x_le0[0]) / c0[0] > 0.985
    crossing = (sp[st][:, :, 1].min(1) <= y0) & (sp[st][:, :, 1].max(1) >= y0)
    ax.add_collection(LineCollection(section(sp, st[te & crossing]), colors=GRAY, linewidths=2.0, label="source surface"))
    tc = (cen[ctri].mean(1)[:, 0] - x_le0[0]) / c0[0] > 0.985
    crossing_c = (cen[ctri][:, :, 1].min(1) <= y0) & (cen[ctri][:, :, 1].max(1) >= y0)
    ax.add_collection(LineCollection(section(cen, ctri[tc & crossing_c]), colors=PURPLE, linewidths=1.2, linestyles="--",
                                     label="centroid mesh"))
    sel_q = (np.abs(q[:, 1] - y0) < 0.0006) & ((q[:, 0] - x_le0[0]) / c0[0] > 0.985)
    foot = np.einsum("nj,njd->nd", w[sel_q], cen[ctri[idx[sel_q]]])
    ax.plot(q[sel_q, 0], q[sel_q, 2], "^", color=ORANGE, ms=4, label="target centroids (|y-0.4| < 6e-4)")
    for a_, b_ in zip(q[sel_q], foot):
        ax.plot([a_[0], b_[0]], [a_[2], b_[2]], "-", color="k", lw=0.6)
    nq = d["qn"][sel_q]
    ax.quiver(q[sel_q, 0], q[sel_q, 2], nq[:, 0], nq[:, 2], color=ORANGE, scale=40, width=0.003)
    xt = x_le0[0] + c0[0]
    ax.set_xlim(xt - 0.0048, xt + 0.0008)
    ax.set_ylim(-0.0033, 0.0033)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend(loc="lower left", fontsize=8)
    nchg = int(np.sum(d["idx_nn"] != d["idx"]))
    ax.set_title("(b) section y = 0.4 around the rounded trailing edge, target normals (arrows).\n"
                 f"Triangles beyond 60° are skipped; skipping changed {nchg} of {len(q):,} picks here")
    fig.suptitle("Step 5  Project target centroids and interpolate with barycentric weights")
    fig.savefig(out / "step5-projection.png", dpi=150)
    plt.close(fig)

    # --- ステップ 6: 結果
    fig = plt.figure(figsize=(17, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95])
    up_r = (rn[:, 2] > 0.3) & ~tip_r
    mid = up_r & (((q[:, 0] - planform(geom, q[:, 1])[0]) / planform(geom, q[:, 1])[1]) > 0.03) & \
        (((q[:, 0] - planform(geom, q[:, 1])[0]) / planform(geom, q[:, 1])[1]) < 0.97)
    lim = np.percentile(ref_cell[mid], [1, 99])
    for k, (val, title) in enumerate([(ref_cell, "(a) reference: reference mesh solved at AoA 10°"),
                                      (d["map_RING"], "(b) d10 cells mapped by the centroid mesh (RING)")]):
        ax = fig.add_subplot(gs[0, k])
        pc = PolyCollection(rp[rt[up_r]][:, :, :2], array=val[up_r], cmap="viridis", clim=lim, edgecolors="none")
        ax.add_collection(pc)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 0.82)
        ax.set_aspect("equal")
        ax.set_title(title + "\nupper surface, cell values")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(pc, ax=ax, shrink=0.7, label="Cp")
    ax = fig.add_subplot(gs[0, 2])
    diff = d["map_RING"] - ref_cell
    pc = PolyCollection(rp[rt[up_r]][:, :, :2], array=diff[up_r], cmap="RdBu_r", clim=(-0.02, 0.02), edgecolors="none")
    ax.add_collection(pc)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.82)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("(c) RING - reference (includes the CFD difference\nbetween the two meshes)")
    fig.colorbar(pc, ax=ax, shrink=0.7, label="ΔCp")
    # (d) 前縁まわりの分布（翼幅中央の帯）
    ax = fig.add_subplot(gs[1, 0:2])
    band = (np.abs(q[:, 1] - 0.4) < 0.0045) & (((q[:, 0] - planform(geom, q[:, 1])[0]) / planform(geom, q[:, 1])[1]) < 0.03)
    s_q = Q2[band, 1]
    o = np.argsort(s_q)
    ax.plot(s_q[o], ref_cell[band][o], "-", color="k", lw=2.0, label="reference (cells)")
    for name, col, mk in [("RING", PURPLE, "o"), ("FACE", RED, "s"), ("NEAR3", "#1D9E75", "^"), ("NODE", BLUE, "x")]:
        ax.plot(s_q[o], d[f"map_{name}"][band][o], mk, color=col, ms=3.5, mfc="none" if mk != "x" else None, label=name)
    ax.set_xlabel("arc length from LE s (upper +), faces with |y - 0.4| < 0.0045 and x/c < 3%")
    ax.set_ylabel("Cp")
    ax.legend(ncol=5, loc="lower left")
    ax.set_title("(d) leading edge near mid-span (no face centroid lies within 4e-4 of the nose line):\n"
                 "the interpolations agree; FACE scatters by one cell; the reference differs slightly near the stagnation")
    ax.set_xlim(-0.006, 0.006)
    # (e) 解析的な場の誤差
    ax = fig.add_subplot(gs[1, 2])
    summ = json.loads((out / "summary.json").read_text())
    pr = [p for p in summ["pairs"] if p["pair"].startswith("d10")][0]
    names = [m_["method"] for m_ in pr["methods"]]
    regions = [("rest", "away from edges"), ("edge", "LE/TE (3% chord)"), ("tip_face", "tip face"), ("uncovered", "uncovered band")]
    x = np.arange(len(names))
    wd = 0.2
    for i, (r, lab) in enumerate(regions):
        ax.bar(x + (i - 1.5) * wd, [m_["analytic_err"][r]["rms"] for m_ in pr["methods"]], wd, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("RMS error vs exact face mean")
    ax.legend(fontsize=8)
    ax.set_title("(e) analytic field sin(5x)cos(3y) + 8z on the same meshes\n(interpolation error only)")
    fig.suptitle("Step 6  Result for the d10 -> reference pair")
    fig.savefig(out / "step6-result.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:4])
