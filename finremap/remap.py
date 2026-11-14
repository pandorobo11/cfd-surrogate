"""表面三角形格子の間の補間と、剛体回転・力の積分。形状には依存しない。

補間は、問い合わせる点から元格子の三角形面上の最近接点を求め、その重心座標で頂点の値を平均する。
"""
import numpy as np
from scipy.spatial import cKDTree


def rotate(points, origin, axis, angle_deg):
    """点群を、origin を通り axis に平行な軸のまわりに右ねじの向きで angle_deg だけ回す（Rodrigues の式）。"""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    a = np.radians(angle_deg)
    p = np.asarray(points, float) - np.asarray(origin, float)
    out = p * np.cos(a) + np.cross(k, p) * np.sin(a) + np.outer(p @ k, k) * (1.0 - np.cos(a))
    return out + np.asarray(origin, float)


def face_normals_areas(points, tri):
    n = np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]])
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area, 1e-300)[:, None], area


def node_normals(points, tri):
    """節点の法線（まわりの三角形の面積重み付き平均）。"""
    fn, area = face_normals_areas(points, tri)
    nn = np.zeros_like(points)
    for k in range(3):
        np.add.at(nn, tri[:, k], fn * area[:, None])
    return nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-300)


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


def map_nearest(src_points, src_values, query):
    """最近傍の節点の値をそのまま使う（比較用）。"""
    return src_values[cKDTree(src_points).query(query)[1]]


def map_barycentric(src_points, src_tri, src_values, query, query_normals=None, k=24, chunk=20000):
    """最近接点を三角形面上に投影し、重心座標で補間する。

    query_normals を渡すと、法線が逆向きの三角形（裏面）を候補から外す。
    候補は、重心が近い k 個の三角形。戻り値は (補間値, 投影距離)。候補がすべて裏面の点の距離は inf。
    """
    tree = cKDTree(src_points[src_tri].mean(axis=1))
    fn = face_normals_areas(src_points, src_tri)[0] if query_normals is not None else None
    k = min(k, len(src_tri))
    values, dists = np.empty(len(query)), np.empty(len(query))
    for s in range(0, len(query), chunk):
        q = query[s:s + chunk]
        cand = tree.query(q, k=k)[1].reshape(len(q), k)
        v = src_points[src_tri[cand]]  # (n, k, 3, 3)
        w = closest_point_barycentric(q[:, None, :], v[:, :, 0], v[:, :, 1], v[:, :, 2])
        dist = np.linalg.norm(np.einsum("nkj,nkjd->nkd", w, v) - q[:, None, :], axis=-1)
        if fn is not None:
            back = np.einsum("nkd,nd->nk", fn[cand], query_normals[s:s + chunk]) < 0.0
            dist = np.where(back, np.inf, dist)
        best = np.argmin(dist, axis=1)
        rows = np.arange(len(q))
        values[s:s + chunk] = np.sum(w[rows, best] * src_values[src_tri[cand[rows, best]]], axis=1)
        dists[s:s + chunk] = dist[rows, best]
    return values, dists


def orient_outward(points, tri, interior):
    """各三角形の向きを、interior（三角形ごとの内部点 (M, 3)）から離れる向きにそろえた三角形を返す。"""
    n, _ = face_normals_areas(points, tri)
    flip = np.sum(n * (points[tri].mean(axis=1) - interior), axis=1) < 0
    out = tri.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def integrate_forces(points, tri, cp, ref_area, moment_origin):
    """Cp の面積分による力係数 (3,) とモーメント係数 (3,)。三角形の向きは外向き。基準長は 1。"""
    n, area = face_normals_areas(points, tri)
    f = -cp[tri].mean(axis=1)[:, None] * n * area[:, None] / ref_area
    r = points[tri].mean(axis=1) - np.asarray(moment_origin, float)
    return f.sum(axis=0), np.cross(r, f).sum(axis=0)
