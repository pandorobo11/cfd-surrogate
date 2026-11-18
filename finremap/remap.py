"""表面三角形格子の間の補間と、剛体回転・力の積分。形状には依存しない。

補間は、問い合わせる点から元格子の三角形面上の最近接点を求め、その重心座標で頂点の値を平均する。
三角形ごとの値（セル中心の値）は、セル中心を頂点とする三角形格子（centroid_mesh）を作って同じように補間する
（map_cell_values）。
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


def rotate_vectors(vectors, axis, angle_deg):
    """ベクトルの成分を、axis のまわりに angle_deg だけ回す（位置ではないので、軸の通る点には依存しない）。"""
    return rotate(vectors, [0.0, 0.0, 0.0], axis, angle_deg)


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


def map_barycentric(src_points, src_tri, src_values, query, query_normals=None, k=24, chunk=20000,
                    min_cos=0.0, return_index=False):
    """最近接点を三角形面上に投影し、重心座標で補間する。

    src_values は節点ごとのスカラー (N,) か、複数の成分 (N, C)。成分はそれぞれ独立に補間する。
    query_normals を渡すと、法線との内積が min_cos 未満の三角形（既定の 0 では裏面）を候補から外す。
    候補は、重心が近い k 個の三角形。戻り値は (補間値, 投影距離)。候補がすべて外れた点は、値を NaN、距離を inf にする。
    return_index なら、選んだ三角形の番号（外れた点は -1）と重心座標 (M, 3) も返す。
    """
    src_values = np.asarray(src_values, float)
    tree = cKDTree(src_points[src_tri].mean(axis=1))
    fn = face_normals_areas(src_points, src_tri)[0] if query_normals is not None else None
    k = min(k, len(src_tri))
    values, dists = np.empty((len(query),) + src_values.shape[1:]), np.empty(len(query))
    index, weights = np.empty(len(query), np.int64), np.empty((len(query), 3))
    for s in range(0, len(query), chunk):
        q = query[s:s + chunk]
        cand = tree.query(q, k=k)[1].reshape(len(q), k)
        v = src_points[src_tri[cand]]  # (n, k, 3, 3)
        w = closest_point_barycentric(q[:, None, :], v[:, :, 0], v[:, :, 1], v[:, :, 2])
        dist = np.linalg.norm(np.einsum("nkj,nkjd->nkd", w, v) - q[:, None, :], axis=-1)
        if fn is not None:
            back = np.einsum("nkd,nd->nk", fn[cand], query_normals[s:s + chunk]) < min_cos
            dist = np.where(back, np.inf, dist)
        best = np.argmin(dist, axis=1)
        rows = np.arange(len(q))
        d = dist[rows, best]
        val = np.einsum("nj,nj...->n...", w[rows, best], src_values[src_tri[cand[rows, best]]])
        val[~np.isfinite(d)] = np.nan
        values[s:s + chunk] = val
        dists[s:s + chunk] = d
        index[s:s + chunk] = np.where(np.isfinite(d), cand[rows, best], -1)
        weights[s:s + chunk] = w[rows, best]
    if return_index:
        return values, dists, index, weights
    return values, dists


def cell_values(tri, node_values):
    """節点値から三角形ごとの値を作る（3 頂点の平均）。節点値が三角形の中で線形なら、面平均に等しい。"""
    return np.asarray(node_values, float)[tri].mean(axis=1)


def _ear_clip(P, scale2):
    """反時計回りの単純多角形 P (m, k, 2) を、耳切りで k-2 枚の三角形に分ける。

    切り落とせる耳のうち、三角形の最小角が最大のものから切る。戻り値は (局所番号 (m, k-2, 3), 分けられたか (m,))。
    """
    m, k, _ = P.shape
    rows = np.arange(m)[:, None]
    idx = np.tile(np.arange(k), (m, 1))
    out = np.empty((m, k - 2, 3), np.int64)
    ok = np.ones(m, bool)
    eps = 1e-12 * scale2[:, None]

    def cross(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

    def min_angle(A, B, C):
        def ang(p, q, r):
            u, v = q - p, r - p
            c = np.sum(u * v, -1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-300)
            return np.arccos(np.clip(c, -1.0, 1.0))
        return np.minimum(np.minimum(ang(A, B, C), ang(B, C, A)), ang(C, A, B))

    for s in range(k - 2):
        n = idx.shape[1]
        X = P[rows, idx]
        A, C = np.roll(X, 1, axis=1), np.roll(X, -1, axis=1)
        valid = cross(A, X, C) > eps
        if n > 3:  # 耳の中に他の頂点が入らないこと
            Q = X[:, None, :, :]
            inside = ((cross(A[:, :, None], X[:, :, None], Q) > eps[..., None])
                      & (cross(X[:, :, None], C[:, :, None], Q) > eps[..., None])
                      & (cross(C[:, :, None], A[:, :, None], Q) > eps[..., None]))
            valid &= ~inside.any(-1)
            score = np.where(valid, min_angle(A, X, C), -1.0)
            best = np.argmax(score, axis=1)
        else:
            best = np.ones(m, np.int64)
            score = np.where(valid, 1.0, -1.0)
        r = np.arange(m)
        ok &= score[r, best] >= 0
        out[:, s] = np.stack([idx[r, best - 1], idx[r, best], idx[r, (best + 1) % n]], axis=1)
        keep = np.ones((m, n), bool)
        keep[r, best] = False
        idx = idx[keep].reshape(m, n - 1)
    return out, ok


def centroid_mesh(points, tri, feature_angle_deg=45.0, part=None, chunk=50000):
    """セル中心（三角形の重心）を頂点とする三角形格子を、元格子のつながりから作る。セル中心のデータの補間用。

    元格子の各節点について、その節点を共有する三角形を辺でたどって並べ（外から見て反時計回り）、
    重心を順に結んだ多角形（節点の環）を耳切りで三角形に分ける。どの環の辺も隣の環と共有されるので、
    三角形は重ならずに表面を覆う。境界の辺と、隣の三角形との法線の角度が feature_angle_deg を超える辺
    （特徴線）、part（三角形ごとの部位番号）が変わる辺では環を切る。切れた環は、両端の重心を結んで閉じる。

    tri は外向きにそろえておく。戻り値は (重心 (F, 3), 三角形 (K, 3)（重心の番号＝元の三角形の番号）, 情報)。
    情報の ring_vertex は、各三角形を作った環の節点の番号。
    """
    points, tri = np.asarray(points, float), np.asarray(tri, np.int64)
    nf, npt = len(tri), len(points)
    cen = points[tri].mean(axis=1)
    fn, area = face_normals_areas(points, tri)
    # 角 c = 3f+i の節点 v、次の節点 a、前の節点 b。角 c は半辺 v->a を持つ
    v, a, b = tri.ravel(), np.roll(tri, -1, axis=1).ravel(), np.roll(tri, -2, axis=1).ravel()
    key = v * npt + a
    order = np.argsort(key, kind="stable")
    sk = key[order]
    dup = np.zeros(len(key), bool)
    same = sk[1:] == sk[:-1]
    dup[order[1:][same]] = dup[order[:-1][same]] = True
    ukey = np.minimum(v, a) * npt + np.maximum(v, a)
    _, inv, cnt = np.unique(ukey, return_inverse=True, return_counts=True)
    # v のまわりで次の三角形は、辺 (v, b) の向こう側（半辺 v->b を持つ角）
    q = v * npt + b
    pos = np.minimum(np.searchsorted(sk, q), len(sk) - 1)
    nxt = np.where(sk[pos] == q, order[pos], -1)
    f = np.repeat(np.arange(nf), 3)
    corner_bv = 3 * f + np.tile([2, 0, 1], nf)  # 半辺 b->v を持つ同じ三角形の角
    g = np.where(nxt >= 0, nxt // 3, 0)
    cut = (nxt < 0) | (cnt[inv[corner_bv]] != 2) | dup[corner_bv] | dup[np.maximum(nxt, 0)]
    cut |= np.sum(fn[f] * fn[g], axis=1) < np.cos(np.radians(feature_angle_deg))
    if part is not None:
        part = np.asarray(part)
        cut |= part[f] != part[g]
    nxt = np.where(cut, -1, nxt)
    prv = np.full(len(nxt), -1)
    prv[nxt[nxt >= 0]] = np.nonzero(nxt >= 0)[0]

    # 環（閉じた列）と、切れた環（開いた列）を取り出す
    nxt_l, visited, chains = nxt.tolist(), np.zeros(len(nxt), bool), []
    for start in np.concatenate([np.nonzero(prv < 0)[0], np.arange(len(nxt))]).tolist():
        if visited[start]:
            continue
        c, faces = start, []
        while c >= 0 and not visited[c]:
            visited[c] = True
            faces.append(c // 3)
            c = nxt_l[c]
        chains.append((int(v[start]), faces, c == start))
    info = dict(faces=nf, closed_rings=sum(1 for c in chains if c[2]), open_rings=sum(1 for c in chains if not c[2]),
                short_rings=sum(1 for c in chains if len(c[1]) < 3), failed_rings=0, failed_vertices=[],
                boundary_edges=int(np.sum(cnt == 1)),
                cut_edges=int(len(np.unique(ukey[corner_bv[cut & (cnt[inv[corner_bv]] != 1)]]))))
    out_tri, out_v = [], []
    by_len = {}
    for vert, faces, closed in chains:
        if len(faces) >= 3:
            by_len.setdefault(len(faces), []).append((vert, faces, closed))
    for k, group in sorted(by_len.items()):
        for s in range(0, len(group), chunk):
            part_g = group[s:s + chunk]
            F = np.array([g_[1] for g_ in part_g])  # (m, k) 三角形の番号
            vert = np.array([g_[0] for g_ in part_g])
            closed = np.array([g_[2] for g_ in part_g])
            n = np.sum(fn[F] * area[F][..., None], axis=1)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
            e1 = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
            e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
            e2 = np.cross(n, e1)
            X = cen[F] - points[vert][:, None, :]
            P = np.stack([np.sum(X * e1[:, None], -1), np.sum(X * e2[:, None], -1)], axis=-1)
            scale2 = np.max(np.sum(P**2, -1), axis=1)
            signed = 0.5 * np.sum(P[:, :, 0] * np.roll(P[:, :, 1], -1, 1) - np.roll(P[:, :, 0], -1, 1) * P[:, :, 1], 1)
            good = signed > 1e-12 * scale2
            # 切れた環は、両端の重心を結ぶ線より節点が外にあるときだけ使う（扇の角度が 180° 未満）
            cr = P[:, -1, 0] * P[:, 0, 1] - P[:, -1, 1] * P[:, 0, 0]
            good &= closed | (cr < 0)
            loc, ok = _ear_clip(P, scale2)
            good &= ok
            info["failed_rings"] += int((~good).sum())
            info["failed_vertices"] += vert[~good].tolist()
            T = np.take_along_axis(F[:, None, :].repeat(k - 2, 1), loc, axis=2)[good].reshape(-1, 3)
            out_tri.append(T)
            out_v.append(np.repeat(vert[good], k - 2))
    ctri = np.concatenate(out_tri) if out_tri else np.empty((0, 3), np.int64)
    info["ring_vertex"] = np.concatenate(out_v) if out_v else np.empty(0, np.int64)
    info["triangles"] = len(ctri)
    return cen, ctri, info


def tangent_project(vectors, normals):
    """ベクトルから法線方向の成分を除く（normals は単位ベクトル）。"""
    return vectors - np.sum(vectors * normals, axis=1, keepdims=True) * normals


def orient_outward(points, tri, interior):
    """各三角形の向きを、interior（三角形ごとの内部点 (M, 3)）から離れる向きにそろえた三角形を返す。"""
    n, _ = face_normals_areas(points, tri)
    flip = np.sum(n * (points[tri].mean(axis=1) - interior), axis=1) < 0
    out = tri.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def integrate_forces(points, tri, cp, ref_area, moment_origin):
    """Cp の面積分による力係数 (3,) とモーメント係数 (3,)。三角形の向きは外向き。基準長は 1。

    Cp が NaN の節点を含む三角形は積分から外す。
    """
    tri = tri[np.isfinite(cp[tri]).all(axis=1)]
    n, area = face_normals_areas(points, tri)
    f = -cp[tri].mean(axis=1)[:, None] * n * area[:, None] / ref_area
    r = points[tri].mean(axis=1) - np.asarray(moment_origin, float)
    return f.sum(axis=0), np.cross(r, f).sum(axis=0)


def plane_barycentric(p, a, b, c):
    """点 p を三角形 abc の平面に射影した点の重心座標 (M, 3)。三角形の外では負の値を含む（外挿）。"""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = np.sum(v0 * v0, -1), np.sum(v0 * v1, -1), np.sum(v1 * v1, -1)
    d20, d21 = np.sum(v2 * v0, -1), np.sum(v2 * v1, -1)
    den = d00 * d11 - d01 * d01
    with np.errstate(divide="ignore", invalid="ignore"):
        wb, wc = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    return np.stack([1 - wb - wc, wb, wc], -1)


def boundary_foot(tri, index, weights, n_points):
    """map_barycentric(return_index=True) の結果について、投影した点が格子の縁（1 枚の三角形にしか属さない辺）の上にあるか。"""
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    key = e[:, 0] * n_points + e[:, 1]
    uk, cnt = np.unique(key, return_counts=True)
    bkey = uk[cnt == 1]
    bvert = np.unique(np.stack([bkey // n_points, bkey % n_points], 1))
    t = tri[np.maximum(index, 0)]
    zero = weights < 1e-9
    nz = zero.sum(1)
    out = np.zeros(len(index), bool)
    for i in range(3):
        m = (nz == 1) & zero[:, i]
        a, b = t[m, (i + 1) % 3], t[m, (i + 2) % 3]
        out[m] = np.isin(np.minimum(a, b) * n_points + np.maximum(a, b), bkey)
    m = nz == 2
    out[m] = np.isin(t[m][~zero[m]], bvert)
    return out & (index >= 0)


def map_cell_values(src_points, src_tri, src_cell, query, query_normals, feature_angle_deg=45.0, part=None,
                    min_cos=0.5, extrapolate=True):
    """三角形ごとの値（セル値）を、問い合わせる点（ふつうは補間先の三角形の重心）へ補間する。

    centroid_mesh で作ったセル中心の三角形格子に点を投影し、重心座標で 3 つのセル値を平均する。
    query_normals（補間先の三角形の法線）との角度が arccos(min_cos) を超える三角形は候補から外す。
    格子が覆わない縁の帯（境界と特徴線に沿った、幅が三角形の 1/3 程度の帯）に落ちた点は、
    extrapolate なら最も近い三角形の平面で重心座標を延長し（線形の外挿）、そうでなければ縁の辺の上の値を使う。
    戻り値は (補間値, 投影距離, 縁の帯に落ちたか)。src_cell は (M,) か (M, C)。
    """
    cen, ctri, _ = centroid_mesh(src_points, src_tri, feature_angle_deg, part)
    values, dist, idx, w = map_barycentric(cen, ctri, src_cell, query, query_normals, min_cos=min_cos,
                                           return_index=True)
    band = boundary_foot(ctri, idx, w, len(cen))
    if extrapolate and band.any():
        t = ctri[idx[band]]
        wx = plane_barycentric(query[band], cen[t[:, 0]], cen[t[:, 1]], cen[t[:, 2]])
        values[band] = np.einsum("nj,nj...->n...", wx, np.asarray(src_cell, float)[t])
    return values, dist, band


def integrate_forces_cells(points, tri, cp_cell, ref_area, moment_origin):
    """三角形ごとの Cp（セル値 (M,)）の面積分による力係数 (3,) とモーメント係数 (3,)。三角形の向きは外向き。

    cp_cell を cell_values で節点値から作ると、integrate_forces と同じ値になる。NaN の三角形は積分から外す。
    """
    ok = np.isfinite(cp_cell)
    n, area = face_normals_areas(points, tri[ok])
    f = -cp_cell[ok][:, None] * n * area[:, None] / ref_area
    r = points[tri[ok]].mean(axis=1) - np.asarray(moment_origin, float)
    return f.sum(axis=0), np.cross(r, f).sum(axis=0)


def integrate_friction(points, tri, cf, ref_area, moment_origin):
    """摩擦係数ベクトル cf (N, 3) の面積分による力係数 (3,) とモーメント係数 (3,)。基準長は 1。

    cf は物体が受ける向きの壁面せん断応力を動圧で割ったもので、points と同じ座標系の成分で渡す。
    法線は掛けないので、三角形の向きには依存しない。cf が NaN の節点を含む三角形は積分から外す。
    """
    tri = tri[np.isfinite(cf[tri]).all(axis=(1, 2))]
    _, area = face_normals_areas(points, tri)
    f = cf[tri].mean(axis=1) * area[:, None] / ref_area
    r = points[tri].mean(axis=1) - np.asarray(moment_origin, float)
    return f.sum(axis=0), np.cross(r, f).sum(axis=0)
