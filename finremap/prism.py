"""翼面の三角形格子から、プリズム層と外側の四面体からなる粘性計算用の格子（SU2 形式）を作る。

手順:
  1. 翼面の三角形を外向き（流体側）にそろえ、節点の法線（角度重み）を求めて平滑化する。
     対称面 (y=0) 上の節点では法線の y 成分を 0 にし、付け根から root_blend の範囲で滑らかにつなぐ。
  2. 節点ごとに層全体の厚さ T を決める。平板の層流境界層の厚さに比例させ（T = thickness_coeff √(x ν/U)、
     x は局所前縁からの距離）、下限を t_min にする。後縁から翼弦の te_zone の範囲では t_te まで薄くする。
  3. 層の高さは等比（公比 ratio、layers 層）。最初の層の高さは T (ratio - 1) / (ratio^layers - 1)。
  4. 層の上面・対称面の残り・遠方境界で閉じた面を作り、Gmsh（HXT）で四面体に分ける。
     格子幅は、四面体だけの格子と同じ「翼面・縁からの距離のしきい値」で決める。

境界の名前は fin（翼面）、sym（対称面: 三角形と、プリズム層の側面の四角形）、far（遠方）。
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from .remap import face_normals_areas, orient_outward, rotate, rotate_vectors
from .workflow import BL_DEFAULTS  # workflow は標準ライブラリだけで動くので、既定値はそちらに置く

HINGE_AXIS = [0.0, 1.0, 0.0]
SU2_TYPE = dict(tri=5, quad=9, tet=10, prism=13)


def planform(geom, y):
    eta = np.asarray(y, float) / geom["span"]
    return geom["tip_le_x"] * eta, geom["root_chord"] + (geom["tip_chord"] - geom["root_chord"]) * eta


def outward_triangles(fin_xyz, tri, geom):
    """翼固定座標の翼面三角形を、流体側（翼の外）へ向く順にそろえる。analysis.load_surface と同じ判定。"""
    cen = fin_xyz[tri].mean(axis=1)
    x_le, chord = planform(geom, cen[:, 1])
    interior = np.stack([x_le + 0.4 * chord, cen[:, 1], np.zeros(len(cen))], axis=1)
    tip = np.all(fin_xyz[tri][:, :, 1] > geom["span"] * (1 - 1e-6), axis=1)
    interior[tip] = cen[tip] - [0.0, 1.0, 0.0]
    return orient_outward(fin_xyz, tri, interior)


def angle_weighted_normals(xyz, tri):
    fn, _ = face_normals_areas(xyz, tri)
    nn = np.zeros_like(xyz)
    for k in range(3):
        a, b, c = xyz[tri[:, k]], xyz[tri[:, (k + 1) % 3]], xyz[tri[:, (k + 2) % 3]]
        u, v = b - a, c - a
        cosang = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        np.add.at(nn, tri[:, k], fn * np.arccos(np.clip(cosang, -1, 1))[:, None])
    return nn / np.linalg.norm(nn, axis=1, keepdims=True)


def adjacency(n, tri):
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    a = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    return a


def smooth(values, adj, iters, unit=False):
    deg = np.asarray(adj.sum(axis=1)).ravel()
    v = values.copy()
    for _ in range(iters):
        avg = (adj @ v) / (deg[:, None] if v.ndim == 2 else deg)
        v = 0.5 * v + 0.5 * avg
        if unit:
            v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def root_loop(tri):
    """一度しか使われない辺（付け根の縁）を、三角形の向きのまま (始点, 終点) で返す。"""
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return e[cnt[inv.ravel()] == 1]


def order_loop(edges):
    nxt = dict(zip(edges[:, 0].tolist(), edges[:, 1].tolist()))
    if len(nxt) != len(edges):
        raise RuntimeError("付け根の縁が 1 本の閉じた輪になっていない")
    start = int(edges[0, 0])
    loop = [start]
    while True:
        v = nxt[loop[-1]]
        if v == start:
            break
        loop.append(v)
        if len(loop) > len(edges):
            raise RuntimeError("付け根の縁が 1 本の閉じた輪になっていない")
    if len(loop) != len(edges):
        raise RuntimeError(f"付け根の縁が複数の輪に分かれている（{len(loop)} / {len(edges)} 辺）")
    return np.array(loop)


def layer_thickness(fin_xyz, geom, bl, reynolds):
    """節点ごとの層全体の厚さ T（翼固定座標で計算）。"""
    y = np.clip(fin_xyz[:, 1], 0.0, geom["span"])
    x_le, chord = planform(geom, y)
    xi = np.clip((fin_xyz[:, 0] - x_le) / chord, 0.0, 1.0)
    nu_over_u = geom["root_chord"] / reynolds  # 付け根翼弦基準のレイノルズ数から
    t = np.maximum(bl["thickness_coeff"] * np.sqrt(xi * chord * nu_over_u), bl["t_min"])
    s = np.clip((1.0 - xi) / bl["te_zone"], 0.0, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * s))  # 後縁で 0、後縁から te_zone 以上離れると 1
    return bl["t_te"] + (t - bl["t_te"]) * w


def extrude(xyz, tri, delta_deg, geom, bl, reynolds):
    """プリズム層を作る。戻り値は dict（節点、プリズム、上面の三角形、対称面の四角形、付け根の輪など）。"""
    origin = [geom["hinge_x"], 0.0, 0.0]
    xyz = xyz.copy()
    xyz[np.abs(xyz[:, 1]) < 1e-9, 1] = 0.0
    fin_xyz = rotate(xyz, origin, HINGE_AXIS, -delta_deg)
    tri = outward_triangles(fin_xyz, tri, geom)
    n = len(xyz)
    adj = adjacency(n, tri)

    normal = smooth(angle_weighted_normals(fin_xyz, tri), adj, bl["smooth_iter"], unit=True)
    b = 0.5 * (1.0 + np.cos(np.pi * np.clip(fin_xyz[:, 1] / bl["root_blend"], 0.0, 1.0)))  # 付け根で 1
    normal[:, 1] *= 1.0 - b
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    thick = smooth(layer_thickness(fin_xyz, geom, bl, reynolds), adj, bl["smooth_iter"])

    edges = root_loop(tri)
    loop = order_loop(edges)
    if np.max(np.abs(xyz[loop, 1])) > 0.0:
        raise RuntimeError("付け根の縁の節点が対称面 (y=0) にない")

    L, r = bl["layers"], bl["ratio"]
    frac = (r ** np.arange(L + 1) - 1.0) / (r ** L - 1.0)  # 0 ... 1
    normal_mesh = rotate_vectors(normal, HINGE_AXIS, delta_deg)
    pts = xyz[None, :, :] + frac[:, None, None] * thick[None, :, None] * normal_mesh[None, :, :]
    pts = pts.reshape(-1, 3)
    pts[:, 1][np.abs(pts[:, 1]) < 1e-12] = 0.0
    gid = lambda k, i: k * n + i

    # SU2（VTK）のプリズムは、底面 (0, 1, 2) の右ねじの向きが上面 (3, 4, 5) と反対を向く順
    cw = tri[:, [0, 2, 1]]
    prisms = np.concatenate([np.concatenate([gid(k, cw), gid(k + 1, cw)], axis=1) for k in range(L)])
    quads = []
    for k in range(L):
        a, c = edges[:, 0], edges[:, 1]
        quads.append(np.stack([gid(k, a), gid(k, c), gid(k + 1, c), gid(k + 1, a)], axis=1))
    quads = np.concatenate(quads)
    # 境界の要素は、SU2 に合わせて法線が流体の側を向く順にする（対称面では +y）
    qn = np.cross(pts[quads[:, 1]] - pts[quads[:, 0]], pts[quads[:, 3]] - pts[quads[:, 0]])
    quads[qn[:, 1] < 0] = quads[qn[:, 1] < 0][:, ::-1]

    # 品質の確認: 各層の三角形が裏返っていないか、プリズムを 3 つの四面体に分けたときの体積
    fn0, _ = face_normals_areas(fin_xyz, tri)
    min_dot, min_rel_vol = 1.0, np.inf
    col = np.einsum("tkj,tj->tk", normal[tri], fn0)  # 押し出しの向きと面の法線
    first_h = thick * frac[1]
    for k in range(L):
        lo, hi = pts[gid(k, tri)], pts[gid(k + 1, tri)]
        fk = np.cross(hi[:, 1] - hi[:, 0], hi[:, 2] - hi[:, 0])
        fk /= np.linalg.norm(fk, axis=1, keepdims=True)
        fm = rotate_vectors(fn0, HINGE_AXIS, delta_deg)
        min_dot = min(min_dot, float(np.min(np.sum(fk * fm, 1))))
        vol = lambda p0, p1, p2, p3: np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
        v = np.stack([vol(lo[:, 0], lo[:, 1], lo[:, 2], hi[:, 0]), vol(lo[:, 1], lo[:, 2], hi[:, 0], hi[:, 1]),
                      vol(lo[:, 2], hi[:, 0], hi[:, 1], hi[:, 2])], axis=1)
        ref = np.linalg.norm(np.cross(lo[:, 1] - lo[:, 0], lo[:, 2] - lo[:, 0]), axis=1) / 2.0 \
            * np.mean((frac[k + 1] - frac[k]) * thick[tri], axis=1) / 3.0
        min_rel_vol = min(min_rel_vol, float(np.min(v / ref[:, None])))
    top_tri = gid(L, tri)
    stats = dict(layers=L, ratio=r, thickness_min=float(thick.min()), thickness_max=float(thick.max()),
                 first_height_min=float(first_h.min()), first_height_max=float(first_h.max()),
                 min_extrusion_dot=float(col.min()), min_layer_face_dot=min_dot, min_prism_rel_volume=min_rel_vol)
    return dict(pts=pts, n_surface=n, tri=tri, prisms=prisms, quads=quads, top_tri=top_tri, top_loop=gid(L, loop),
                normal=normal, thickness=thick, stats=stats, fin_xyz=fin_xyz)


def size_function(xyz, fin_xyz, geom, domain, h_fin, h_edge):
    """四面体だけの格子と同じ格子幅: min(翼面からの距離のしきい値, 縁からの距離のしきい値)。"""
    y = fin_xyz[:, 1]
    x_le, chord = planform(geom, np.clip(y, 0.0, geom["span"]))
    xi = (fin_xyz[:, 0] - x_le) / chord
    zone = geom["edge_zone"]
    edge = (y < geom["span"] - 1e-4) & ((xi < zone) | (xi > 1.0 - zone))
    kd_fin, kd_edge = cKDTree(xyz), cKDTree(xyz[edge])
    h_far = domain["h_far"]

    def thr(d, h, d_min):
        return h if d <= d_min else (h_far if d >= 3.0 else h + (h_far - h) * (d - d_min) / (3.0 - d_min))

    def size(dim, tag, x, y_, z, lc):
        p = (x, y_, z)
        return min(thr(kd_fin.query(p)[0], h_fin, 0.03), thr(kd_edge.query(p)[0], h_edge, 0.002))

    return size


def outer_surfaces(gmsh, top_loop_xyz, domain, size):
    """対称面の残り（付け根の輪の外側）と遠方境界の 5 面を、組み込みの幾何で 2 次元格子にする。

    戻り値: (節点座標, sym の三角形, far の三角形)。付け根の輪の節点は、輪の座標と同じ値になる。
    """
    gmsh.model.add("outer")
    geo = gmsh.model.geo
    (x0, x1), (z0, z1), y1 = domain["x"], domain["z"], domain["y_max"]
    c = {k: geo.addPoint(*v) for k, v in dict(
        a=(x0, 0, z0), b=(x1, 0, z0), cc=(x1, 0, z1), d=(x0, 0, z1),
        e=(x0, y1, z0), f=(x1, y1, z0), g=(x1, y1, z1), h=(x0, y1, z1)).items()}
    L = lambda p, q: geo.addLine(c[p], c[q])
    ab, bc, cd, da = L("a", "b"), L("b", "cc"), L("cc", "d"), L("d", "a")
    ef, fg, gh, he = L("e", "f"), L("f", "g"), L("g", "h"), L("h", "e")
    ae, bf, cg, dh = L("a", "e"), L("b", "f"), L("cc", "g"), L("d", "h")
    ring_pts = [geo.addPoint(*p) for p in top_loop_xyz]
    ring = [geo.addLine(ring_pts[i], ring_pts[(i + 1) % len(ring_pts)]) for i in range(len(ring_pts))]
    loop = lambda curves: geo.addCurveLoop(curves)
    sym = geo.addPlaneSurface([loop([ab, bc, cd, da]), loop(ring)])
    far = [geo.addPlaneSurface([loop([ef, fg, gh, he])]),
           geo.addPlaneSurface([loop([ab, bf, -ef, -ae])]),
           geo.addPlaneSurface([loop([bc, cg, -fg, -bf])]),
           geo.addPlaneSurface([loop([cd, dh, -gh, -cg])]),
           geo.addPlaneSurface([loop([da, ae, -he, -dh])])]
    geo.synchronize()
    for t in ring:
        gmsh.model.mesh.setTransfiniteCurve(t, 2)
    gmsh.model.mesh.setSizeCallback(size)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    pos = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    pos[tags.astype(np.int64)] = np.arange(len(tags))
    get = lambda s: pos[gmsh.model.mesh.getElements(2, s)[2][0].astype(np.int64)].reshape(-1, 3)
    xyz = coords.reshape(-1, 3)
    # 法線を流体の側へ向ける（SU2 の境界要素の向き）。対称面は +y、遠方は箱の中心の側
    center = np.array([0.5 * (x0 + x1), 0.5 * y1, 0.5 * (z0 + z1)])
    sym_tri = flip_to(xyz, get(sym), lambda c: np.array([0.0, 1.0, 0.0]))
    far_tri = flip_to(xyz, np.concatenate([get(s) for s in far]), lambda c: center - c)
    gmsh.model.remove()
    return xyz, sym_tri, far_tri


def flip_to(xyz, tri, direction):
    """三角形の法線が direction(重心) の側を向くように頂点の順をそろえる。"""
    n, _ = face_normals_areas(xyz, tri)
    flip = np.sum(n * direction(xyz[tri].mean(axis=1)), axis=1) < 0
    out = tri.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def volume_mesh(gmsh, bnd_xyz, bnd_tri, size, threads):
    """閉じた三角形の面の内側を四面体に分ける。戻り値: (全節点の座標, 四面体, 最小 SICN)。

    bnd_xyz の i 番目の節点は、戻り値の i 番目の節点になる。
    """
    gmsh.model.add("volume")
    s = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, s, np.arange(1, len(bnd_xyz) + 1), bnd_xyz.ravel())
    gmsh.model.mesh.addElementsByType(s, 2, [], (bnd_tri + 1).ravel())
    sl = gmsh.model.geo.addSurfaceLoop([s])
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setSizeCallback(size)
    for name in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{name}", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("General.NumThreads", threads)
    gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    tags = tags.astype(np.int64)
    xyz = np.zeros((int(tags.max()), 3))
    xyz[tags - 1] = coords.reshape(-1, 3)
    if not np.allclose(xyz[:len(bnd_xyz)], bnd_xyz, rtol=0, atol=0):
        raise RuntimeError("四面体の格子を作るときに境界の節点が動いた")
    tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(4)
    quality = gmsh.model.mesh.getElementQualities(tet_tags.tolist(), "minSICN")
    gmsh.model.remove()
    return xyz, tet_nodes.astype(np.int64).reshape(-1, 4) - 1, float(np.min(quality))


def write_su2(path, xyz, volumes, markers):
    """volumes: [(SU2 の要素番号, 節点 (M, k))]、markers: {名前: [(要素番号, 節点 (M, k))]}。番号は 0 始まり。"""
    with open(path, "w") as f:
        f.write("NDIME= 3\n")
        f.write(f"NELEM= {sum(len(c) for _, c in volumes)}\n")
        i0 = 0
        for etype, conn in volumes:
            body = np.column_stack([np.full(len(conn), etype), conn, i0 + np.arange(len(conn))])
            np.savetxt(f, body, fmt="%d")
            i0 += len(conn)
        f.write(f"NPOIN= {len(xyz)}\n")
        np.savetxt(f, np.column_stack([xyz, np.arange(len(xyz))]), fmt="%.17g %.17g %.17g %d")
        f.write(f"NMARK= {len(markers)}\n")
        for name, parts in markers.items():
            f.write(f"MARKER_TAG= {name}\nMARKER_ELEMS= {sum(len(c) for _, c in parts)}\n")
            for etype, conn in parts:
                np.savetxt(f, np.column_stack([np.full(len(conn), etype), conn]), fmt="%d")


def generate(path, xyz, tri, delta_deg, geom, domain, h_fin, h_edge, bl, reynolds, threads=1):
    """翼面の三角形格子 (xyz, tri) から格子を作って path に書く。mesh.json に入れる情報を返す。"""
    import gmsh

    if reynolds is None:
        raise ValueError("プリズム層の厚さを決めるためにレイノルズ数が要る（flow.reynolds）")
    bl = {**BL_DEFAULTS, **bl}
    ex = extrude(xyz, tri, delta_deg, geom, bl, reynolds)
    if ex["stats"]["min_prism_rel_volume"] <= 0.0 or ex["stats"]["min_layer_face_dot"] <= 0.0:
        raise RuntimeError(f"プリズム層が裏返った: {ex['stats']}")
    pts, n_bl = ex["pts"], len(ex["pts"])
    size = size_function(xyz, ex["fin_xyz"], geom, domain, h_fin, h_edge)

    # 外側の境界: 対称面の残りと遠方。付け根の輪の節点を、プリズム層の上面の輪の節点に対応させる
    out_xyz, sym_tri, far_tri = outer_surfaces(gmsh, pts[ex["top_loop"]], domain, size)
    d, idx = cKDTree(pts[ex["top_loop"]]).query(out_xyz)
    on_ring = d == 0.0
    if on_ring.sum() != len(ex["top_loop"]):
        raise RuntimeError(f"付け根の輪の節点が対応しない（{on_ring.sum()} / {len(ex['top_loop'])}）")
    top_nodes = np.unique(ex["top_tri"])
    # 四面体を作る面の節点: 上面の節点、対称面と遠方の節点（輪の上は除く）
    others = np.flatnonzero(~on_ring)
    bnd_global = np.concatenate([top_nodes, n_bl + np.arange(len(others))])
    local_of_out = np.empty(len(out_xyz), dtype=np.int64)
    local_of_out[others] = len(top_nodes) + np.arange(len(others))
    local_of_top = np.full(n_bl, -1, dtype=np.int64)
    local_of_top[top_nodes] = np.arange(len(top_nodes))
    local_of_out[on_ring] = local_of_top[ex["top_loop"][idx[on_ring]]]
    bnd_xyz = np.concatenate([pts[top_nodes], out_xyz[others]])
    bnd_tri = np.concatenate([local_of_top[ex["top_tri"]], local_of_out[sym_tri], local_of_out[far_tri]])
    vol_xyz, tets_local, min_sicn = volume_mesh(gmsh, bnd_xyz, bnd_tri, size, threads)

    # 四面体の格子の節点番号を全体の番号へ
    n_new = len(vol_xyz) - len(bnd_xyz)
    to_global = np.concatenate([bnd_global, n_bl + len(others) + np.arange(n_new)])
    all_xyz = np.concatenate([pts, out_xyz[others], vol_xyz[len(bnd_xyz):]])
    tets = to_global[tets_local]
    used = np.zeros(len(all_xyz), bool)
    used[ex["prisms"].ravel()] = True
    used[tets.ravel()] = True
    if not used.all():
        raise RuntimeError(f"どの要素にも使われない節点が {np.sum(~used)} 個ある")

    sym_global = to_global[local_of_out[sym_tri]]
    far_global = to_global[local_of_out[far_tri]]
    write_su2(path, all_xyz, [(SU2_TYPE["prism"], ex["prisms"]), (SU2_TYPE["tet"], tets)],
              dict(fin=[(SU2_TYPE["tri"], ex["tri"])],
                   sym=[(SU2_TYPE["tri"], sym_global), (SU2_TYPE["quad"], ex["quads"])],
                   far=[(SU2_TYPE["tri"], far_global)]))
    return dict(nodes=len(all_xyz), prisms=len(ex["prisms"]), tets=len(tets), fin_nodes=ex["n_surface"],
                fin_triangles=len(ex["tri"]), min_sicn=min_sicn, boundary_layer=bl, prism_stats=ex["stats"])
