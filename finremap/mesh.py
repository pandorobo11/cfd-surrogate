"""前縁と後縁に丸みを付けた全遊動翼単体の Euler 用四面体格子（SU2 形式）を、舵角を指定して生成する。

翼は対称面 (y=0) に取り付ける。翼固定座標: x=翼弦, y=翼幅, z=翼厚。
翼型（翼弦で無次元化）: z = ±(a0 sqrt(xi (1 - xi)) + b xi (1 - xi))
  前縁半径と後縁半径はともに r/c = a0^2 / 2、b は最大翼厚比が thickness_ratio になるように決める。
舵軸は y 軸に平行で (x, z) = (hinge_x, 0) を通る。舵角は y 軸まわりの右ねじ回転で、正のとき前縁が +z 側に上がる
（翼単体では一様流の迎角と等価）。

gmsh（pip install gmsh）が必要。境界の名前は fin（翼面）、sym（対称面）、far（遠方）。
"""
import json
from pathlib import Path

import numpy as np

GEOMETRY = dict(root_chord=1.0, tip_chord=0.5, span=0.8, tip_le_x=0.35, thickness_ratio=0.06,
                edge_radius_ratio=0.005, edge_zone=0.03, hinge_x=0.45)
DOMAIN = dict(x=[-2.0, 5.0], y_max=3.0, z=[-3.0, 3.0], h_far=0.5)


def thickness_coeffs(geom):
    """翼型の係数 (a0, b)。"""
    a0 = float(np.sqrt(2.0 * geom["edge_radius_ratio"]))
    return a0, float(4.0 * (0.5 * geom["thickness_ratio"] - 0.5 * a0))  # xi=0.5 で最大（対称な翼型）


def half_thickness(geom, xi):
    a0, b = thickness_coeffs(geom)
    return a0 * np.sqrt(np.clip(xi * (1.0 - xi), 0.0, None)) + b * xi * (1.0 - xi)


def planform(geom, y):
    """翼幅位置 y での (前縁の x, 翼弦長)。"""
    eta = np.asarray(y, float) / geom["span"]
    return geom["tip_le_x"] * eta, geom["root_chord"] + (geom["tip_chord"] - geom["root_chord"]) * eta


def ref_area(geom):
    return 0.5 * (geom["root_chord"] + geom["tip_chord"]) * geom["span"]


def generate(out_dir, delta_deg, geom=GEOMETRY, domain=DOMAIN, h_fin=0.012, h_edge=0.001, algo2d=6, threads=1):
    """格子を作り、out_dir に mesh.su2 と mesh.json を書く。mesh.json の内容を返す。"""
    import gmsh

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zone = geom["edge_zone"]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ

        def section(y):
            x_le, chord = (float(v) for v in planform(geom, y))
            pt = lambda xi, sign: occ.addPoint(x_le + xi * chord, y, sign * float(half_thickness(geom, xi)) * chord)
            s = np.linspace(-np.sqrt(zone), np.sqrt(zone), 41)  # xi = s^2 で丸みを均等に標本化
            nose = [pt(v * v, np.sign(v)) for v in s]  # 前縁: 下面側 → 前縁 → 上面側
            tail = [pt(1.0 - v * v, -np.sign(v)) for v in s]  # 後縁: 上面側 → 後縁 → 下面側
            mid = zone + (1.0 - 2.0 * zone) * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 41)))
            up = [nose[-1]] + [pt(v, 1.0) for v in mid[1:-1]] + [tail[0]]
            lo = [tail[-1]] + [pt(v, -1.0) for v in mid[-2:0:-1]] + [nose[0]]
            return occ.addWire([occ.addSpline(nose), occ.addSpline(up), occ.addSpline(tail), occ.addSpline(lo)])

        fin = occ.addThruSections([section(0.0), section(geom["span"])], makeSolid=True, makeRuled=True)
        rad = np.radians(delta_deg)
        occ.rotate(fin, geom["hinge_x"], 0, 0, 0, 1, 0, rad)
        (x0, x1), (z0, z1), y1 = domain["x"], domain["z"], domain["y_max"]
        box = occ.addBox(x0, 0.0, z0, x1 - x0, y1, z1 - z0)
        fluid, _ = occ.cut([(3, box)], fin)
        occ.synchronize()

        fin_s, edge_s, sym_s, far_s = [], [], [], []
        tol = 1e-6
        for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
            bx0, by0, bz0, bx1, by1, bz1 = gmsh.model.getBoundingBox(dim, tag)
            outer = (abs(bx0 - bx1) < tol and min(abs(bx0 - x0), abs(bx0 - x1)) < tol) \
                or (abs(bz0 - bz1) < tol and min(abs(bz0 - z0), abs(bz0 - z1)) < tol) \
                or (abs(by0 - by1) < tol and abs(by0 - y1) < tol)
            if abs(by0) < tol and abs(by1) < tol:
                sym_s.append(tag)
            elif outer:
                far_s.append(tag)
            else:
                fin_s.append(tag)
                # 重心を翼固定座標に戻し、前縁・後縁まわりの面かどうかを判定する
                cx, cy, cz = occ.getCenterOfMass(dim, tag)
                xf = geom["hinge_x"] + np.cos(rad) * (cx - geom["hinge_x"]) - np.sin(rad) * cz
                x_le, chord = planform(geom, cy)
                xi = (xf - x_le) / chord
                if cy < geom["span"] - 1e-4 and (xi < zone or xi > 1.0 - zone):
                    edge_s.append(tag)
        if not (len(sym_s) == 1 and len(far_s) == 5 and len(fin_s) == 5 and len(edge_s) == 2):
            raise RuntimeError(f"境界面の分類に失敗: sym {sym_s}, far {far_s}, fin {fin_s}, edge {edge_s}")
        gmsh.model.addPhysicalGroup(2, fin_s, name="fin")
        gmsh.model.addPhysicalGroup(2, sym_s, name="sym")
        gmsh.model.addPhysicalGroup(2, far_s, name="far")
        gmsh.model.addPhysicalGroup(3, [t for _, t in fluid], name="fluid")

        f = gmsh.model.mesh.field

        def threshold(surfaces, size_min, d_min):
            d = f.add("Distance")
            f.setNumbers(d, "SurfacesList", surfaces)
            f.setNumber(d, "Sampling", 200)
            t = f.add("Threshold")
            f.setNumber(t, "InField", d)
            f.setNumber(t, "SizeMin", size_min)
            f.setNumber(t, "SizeMax", domain["h_far"])
            f.setNumber(t, "DistMin", d_min)
            f.setNumber(t, "DistMax", 3.0)
            return t

        m = f.add("Min")
        f.setNumbers(m, "FieldsList", [threshold(fin_s, h_fin, 0.03), threshold(edge_s, h_edge, 0.002)])
        f.setAsBackgroundMesh(m)
        for name in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
            gmsh.option.setNumber(f"Mesh.{name}", 0)
        gmsh.option.setNumber("Mesh.Algorithm", algo2d)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(out_dir / "mesh.su2"))

        tets = gmsh.model.mesh.getElementsByType(4)[0]
        fin_nodes, n_tri = set(), 0
        for s in fin_s:
            conn = gmsh.model.mesh.getElements(2, s)[2][0]
            n_tri += len(conn) // 3
            fin_nodes.update(conn.tolist())
        quality = gmsh.model.mesh.getElementQualities(tets.tolist(), "minSICN")
        info = dict(delta_deg=delta_deg, h_fin=h_fin, h_edge=h_edge, algo2d=algo2d, threads=threads,
                    nodes=len(gmsh.model.mesh.getNodes()[0]), tets=len(tets), fin_nodes=len(fin_nodes),
                    fin_triangles=n_tri, min_sicn=float(np.min(quality)), gmsh=gmsh.__version__)
    finally:
        gmsh.finalize()
    (out_dir / "mesh.json").write_text(json.dumps(info, indent=1))
    return info
