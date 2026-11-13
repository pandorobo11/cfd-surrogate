"""前縁と後縁の両方に丸みを付けた全遊動翼単体のEuler用四面体格子を、舵角ごとに生成する。

翼型（翼弦で無次元化）: z = ±(a0 sqrt(xi (1 - xi)) + b xi (1 - xi))
  前縁半径と後縁半径はともに r/c = a0^2 / 2、b は最大翼厚比が T_MAX になるように決める。
翼は対称面 (y=0) に取り付ける。翼固定座標: x=翼弦, y=翼幅, z=翼厚。
舵軸は y 軸に平行で (x, z) = (HINGE_X, 0) を通る。舵角は y 軸まわりの右ねじ回転で、正のとき前縁が +z 側に上がる
（翼単体では一様流の迎角と等価）。

使い方: python gen_mesh.py <delta_deg> <out_dir> [--h-fin 0.012] [--h-le 0.001] [--algo2d 6]
"""
import argparse
import json
from pathlib import Path

import gmsh
import numpy as np

HINGE_X = 0.45
SPAN = 0.8
C_ROOT, C_TIP, X_LE_TIP = 1.0, 0.5, 0.35
T_MAX = 0.06
LE_RADIUS = 0.005  # 前縁半径・後縁半径 / 局所翼弦長
XI_NOSE = 0.03  # 前縁・後縁まわりを別の面にする範囲
H_FAR = 0.5
BOX = dict(x0=-2.0, x1=5.0, y1=3.0, z0=-3.0, z1=3.0)


def thickness_coeffs():
    a0 = np.sqrt(2.0 * LE_RADIUS)
    xi = np.linspace(0.0, 1.0, 20001)
    lo, hi = -1.0, 1.0
    for _ in range(80):
        b = 0.5 * (lo + hi)
        if np.max(a0 * np.sqrt(xi * (1 - xi)) + b * xi * (1 - xi)) > 0.5 * T_MAX:
            hi = b
        else:
            lo = b
    return float(a0), float(b)


A0, B = thickness_coeffs()


def half_thickness(xi):
    return A0 * np.sqrt(np.clip(xi * (1.0 - xi), 0.0, None)) + B * xi * (1.0 - xi)


def section_wire(y, x_le, chord):
    occ = gmsh.model.occ
    pt = lambda xi, sign: occ.addPoint(x_le + xi * chord, y, sign * half_thickness(xi) * chord)
    s = np.linspace(-np.sqrt(XI_NOSE), np.sqrt(XI_NOSE), 41)  # xi = s^2 で丸みを均等に標本化
    nose = [pt(v * v, np.sign(v)) for v in s]  # 前縁: 下面側 → 前縁 → 上面側
    tail = [pt(1.0 - v * v, -np.sign(v)) for v in s]  # 後縁: 上面側 → 後縁 → 下面側
    xi_mid = XI_NOSE + (1.0 - 2.0 * XI_NOSE) * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 41)))
    up = [nose[-1]] + [pt(v, 1.0) for v in xi_mid[1:-1]] + [tail[0]]
    lo = [tail[-1]] + [pt(v, -1.0) for v in xi_mid[-2:0:-1]] + [nose[0]]
    return occ.addWire([occ.addSpline(nose), occ.addSpline(up), occ.addSpline(tail), occ.addSpline(lo)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("delta_deg", type=float)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--h-fin", type=float, default=0.012)
    ap.add_argument("--h-le", type=float, default=0.001)
    ap.add_argument("--algo2d", type=int, default=6)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    fin = occ.addThruSections([section_wire(0.0, 0.0, C_ROOT), section_wire(SPAN, X_LE_TIP, C_TIP)],
                              makeSolid=True, makeRuled=True)
    rad = np.radians(a.delta_deg)
    occ.rotate(fin, HINGE_X, 0, 0, 0, 1, 0, rad)
    box = occ.addBox(BOX["x0"], 0.0, BOX["z0"], BOX["x1"] - BOX["x0"], BOX["y1"], BOX["z1"] - BOX["z0"])
    fluid, _ = occ.cut([(3, box)], fin)
    occ.synchronize()

    fin_s, nose_s, sym_s, far_s = [], [], [], []
    tol = 1e-6
    for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
        on_outer = (abs(x0 - x1) < tol and (abs(x0 - BOX["x0"]) < tol or abs(x0 - BOX["x1"]) < tol)) \
            or (abs(z0 - z1) < tol and (abs(z0 - BOX["z0"]) < tol or abs(z0 - BOX["z1"]) < tol)) \
            or (abs(y0 - y1) < tol and abs(y0 - BOX["y1"]) < tol)
        if abs(y0) < tol and abs(y1) < tol:
            sym_s.append(tag)
        elif on_outer:
            far_s.append(tag)
        else:
            fin_s.append(tag)
            # 重心を翼固定座標に戻し、前縁・後縁まわりの面かどうかを判定する
            cx, cy, cz = occ.getCenterOfMass(dim, tag)
            dx, dz = cx - HINGE_X, cz
            xf = HINGE_X + np.cos(rad) * dx - np.sin(rad) * dz
            eta = cy / SPAN
            xi = (xf - X_LE_TIP * eta) / (C_ROOT + (C_TIP - C_ROOT) * eta)
            if cy < SPAN - 1e-4 and (xi < XI_NOSE or xi > 1.0 - XI_NOSE):
                nose_s.append(tag)
    assert len(sym_s) == 1 and len(far_s) == 5 and len(fin_s) == 5 and len(nose_s) == 2, (sym_s, far_s, fin_s, nose_s)
    gmsh.model.addPhysicalGroup(2, fin_s, name="fin")
    gmsh.model.addPhysicalGroup(2, sym_s, name="sym")
    gmsh.model.addPhysicalGroup(2, far_s, name="far")
    gmsh.model.addPhysicalGroup(3, [t for _, t in fluid], name="fluid")

    f = gmsh.model.mesh.field

    def threshold(surfaces, size_min, size_max, d_min, d_max):
        d = f.add("Distance")
        f.setNumbers(d, "SurfacesList", surfaces)
        f.setNumber(d, "Sampling", 200)
        t = f.add("Threshold")
        f.setNumber(t, "InField", d)
        f.setNumber(t, "SizeMin", size_min)
        f.setNumber(t, "SizeMax", size_max)
        f.setNumber(t, "DistMin", d_min)
        f.setNumber(t, "DistMax", d_max)
        return t

    t_fin = threshold(fin_s, a.h_fin, H_FAR, 0.03, 3.0)
    t_le = threshold(nose_s, a.h_le, H_FAR, 0.002, 3.0)  # 遠方では t_fin と同じ上限に戻す
    m = f.add("Min")
    f.setNumbers(m, "FieldsList", [t_fin, t_le])
    f.setAsBackgroundMesh(m)
    for name in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{name}", 0)
    gmsh.option.setNumber("Mesh.Algorithm", a.algo2d)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("General.NumThreads", 8)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(a.out_dir / "mesh.su2"))

    tets = gmsh.model.mesh.getElementsByType(4)[0]
    fin_nodes, n_tri = set(), 0
    for s in fin_s:
        conn = gmsh.model.mesh.getElements(2, s)[2][0]
        n_tri += len(conn) // 3
        fin_nodes.update(conn.tolist())
    q = gmsh.model.mesh.getElementQualities(tets.tolist(), "minSICN")
    info = dict(delta_deg=a.delta_deg, hinge_x=HINGE_X, hinge_axis="y", span=SPAN, t_max=T_MAX,
                le_radius_over_chord=LE_RADIUS, te_radius_over_chord=LE_RADIUS, a0=A0, b=B, xi_nose=XI_NOSE, h_fin=a.h_fin, h_le=a.h_le,
                algo2d=a.algo2d, h_far=H_FAR, box=BOX, nodes=len(gmsh.model.mesh.getNodes()[0]),
                tets=len(tets), fin_nodes=len(fin_nodes), fin_triangles=n_tri, min_sicn=float(np.min(q)),
                gmsh=gmsh.__version__)
    (a.out_dir / "mesh.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("delta_deg", "a0", "b", "nodes", "tets", "fin_nodes", "fin_triangles", "min_sicn")}))
    gmsh.finalize()


if __name__ == "__main__":
    main()
