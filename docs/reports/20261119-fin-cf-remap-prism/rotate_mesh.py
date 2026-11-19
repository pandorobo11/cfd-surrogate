"""SU2 の格子（ASCII）の全節点を、舵軸まわりに回した格子を書く。要素と境界はそのまま写す。

計算領域の箱ごと回すので、翼・格子・箱の相対的な位置は元の格子と全く同じになる。舵角を付けて作った格子
（mesh.py の舵角）と同じ向き（y 軸まわりの右ねじ、正で前縁が +z 側に上がる）に回す。y 座標は変わらないので、
対称面 (y=0) はそのまま残る。

使い方:
  python3 rotate_mesh.py <元の mesh.su2> <書き出す mesh.su2> <角度 [deg]> [舵軸の x（既定 0.45）]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from finremap.remap import rotate  # noqa: E402


def main(src, dst, angle_deg, hinge_x=0.45):
    with open(src) as fin, open(dst, "w") as out:
        for line in fin:
            out.write(line)
            if line.startswith("NPOIN="):
                n = int(line.split("=")[1].split()[0])
                rows = [fin.readline() for _ in range(n)]
                data = np.loadtxt(rows)
                xyz = rotate(data[:, :3], [hinge_x, 0.0, 0.0], [0.0, 1.0, 0.0], angle_deg)
                y0 = data[:, 1] == 0.0
                xyz[y0, 1] = 0.0  # 対称面の節点は y=0 のまま（丸め誤差を残さない）
                np.savetxt(out, np.column_stack([xyz, data[:, 3]]), fmt="%.17g %.17g %.17g %d")
    return n, int(y0.sum())


if __name__ == "__main__":
    a = sys.argv[1:]
    n, n_sym = main(a[0], a[1], float(a[2]), float(a[3]) if len(a) > 3 else 0.45)
    print(f"{n} 節点を {a[2]} deg 回した（y=0 の節点 {n_sym}）")
