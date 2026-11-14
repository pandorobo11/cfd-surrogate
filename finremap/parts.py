"""一般の表面 VTU で、部位を選んで舵軸まわりに回す処理の雛形。

想定する使い方:
  胴体と複数の操舵翼を含む表面格子から、翼ごとに節点を選び、それぞれの舵軸まわりに舵角ぶん逆回転して
  「基準形状（全舵角 0）」の座標を作る。その座標で remap.map_barycentric を使えば、共通格子へ補間できる。

部位の選び方（Part.select）:
  {"array": "PartId", "values": [2]}           節点データまたはセルデータの値で選ぶ
  {"box": [[x0, y0, z0], [x1, y1, z1]]}       直方体の中の節点を選ぶ
  {"component_near": [x, y, z]}               その点に最も近い節点を含む、つながった面を選ぶ
  {"all": true}                               全節点

注意:
  - SU2 の surface_flow.vtu は、境界の名前を持たない。翼と胴体が離れていれば component_near で分けられる。
    翼の付け根が胴体と節点を共有しているときは、box で選ぶか、部位番号の配列を別に用意する。
  - 舵角によって胴体側の濡れ面が変わる部分（付け根の隙間など）は、回しても形状が一致しない。
    その領域は補間後の投影距離（map_barycentric の戻り値）が大きくなるので、しきい値で除外する。

コマンドとしての使い方:
  python -m finremap.parts <入力.vtu> <部位の定義.json> <出力.vtu>
  部位の定義は Part の一覧（下の parts_from_json を参照）。出力には、基準形状の座標 ReferencePosition と
  部位番号 PartIndex（どの部位にも入らない節点は -1）を節点データとして加える。
"""
import json
import sys
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .remap import rotate
from .vtu import read_vtu, write_vtu


@dataclass
class Part:
    name: str
    select: dict
    origin: tuple = (0.0, 0.0, 0.0)  # 舵軸が通る点
    axis: tuple = (0.0, 1.0, 0.0)  # 舵軸の向き
    angle_deg: float = 0.0  # この格子での舵角（右ねじの向きを正）


def point_components(mesh):
    """セルでつながった節点のかたまりごとの番号 (N,)。"""
    start = np.concatenate([[0], mesh.offsets[:-1]])
    first = np.repeat(mesh.connectivity[start], mesh.offsets - start)
    n = len(mesh.points)
    graph = coo_matrix((np.ones(len(first)), (first, mesh.connectivity)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def select_points(mesh, select):
    """部位の選び方から、節点のマスク (N,) を作る。"""
    n = len(mesh.points)
    if select.get("all"):
        return np.ones(n, bool)
    if "array" in select:
        name, values = select["array"], np.asarray(select["values"])
        if name in mesh.point_data:
            return np.isin(mesh.point_data[name], values)
        if name not in mesh.cell_data:
            raise KeyError(f"配列 {name} が節点データにもセルデータにもない")
        cells = np.isin(mesh.cell_data[name], values)
        start = np.concatenate([[0], mesh.offsets[:-1]])
        mask = np.zeros(n, bool)
        mask[mesh.connectivity[np.repeat(cells, mesh.offsets - start)]] = True
        return mask
    if "box" in select:
        lo, hi = np.asarray(select["box"], float)
        return np.all((mesh.points >= lo) & (mesh.points <= hi), axis=1)
    if "component_near" in select:
        comp = point_components(mesh)
        seed = cKDTree(mesh.points).query(np.asarray(select["component_near"], float))[1]
        return comp == comp[seed]
    raise ValueError(f"部位の選び方が分からない: {select}")


def to_reference(mesh, parts):
    """各部位を舵軸まわりに -angle_deg 回した座標 (N, 3) と、部位番号 (N,) を返す。

    複数の部位に入る節点があればエラーにする（どちらの軸で回すか決まらないため）。
    """
    ref = mesh.points.copy()
    index = np.full(len(ref), -1)
    for i, part in enumerate(parts):
        mask = select_points(mesh, part.select)
        if not mask.any():
            raise ValueError(f"部位 {part.name}: 節点が 1 つも選ばれなかった")
        if np.any(index[mask] >= 0):
            other = parts[int(index[mask].max())].name
            raise ValueError(f"部位 {part.name} と {other} が同じ節点を含む")
        index[mask] = i
        ref[mask] = rotate(mesh.points[mask], part.origin, part.axis, -part.angle_deg)
    return ref, index


def parts_from_json(path):
    """[{"name": ..., "select": {...}, "origin": [...], "axis": [...], "angle_deg": ...}, ...] を読む。"""
    return [Part(**p) for p in json.loads(open(path).read())]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        sys.exit(__doc__)
    mesh = read_vtu(argv[0])
    parts = parts_from_json(argv[1])
    ref, index = to_reference(mesh, parts)
    mesh.point_data["ReferencePosition"] = ref
    mesh.point_data["PartIndex"] = index.astype(np.int32)
    write_vtu(argv[2], mesh)
    for i, part in enumerate(parts):
        print(f"{part.name}: {int((index == i).sum())} 節点を {-part.angle_deg:+g} deg 回転")
    print(f"どの部位にも入らない節点: {int((index < 0).sum())}")


if __name__ == "__main__":
    main()
