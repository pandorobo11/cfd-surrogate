import numpy as np
import pytest

from finremap import prism
from finremap.workflow import BL_DEFAULTS, DEFAULTS

GEOM = DEFAULTS["geometry"]


def test_order_loop_follows_edges():
    edges = np.array([[2, 3], [0, 1], [3, 0], [1, 2]])
    assert prism.order_loop(edges).tolist() == [2, 3, 0, 1]
    with pytest.raises(RuntimeError):
        prism.order_loop(np.array([[0, 1], [1, 0], [2, 3], [3, 2]]))


def test_layer_thickness_limits():
    bl = dict(BL_DEFAULTS)
    x = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    t = prism.layer_thickness(x, GEOM, bl, 5000)
    assert t[0] == pytest.approx(bl["t_min"])  # 前縁は下限
    assert t[1] == pytest.approx(bl["thickness_coeff"] * np.sqrt(0.5 / 5000))  # 平板の層流境界層に比例
    assert t[2] == pytest.approx(bl["t_te"])  # 後縁で薄くする


def read_su2(path):
    lines = open(path).read().split("\n")
    i = 0
    out = dict(markers={})
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("NELEM="):
            n = int(ln.split("=")[1])
            out["elems"] = [list(map(int, z.split())) for z in lines[i + 1:i + 1 + n]]
            i += n
        elif ln.startswith("NPOIN="):
            n = int(ln.split("=")[1])
            out["xyz"] = np.array([list(map(float, z.split()[:3])) for z in lines[i + 1:i + 1 + n]])
            i += n
        elif ln.startswith("MARKER_TAG="):
            name = ln.split("=")[1].strip()
            n = int(lines[i + 1].split("=")[1])
            out["markers"][name] = [list(map(int, z.split())) for z in lines[i + 2:i + 2 + n]]
            i += n + 1
        i += 1
    return out


def test_coarse_mesh(tmp_path):
    pytest.importorskip("gmsh")
    from finremap import mesh

    info = mesh.generate(tmp_path, 10.0, GEOM, DEFAULTS["domain"], h_fin=0.08, h_edge=0.01, algo2d=6,
                         boundary_layer=dict(layers=3), reynolds=5000)
    m = read_su2(tmp_path / "mesh.su2")
    kinds = np.array([e[0] for e in m["elems"]])
    assert np.sum(kinds == 13) == info["prisms"] == 3 * info["fin_triangles"]
    assert np.sum(kinds == 10) == info["tets"]
    used = np.zeros(len(m["xyz"]), bool)
    for e in m["elems"]:
        used[e[1:-1]] = True
    assert used.all()
    # 対称面の要素はすべて y=0 にあり、四角形は付け根の辺 × 層の数
    sym = m["markers"]["sym"]
    assert all(np.all(m["xyz"][e[1:], 1] == 0.0) for e in sym)
    n_quad = sum(e[0] == 9 for e in sym)
    assert n_quad % 3 == 0 and n_quad > 0
    assert info["prism_stats"]["min_prism_rel_volume"] > 0.0


def test_prepare_boundary_layer_settings(tmp_path):
    import json

    from finremap import workflow

    base = dict(flow=dict(reynolds=5000), meshes=dict(deflections=[dict(deg=5)], boundary_layer=dict(layers=10)))
    (tmp_path / "a.json").write_text(json.dumps(dict(name="a", **base)))
    plan = json.loads((workflow.prepare(tmp_path / "a.json", skip_image_check=True) / "plan.json").read_text())
    bl = plan["meshes"]["d05"]["boundary_layer"]
    assert bl == {**BL_DEFAULTS, "layers": 10} and plan["meshes"]["ref"]["boundary_layer"] == bl
    (tmp_path / "b.json").write_text(json.dumps(dict(name="b", meshes=base["meshes"])))  # Euler では使えない
    with pytest.raises(ValueError):
        workflow.prepare(tmp_path / "b.json", skip_image_check=True)
    bad = dict(base, meshes=dict(deflections=[dict(deg=5)], boundary_layer=dict(layer=10)))  # 項目名の誤り
    (tmp_path / "c.json").write_text(json.dumps(dict(name="c", **bad)))
    with pytest.raises(ValueError):
        workflow.prepare(tmp_path / "c.json", skip_image_check=True)
