import json

import numpy as np
import pytest

from finremap import parts, remap, workflow
from finremap.vtu import Mesh, read_vtu, triangle_mesh, write_vtu


def grid(nx, ny, z=0.0, x0=0.0, seed=None):
    """[x0, x0+1] x [0, 1] の平面を三角形に分けた格子。seed を与えると内部の節点を乱す。"""
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
    if seed is not None:
        rng = np.random.default_rng(seed)
        x[1:-1, 1:-1] += rng.uniform(-0.3, 0.3, (nx - 2, ny - 2)) / (nx - 1)
        y[1:-1, 1:-1] += rng.uniform(-0.3, 0.3, (nx - 2, ny - 2)) / (ny - 1)
    pts = np.stack([x.ravel() + x0, y.ravel(), np.full(x.size, z)], axis=1)
    i = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = i[:-1, :-1].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel(), i[:-1, 1:].ravel()
    return pts, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def test_vtu_roundtrip(tmp_path):
    pts, tri = grid(4, 3)
    mesh = triangle_mesh(pts, tri, dict(Cp=np.arange(len(pts), dtype=np.float32), V=pts * 2.0),
                         dict(Part=np.arange(len(tri), dtype=np.int32)))
    write_vtu(tmp_path / "a.vtu", mesh)
    back = read_vtu(tmp_path / "a.vtu")
    assert np.array_equal(back.points, pts) and np.array_equal(back.triangles()[0], tri)
    assert np.array_equal(back.point_data["Cp"], mesh.point_data["Cp"]) and back.point_data["V"].shape == (len(pts), 3)
    assert np.array_equal(back.cell_data["Part"], np.arange(len(tri)))


def test_quads_are_split():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    mesh = Mesh(pts, np.arange(4), np.array([4]), np.array([9]))
    tri, cell = mesh.triangles()
    assert tri.tolist() == [[0, 1, 2], [0, 2, 3]] and cell.tolist() == [0, 0]


@pytest.mark.parametrize("mode", ["ascii", "binary", "appended"])
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("header", ["UInt32", "UInt64"])
@pytest.mark.parametrize("encode_appended", [False, True])
def test_reads_files_written_by_vtk(tmp_path, mode, compress, header, encode_appended):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import numpy_to_vtk

    if mode != "appended" and encode_appended:
        pytest.skip("appended 以外では意味がない")
    pts, tri = grid(7, 5, seed=0)
    cp = np.sin(pts[:, 0] * 3) + pts[:, 1]
    g = vtk.vtkUnstructuredGrid()
    p = vtk.vtkPoints()
    p.SetData(numpy_to_vtk(pts, deep=True))
    g.SetPoints(p)
    for t in tri:
        g.InsertNextCell(vtk.VTK_TRIANGLE, 3, [int(v) for v in t])
    arr = numpy_to_vtk(cp, deep=True)
    arr.SetName("Cp")
    g.GetPointData().AddArray(arr)
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(tmp_path / "v.vtu"))
    w.SetInputData(g)
    {"ascii": w.SetDataModeToAscii, "binary": w.SetDataModeToBinary, "appended": w.SetDataModeToAppended}[mode]()
    w.SetEncodeAppendedData(encode_appended)
    w.SetCompressorTypeToZLib() if compress else w.SetCompressorTypeToNone()
    w.SetHeaderTypeToUInt32() if header == "UInt32" else w.SetHeaderTypeToUInt64()
    assert w.Write() == 1
    mesh = read_vtu(tmp_path / "v.vtu")
    assert np.allclose(mesh.points, pts, atol=1e-12) and np.array_equal(mesh.triangles()[0], tri)
    assert np.allclose(mesh.point_data["Cp"], cp, atol=1e-12)


def test_rotate():
    assert np.allclose(remap.rotate([[1.0, 0, 0]], [0, 0, 0], [0, 1, 0], 90), [[0, 0, -1]])  # 前縁側 (x<軸) は +z へ
    pts = np.random.default_rng(0).normal(size=(20, 3))
    there = remap.rotate(pts, [0.4, 0.1, -0.2], [1, 2, 3], 37.0)
    assert np.allclose(remap.rotate(there, [0.4, 0.1, -0.2], [1, 2, 3], -37.0), pts)
    assert np.allclose(np.linalg.norm(there - [0.4, 0.1, -0.2], axis=1), np.linalg.norm(pts - [0.4, 0.1, -0.2], axis=1))


def test_barycentric_is_exact_for_linear_fields():
    src_p, src_t = grid(9, 7, seed=1)
    qry_p, _ = grid(13, 11, seed=2)
    field = lambda p: 2.0 * p[:, 0] - 3.0 * p[:, 1] + 0.5
    val, dist = remap.map_barycentric(src_p, src_t, field(src_p), qry_p)
    assert np.allclose(val, field(qry_p), atol=1e-12) and dist.max() < 1e-12
    assert np.abs(remap.map_nearest(src_p, field(src_p), qry_p) - field(qry_p)).max() > 0.05


def test_normal_check_picks_the_right_side_of_a_thin_plate():
    top_p, top_t = grid(6, 6, z=+1e-4)
    bot_p, bot_t = grid(6, 6, z=-1e-4)
    pts = np.concatenate([top_p, bot_p])
    tri = np.concatenate([top_t, bot_t[:, [0, 2, 1]] + len(top_p)])  # 下面は法線を -z に向ける
    values = np.concatenate([np.ones(len(top_p)), -np.ones(len(bot_p))])
    q = np.array([[0.31, 0.47, -5e-5]])  # 上面の点だが、下面のほうが近い位置にある
    assert remap.map_barycentric(pts, tri, values, q)[0][0] == pytest.approx(-1.0)
    assert remap.map_barycentric(pts, tri, values, q, np.array([[0, 0, 1.0]]))[0][0] == pytest.approx(1.0)


def test_forces_on_a_flat_plate():
    pts, tri = grid(5, 5)
    f, m = remap.integrate_forces(pts, tri, np.ones(len(pts)), 1.0, [0.5, 0.5, 0.0])
    assert np.allclose(f, [0, 0, -1]) and np.allclose(m, 0, atol=1e-12)  # 法線 +z の面に Cp=1


def two_part_mesh(angle):
    """離れた 2 枚の面。2 枚目だけを軸 (x=2.5, z=0, y 方向) のまわりに angle 回してある。"""
    p1, t1 = grid(4, 4)
    p2, t2 = grid(4, 4, x0=2.0)
    moved = remap.rotate(p2, [2.5, 0, 0], [0, 1, 0], angle)
    mesh = triangle_mesh(np.concatenate([p1, moved]), np.concatenate([t1, t2 + len(p1)]),
                         cell_data=dict(Part=np.repeat([1, 2], [len(t1), len(t2)])))
    return mesh, np.concatenate([p1, p2])


@pytest.mark.parametrize("select", [dict(component_near=[2.5, 0.5, 0.0]), dict(array="Part", values=[2]),
                                    dict(box=[[1.5, -1, -1], [3.5, 2, 1]])])
def test_part_is_rotated_back(select):
    mesh, original = two_part_mesh(20.0)
    ref, index = parts.to_reference(mesh, [parts.Part("fin", select, origin=(2.5, 0, 0), axis=(0, 1, 0), angle_deg=20.0)])
    assert np.allclose(ref, original) and (index == 0).sum() == 16 and (index == -1).sum() == 16


def test_overlapping_parts_are_rejected():
    mesh, _ = two_part_mesh(0.0)
    with pytest.raises(ValueError):
        parts.to_reference(mesh, [parts.Part("a", dict(all=True)), parts.Part("b", dict(box=[[0, 0, -1], [1, 1, 1]]))])


def test_parts_command(tmp_path):
    mesh, original = two_part_mesh(15.0)
    write_vtu(tmp_path / "in.vtu", mesh)
    (tmp_path / "parts.json").write_text(json.dumps([dict(name="fin", select=dict(component_near=[2.5, 0.5, 0]),
                                                         origin=[2.5, 0, 0], axis=[0, 1, 0], angle_deg=15.0)]))
    parts.main([str(tmp_path / "in.vtu"), str(tmp_path / "parts.json"), str(tmp_path / "out.vtu")])
    out = read_vtu(tmp_path / "out.vtu")
    assert np.allclose(out.point_data["ReferencePosition"], original) and set(out.point_data["PartIndex"]) == {-1, 0}


def make_run(tmp_path, **execution):
    cfg = dict(name="t", output_root="runs", meshes=dict(deflections=[dict(deg=5, h_fin=0.02), dict(deg=7.5)]),
               execution=dict(container=dict(image=None), **execution))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    return workflow.prepare(tmp_path / "c.json")


def test_prepare_builds_cases_and_refuses_overwrite(tmp_path):
    run = make_run(tmp_path)
    plan = json.loads((run / "plan.json").read_text())
    assert plan["order"] == ["ref-a05", "d05-a00", "ref-a07p5", "d07p5-a00"]
    assert plan["meshes"]["d05"]["h_fin"] == 0.02 and plan["meshes"]["d07p5"]["h_fin"] == plan["meshes"]["ref"]["h_fin"]
    assert "AOA= 7.5" in (run / "cases" / "ref-a07p5" / "case.cfg").read_text()
    assert "AOA= 0.0" in (run / "cases" / "d07p5-a00" / "case.cfg").read_text()
    assert (run / "scripts" / "finremap" / "workflow.py").exists() and plan["execution"]["container"] is None
    with pytest.raises(ValueError):
        workflow.prepare(tmp_path / "c.json")


def test_unknown_setting_is_rejected(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps(dict(name="t", solver=dict(itr=10), meshes=dict(deflections=[dict(deg=5)]))))
    with pytest.raises(ValueError):
        workflow.prepare(tmp_path / "c.json")


def test_job_script_and_container_command(tmp_path):
    run = make_run(tmp_path, setup=["module load mpi"], pbs=dict(queue="q8", job_prefix="f"))
    plan = json.loads((run / "plan.json").read_text())
    script = workflow.job_script(plan, run, "d05-a00")
    assert "#PBS -N f002" in script and "#PBS -q q8" in script and "module load mpi" in script
    assert f"export PYTHONPATH={run}/scripts" in script
    assert script.rstrip().endswith(f"-m finremap.workflow run-case {run} d05-a00")
    assert workflow.job_script(plan, run, None).rstrip().endswith(f"-m finremap.workflow analyze {run}")
    ex = dict(plan["execution"], container=dict(engine="podman", image_id="sha256:abc", options=["--shm-size=2g"]))
    cmd, name = workflow.solver_command(ex, run / "cases" / "d05-a00", run, "t-d05-a00")
    assert cmd[:3] == ["podman", "run", "--rm"] and name.startswith("su2-t-d05-a00-")
    assert f"{run}:{run}" in cmd and cmd[-6:] == ["sha256:abc", "mpirun", "-np", "8", "SU2_CFD", "case.cfg"]
    host, none = workflow.solver_command(plan["execution"], run, run, "x")
    assert host == ["mpirun", "-np", "8", "SU2_CFD", "case.cfg"] and none is None


def test_submit_needs_meshes_and_started_case_is_not_rerun(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(ValueError, match="格子"):
        workflow.submit(run, backend="local")
    (run / "cases" / "d05-a00" / "state.json").write_text('{"status": "running"}')
    with pytest.raises(ValueError, match="既に始まっている"):
        workflow.run_case(run, "d05-a00")
