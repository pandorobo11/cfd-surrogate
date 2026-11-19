"""箱ごと回した格子の解（rot05-a00）で、圧力の差が座標系によるのか、箱の向きによるのかを切り分ける。

rot05 は ref の全節点（計算領域の箱を含む）を舵軸まわりに 5° 回した格子（rotate_mesh.py）。rot05-a00 は迎角 0 で解く。
翼固定座標で見ると:
  rot05-a00 と ref-a05 は、格子も箱の向きも同じで、ソルバーの座標系（一様流が x 軸に沿うか、傾いているか）だけが違う
  rot05-a00 と d05-a00 は、座標系が同じで、箱の向きと格子（翼のまわりの格子の作り方。ref2-a05 で影響は小さいと分かっている）が違う

rot05-a00 の翼面は、回転を戻すと ref の翼面と同じ節点になるので、analyze と同じ補間は恒等写像になる（投影距離で確かめる）。
3 つの Cp・Cf を ref の翼面に並べ、力と場所ごとの差を比べる。analysis/remapped-rot05-a00.vtu も書く。

使い方（リポジトリの直下で。実行フォルダの scripts/ を使う）:
  PYTHONPATH=<実行フォルダ>/scripts python3 docs/reports/20261119-fin-cf-remap-prism/frame_check.py <実行フォルダ> [接尾辞] > frame_check.json
接尾辞（例 -nolim）を付けると、rot05-a00<接尾辞> と ref-a05<接尾辞> を比べる。d05-a00<接尾辞> がなければ d05 との比較は省く。
"""
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

from finremap import remap
from finremap.analysis import analyze_pair
from finremap.mesh import ref_area
from finremap.vtu import read_vtu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cp_gap_breakdown import breakdown_fields  # noqa: E402

KEEP = ["CL", "CD", "CMy", "CFz", "rms_rho", "iter"]

if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve()
    sfx = sys.argv[2] if len(sys.argv) > 2 else ""
    PAIR = dict(source=f"rot05-a00{sfx}", reference=f"ref-a05{sfx}", delta_deg=5.0)
    plan = json.loads((run / "plan.json").read_text())
    geom = plan["geometry"]
    with contextlib.redirect_stdout(sys.stderr):  # analyze_pair の表は標準エラーへ
        rows = analyze_pair(run, plan, PAIR, None)
    rot = read_vtu(run / "analysis" / f"remapped-rot05-a00{sfx}.vtu")
    xyz, tri = rot.points, rot.triangles()[0]
    cp = dict(ref_a05=rot.point_data["Cp_reference"], rot05_a00=rot.point_data["Cp_mapped"])
    cf = dict(ref_a05=rot.point_data["Cf_reference"], rot05_a00=rot.point_data["Cf_mapped"])
    has_d05 = (run / "analysis" / f"remapped-d05-a00{sfx}.vtu").exists()
    if has_d05:
        d05 = read_vtu(run / "analysis" / f"remapped-d05-a00{sfx}.vtu")
        if not (np.array_equal(xyz, d05.points) and np.array_equal(tri, d05.triangles()[0])):
            raise RuntimeError("2 つの VTU の基準格子が違う")
        cp["d05_a00"], cf["d05_a00"] = d05.point_data["Cp_mapped"], d05.point_data["Cf_mapped"]
    cp = {k: v.astype(float) for k, v in cp.items()}
    cf = {k: v.astype(float) for k, v in cf.items()}
    area, origin = ref_area(geom), [geom["hinge_x"], 0.0, 0.0]
    forces = {}
    for k in cp:
        f, m = remap.integrate_forces(xyz, tri, cp[k], area, origin)
        ff, _ = remap.integrate_friction(xyz, tri, cf[k], area, origin)
        forces[k] = dict(CN=round(float(f[2]), 6), CA=round(float(f[0]), 6), CMy=round(float(m[1]), 6),
                         CA_f=round(float(ff[0]), 6))
    rms = lambda v: round(float(np.sqrt(np.mean(v**2))), 5)
    cf_scale = float(np.sqrt(np.mean(np.sum(cf["ref_a05"] ** 2, axis=1))))
    comparisons = {}
    for a, b, meaning in [("rot05_a00", "ref_a05", "座標系だけが違う"),
                          ("d05_a00", "rot05_a00", "箱の向きと格子が違う（座標系は同じ）"),
                          ("d05_a00", "ref_a05", "これまでの比較（両方が違う）")]:
        if a not in cp or b not in cp:
            continue
        pct = lambda q: round(100.0 * (forces[a][q] / forces[b][q] - 1.0), 3)
        comparisons[f"{a} vs {b}"] = dict(
            meaning=meaning, CN_pct=pct("CN"), CA_pct=pct("CA"), CMy_pct=pct("CMy"), CA_f_pct=pct("CA_f"),
            cp_diff_rms=rms(cp[a] - cp[b]),
            cf_diff_rms_over_cf_rms_pct=round(100.0 * float(np.sqrt(np.mean(np.sum((cf[a] - cf[b]) ** 2, axis=1)))) / cf_scale, 3),
            breakdown=breakdown_fields(xyz, tri, cp[a], cp[b], geom))
    cases = {}
    for cid in [PAIR["source"], PAIR["reference"]] + ([f"d05-a00{sfx}"] if has_d05 else []):
        r = json.loads((run / "cases" / cid / "run.json").read_text())
        cases[cid] = dict(mesh=plan["cases"][cid]["mesh"], aoa_deg=plan["cases"][cid]["aoa_deg"], outcome=r["outcome"],
                          ranks=r["ranks"], wall_s=r["wall_s"], final={k: r["final"].get(k) for k in KEEP},
                          forces_breakdown=r.get("forces_breakdown"))
    proj = next(r for r in rows if r["method"].startswith("元格子"))
    json.dump(dict(pair=PAIR, suffix=sfx, cases=cases, projection_distance_max=proj["proj_dist_max"], forces_on_ref_surface=forces,
                   comparisons=comparisons), sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
