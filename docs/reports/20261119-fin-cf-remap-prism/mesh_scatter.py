"""同じ向き（舵角 0、迎角 5°）で、格子だけが違う 2 つの解を比べる（格子による差の大きさの切り分け）。

ref2-a05 の翼面の Cp・Cf を、finremap の analyze と同じ手順で ref-a05 の翼面へ補間し（舵角 0 なので回転はない）、
ref-a05 の解と比べる。analysis/remapped-ref2-a05.vtu も書く（cp_gap_breakdown.py で場所ごとに分けられる）。

使い方（リポジトリの直下で。実行フォルダの scripts/ を使う）:
  PYTHONPATH=<実行フォルダ>/scripts python3 docs/reports/20261119-fin-cf-remap-prism/mesh_scatter.py <実行フォルダ> > mesh_scatter.json
"""
import contextlib
import json
import sys
from pathlib import Path

from finremap.analysis import analyze_pair

PAIR = dict(source="ref2-a05", reference="ref-a05", delta_deg=0.0)
KEEP = ["CL", "CD", "CMy", "CFz", "rms_rho", "iter"]

if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve()
    plan = json.loads((run / "plan.json").read_text())
    with contextlib.redirect_stdout(sys.stderr):  # analyze_pair の表は標準エラーへ
        rows = analyze_pair(run, plan, PAIR, None)
    cases = {}
    for cid in (PAIR["source"], PAIR["reference"]):
        r = json.loads((run / "cases" / cid / "run.json").read_text())
        m = json.loads((run / "meshes" / plan["cases"][cid]["mesh"] / "mesh.json").read_text())
        cases[cid] = dict(mesh=plan["cases"][cid]["mesh"], h_fin=m["h_fin"], h_edge=m["h_edge"], nodes=m["nodes"],
                          fin_nodes=m["fin_nodes"], min_sicn=round(m["min_sicn"], 4), outcome=r["outcome"],
                          ranks=r["ranks"], wall_s=r["wall_s"], final={k: r["final"].get(k) for k in KEEP},
                          forces_breakdown=r.get("forces_breakdown"))
    json.dump(dict(pair=PAIR, cases=cases, rows=rows), sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
