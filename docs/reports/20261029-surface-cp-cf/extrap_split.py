"""Split surfcpcf evaluation errors by side of the training range (Reynolds or angle of attack).

Usage: python extrap_split.py <airfrans_surface> <run_dir> {re|aoa}
"""
import csv, json, sys
import numpy as np

surface, run, key = sys.argv[1:4]
cases = json.load(open(f"{surface}/index.json"))["cases"]
metrics = json.load(open(f"{run}/eval/metrics.json"))
lo, hi = metrics["training_range"]["reynolds_unit_chord" if key == "re" else "aoa_deg"]
rows = list(csv.DictReader(open(f"{run}/eval/per_case.csv")))
value = np.array([cases[r["case"]]["uinf"] / cases[r["case"]]["nu"] if key == "re" else cases[r["case"]]["aoa_deg"]
                  for r in rows])
med = lambda rs, f: float(np.median([f(r) for r in rs]))
for side, mask in (("below", value < lo), ("above", value > hi), ("inside", (value >= lo) & (value <= hi))):
    rs = [r for r, m in zip(rows, mask) if m]
    if not rs:
        continue
    print(side, len(rs),
          "cp_rel_l2 %.3f" % med(rs, lambda r: float(r["cp_relative_l2"])),
          "cl_rel %.3f" % med(rs, lambda r: float(r["cl_relative_error"])),
          "cl_abs %.4f" % med(rs, lambda r: abs(float(r["pred_cl"]) - float(r["true_cl"]))),
          "cd_abs %.5f" % med(rs, lambda r: abs(float(r["pred_cd"]) - float(r["true_cd"]))),
          "true_cd %.4f" % med(rs, lambda r: float(r["true_cd"])))
distance = np.where(value < lo, lo - value, np.where(value > hi, value - hi, 0.)) / (hi - lo)
for a, b in ((0, .1), (.1, .2), (.2, .3), (.3, np.inf)):
    rs = [r for r, d in zip(rows, distance) if a < d <= b]
    if rs:
        print(f"distance {a:g}-{b:g} of range", len(rs),
              "cp_rel_l2 %.3f" % med(rs, lambda r: float(r["cp_relative_l2"])),
              "cl_rel %.3f" % med(rs, lambda r: float(r["cl_relative_error"])))
