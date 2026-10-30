# Surface-only Cp/Cf Transolver

SURF-01 uses every point of the AirfRANS airfoil patch to predict Cp and signed Cf.
This is a surface-only surrogate setting, distinct from the published volume-field
benchmark. Predictions are not validated flow solutions. The full 200-epoch run
requires review; the smoke commands below run only two epochs.

Use the existing benchmark Python environment (no installation is needed). Run
from the repository root; every data/source/output path can be supplied by CLI.
The commands use relative paths and do not download data or upstream code.

```sh
PY=local/venv-bench/bin/python
$PY -m pytest -q tests
$PY -m surfcpcf.preprocess --dataset data/airfrans/Dataset --out data/airfrans_surface --splits scarce_train full_test
$PY -m surfcpcf.checks --surface data/airfrans_surface --dataset data/airfrans/Dataset --transolver-dir external/Transolver/Airfoil-Design-AirfRANS --out results/surface/checks.json
$PY -m surfcpcf.train --task scarce --surface data/airfrans_surface --transolver-dir external/Transolver/Airfoil-Design-AirfRANS --out results/surface/SMOKE --n-hidden 128 --n-layers 5 --slice-num 32 --epochs 2 --val-every 1
$PY -m surfcpcf.evaluate --run results/surface/SMOKE --surface data/airfrans_surface --transolver-dir external/Transolver/Airfoil-Design-AirfRANS
```

`--device mps|cpu` overrides automatic MPS-else-CPU selection in checks, training,
and evaluation. Every run records the device and the environment value of
`PYTORCH_ENABLE_MPS_FALLBACK`; this package does not enable fallback itself.
The model retains official grid extents and `unified_pos=1`, with unnormalized
coordinates for grid distances. It loads upstream code at runtime without
copying or editing it. `--model mlp` trains the point MLP baseline instead
(`--n-hidden`, `--n-layers` set its width and depth; `--slice-num` is ignored);
evaluation rebuilds whichever model the run config names.

Preprocessing writes one compressed NPZ per case plus `index.json`, which retains
the official manifest and metadata. Existing files are skipped; `--overwrite`
regenerates them. `--limit N` is for debugging only, and an incomplete split
cannot be used for training/evaluation. Array order is the VTP patch point order.
`cell_normal` is the **point-averaged normal from `airfoil.ptc()`**, matching
AirfRANS force integration, not the original cell normal. Schema version 2
identifies this convention. Regenerate files produced before this convention.
`reference` has order `cd, cdp, cdv, cl, clp, clv`; angles are saved in radians
(`aoa_rad`) and degrees (metadata). Lengths and coordinates use unit chord.
Pressure and shear are density-divided, and `q = Uinf**2 / 2`.

The tangent is the original point normal rotated +90 degrees: `(-ny, nx)`.
Normals are never flipped or renormalized. The full shear vector is retained for
the exact reference-force check. Prediction integration reconstructs only the
tangential shear, so checks also quantify the loss from dropping `cf_n`.
Normal direction is measured relative to the point centroid. The upper-surface
Cf diagnostic uses `y > mean(y)` and excludes the first 5% of chord; it is an
orientation heuristic, not a separation detector. A failed force tolerance
(maximum relative error above `1e-6`) or failed forward/backward pass causes a
nonzero checks exit status. A CPU pass does not verify MPS behavior.

The official train list's final `int(0.1 * count)` cases are validation, leaving
180 training and 20 validation cases for scarce; scarce uses the 200-case
`full_test` split. Input `[x, y, Ux, Uy, nx, ny]` and target `[Cp, Cf]` statistics
are point-weighted over training cases only, with `1e-8` added to standard
deviations. One case is one sample, without subsampling. Training loss is the
mean of both normalized component MSEs; validation reports component MSEs,
first averaging over each case's points and then equally over cases.

Training uses Adam and official OneCycleLR step count `(n_train + 1) * epochs`,
with a scheduler step after every optimizer step. Seeds cover Python, NumPy,
and torch; cross-device bitwise reproducibility is not guaranteed. Each epoch
includes validation when scheduled (first epoch, then every `--val-every`).
Timings synchronize MPS at measurement boundaries. MPS memory is the driver
allocation at epoch end, not a measured peak. Timing observations must be
reported separately from accuracy conclusions. Existing run files are protected
from accidental overwrite; choose a new output directory to start another run.

Training outputs: `model.pt`, `norm.npz`, `config.json`, `log.csv`. Config records
split membership, training ranges, seed, model dimensions, device, versions,
timestamps and runtime. Dataset-dependent settings and paths stay in ignored
outputs. Evaluation reconstructs the model and verifies saved split membership.
It writes `eval/metrics.json`, `eval/per_case.csv`, and `eval/cases.png` using Agg.
Field errors are per-case relative L2; force errors use absolute relative error
against all six official reference coefficients, without an epsilon floor.
Zero-reference errors may be infinite in CSV; nonfinite JSON aggregates are
null with an explicit count. Spearman statistics are null when undefined.

The representative figure selects minimum, upper-middle, and maximum Cp-error
cases. Upper/lower curves use the point normal's y sign after determining the
normal orientation, and are sorted separately by x. A centroid-y cut would mix
the surfaces of cambered airfoils. Reynolds (unit chord), angle-of-attack range flags, and previously
unseen case-name geometry parameter tuples are recorded per case. The geometry
flag identifies absent parameter tuples, not geometric distance or a guarantee
of physical interpolation. Test metrics for two epochs establish pipeline
operation only; they do not establish convergence or physical validation.

Sources: AirfRANS 0.1.5.1 and the official Transolver AirfRANS implementation.
See the repository research plan for acquisition, version and license records.
