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

## NACA family holdout tasks

Both surface learning and case-level regression accept `--task naca4to5` and
`--task naca5to4`. Family is determined from the three (four-digit) or four
(five-digit, including reflex) geometry parameters after speed and angle in the
case name. These are custom family holdouts derived from the official `full`
lists, not additional official AirfRANS benchmark tasks. The four existing
tasks (`full`, `scarce`, `reynolds`, `aoa`) retain their original splits.

- `naca4to5`: filter `full_train` to four-digit cases (378), then use its final
  `int(0.1 * 378)` cases for validation: 341 train / 37 validation. Test on the
  89 five-digit cases of `full_test`.
- `naca5to4`: filter `full_train` to five-digit cases (422): 380 train /
  42 validation. Test on the 111 four-digit cases of `full_test`.

Filtering preserves manifest order. All test cases have unseen geometry
parameter tuples; this flag still measures tuple membership, not geometric
distance. Family holdout evaluates transfer between these two NACA families;
it alone cannot establish generalization to arbitrary non-NACA shapes.
Evaluation reads the task from the saved config. Optional `evaluate --task`
must agree with it. Metrics label the subset as `full_test (5-digit only)` or
`full_test (4-digit only)`.

```sh
$PY -m surfcpcf.train --task naca4to5 --surface data/airfrans_surface \
  --transolver-dir external/Transolver/Airfoil-Design-AirfRANS \
  --out results/surface/SMOKE-FAM-T --epochs 2 --val-every 1
$PY -m surfcpcf.evaluate --run results/surface/SMOKE-FAM-T \
  --surface data/airfrans_surface \
  --transolver-dir external/Transolver/Airfoil-Design-AirfRANS
```

## Integrated force loss

```sh
$PY -m surfcpcf.train --task scarce --surface data/airfrans_surface \
  --transolver-dir external/Transolver/Airfoil-Design-AirfRANS \
  --out results/surface/SMOKE-FORCE --epochs 2 --val-every 1 \
  --force-loss-weight 1 --force-loss-coeffs cl,cd
$PY -m surfcpcf.evaluate --run results/surface/SMOKE-FORCE \
  --surface data/airfrans_surface \
  --transolver-dir external/Transolver/Airfoil-Design-AirfRANS
```

`--force-loss-weight` defaults to zero, preserving the original field loss,
its gradients and optimizer updates. Positive weights enable

```text
L = mean((Y_pred_normalized - Y_true_normalized)^2) + weight * L_force
L_force = mean over selected C of ((C_pred - C_target) / s_C)^2
s_C = max(population_std(C_target over training cases), 1e-8)
```

`--force-loss-coeffs` defaults to `cl,cd` and accepts distinct members of
`cd,cdp,cdv,cl,clp,clv`. The scale is computed once, equally weighting training
cases, and saved with the selected coefficients and weight in `config.json`.
The `1e-8` floor only protects degenerate training distributions.

The differentiable integral reverses target normalization, forms the point
shear `Cf * (-ny, nx)`, averages both endpoint values for each line, integrates
pressure with `cell_normal * length` and shear with `length`, then rotates with
`[[cos(a), sin(a)], [-sin(a), cos(a)]]`. Dynamic pressure cancels. It neither
flips nor normalizes normals. All surface points and lines are retained.

**Loss targets** are integrated ground-truth Cp and tangential Cf, not the
full-vector official `reference`. Targets use the same fp32 inverse transform
as predictions so a prediction equal to the normalized truth has zero force
loss. `SurfaceDataset(..., include_forces=True)` caches geometry and these
`target_coefficients`; its existing `(data, target)` item interface is unchanged.
**Validation errors** use the official reference, like test evaluation, and may
therefore retain a small difference due to omitted normal shear even for exact
Cp/Cf predictions. In `log.csv`, `train_loss` is the total loss (field loss plus
weighted force loss). `train_field_loss` is the epoch mean of the per-case field
MSE, directly comparable to historical field-only `train_loss`; at weight zero,
`train_field_loss` equals `train_loss`. The log also includes `train_force_loss`
(unweighted; zero when integration is disabled), `val_cl_rel_median` and
`val_cd_rel_median`. Validation
errors are recorded on validation epochs even at weight zero. Old configs
without the force-loss keys remain evaluable.

## Case-level parameter regression (SURF-09 to SURF-12)

```sh
$PY -m surfcpcf.param_regression --task full --surface data/airfrans_surface \
  --out results/surface/SMOKE-PARAM --epochs 200
# Full planned runs require a separate reviewed decision:
# --epochs 2000 --lr 1e-3 --batch-size 32 --n-hidden 128 --n-layers 3 --seed 0
```

With `--features naca` (the default), each case produces the same original
seven-component input in this order:
`[Uinf, aoa_deg, series, p1, p2, p3_or_zero, thickness]`. `series=0` denotes
four-digit NACA with parameters `[p1,p2,0,p3]`; `series=1` denotes five-digit
NACA with `[p1,p2,p3,p4]`. Thus thickness is always last, and the five-digit
reflex flag occupies the sixth input. Case-name speed and angle are checked
against `index.json` metadata (absolute tolerance `1e-8`). Outputs are `[Cl,Cd]`
from the official full-vector `reference` (indices 3 and 0).

The official train list's final 10% is validation; scarce uses `full_test`.
Input/output normalization uses equally weighted training cases only, with
`1e-8` added to population standard deviations. Defaults are three hidden
layers of width 128, GELU, Adam at `1e-3`, batch size 32, seed 0 and 2000 epochs.
Each epoch shuffles the training cases. OneCycleLR uses exactly
`ceil(n_train / batch_size) * epochs` optimizer steps. The final epoch model is
saved, without validation selection. Validation runs every 100 epochs and at
the final epoch, including shorter smoke runs.

Outputs are `model.pt`, `norm.npz`, `config.json`, `log.csv`,
`eval/metrics.json` and `eval/per_case.csv`. Config includes input/target order,
training ranges, splits, runtime, model size, device and fallback state. Logs
contain per-epoch training MSE and timing; validation columns are filled only
on validation epochs. Test evaluation shares relative errors, nonfinite-value
handling, coefficient summaries, Spearman correlations and extrapolation rules
with surface-field evaluation. No field or pressure/viscous-component errors
are reported for this model. Saved training ranges and per-case flags mark
Reynolds/angle extrapolation and unseen geometry parameter tuples. A 200-epoch
smoke run verifies this pipeline only; it is not evidence of convergence or a
validated flow solution, and timings must be kept separate from accuracy.

### Measured geometry features

`--features geometry --stations K` replaces NACA parameters with measurements
from NPZ `pos` and `normal`; K defaults to 20. Input order is
`[Uinf, aoa_deg, thickness_1, ..., thickness_K, camber_1, ..., camber_K]`
(42 components at the default). Speed and angle come from index metadata;
shape extraction does not parse or use the case name. Targets, normalization,
optimizer, scheduler and evaluation metrics are unchanged.

For each case, x is mapped to `[0, 1]` using its minimum and maximum. The
existing unit-chord y coordinates are retained. At
`x_k = 0.5 * (1 - cos(pi * k / (K + 1)))`, k=1..K, upper and lower y are
linearly interpolated with `np.interp`. Inward normals identify upper points
by `normal_y < 0`, lower points by `normal_y > 0`; zero-y normals are omitted.
Repeated x coordinates on a side use the outer envelope (maximum upper y,
minimum lower y), then unique x values are sorted. Each side needs at least
two distinct x coordinates. Outside a side's sampled x range, `np.interp`
holds its endpoint y. Thickness is `y_upper - y_lower`; camber is
`(y_upper + y_lower) / 2`. Negative thickness is retained, not clipped, so it
can be audited. Shapes must use the same chord alignment and inward-normal
convention; the extractor does not rotate arbitrary coordinates.

Config records `features`, `stations`, the ordered `input_definition`, and
`station_x` for geometry inputs; metrics also identify `model.features`.
`--stations` must be positive and has no effect on NACA inputs.

```sh
$PY -m surfcpcf.param_regression --task naca4to5 --features geometry \
  --surface data/airfrans_surface --out results/surface/SMOKE-FAM-PG --epochs 200
$PY -m surfcpcf.param_regression --task naca4to5 --features naca \
  --surface data/airfrans_surface --out results/surface/SMOKE-FAM-PN --epochs 200
$PY -m surfcpcf.param_regression --task full --features geometry \
  --surface data/airfrans_surface --out results/surface/SMOKE-FULL-PG --epochs 200
```

These bounded runs verify execution only. The NACA family indicator and the
five-digit reflex input are constant in four-digit training, so transferring
that baseline to five-digit inputs can produce extreme normalized values
under the unchanged `std + 1e-8` rule. Report this behavior explicitly when
interpreting holdout errors; do not silently alter baseline normalization.
