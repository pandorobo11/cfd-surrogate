# Repository instructions

- Keep this repository private unless the user explicitly requests a visibility change.
- Do not commit personal names, email addresses, account identifiers, absolute home paths, hostnames, device identifiers, credentials, or private remote URLs in tracked files.
- Keep datasets, training outputs, checkpoints, raw logs, upstream clones, and machine-specific configuration in the ignored `data/`, `results/`, `external/`, and `local/` directories.
- Do not commit dataset files or upstream source code. Record how to obtain them (URL, version or commit, license) instead.
- Do not force-add ignored files without an explicit user request and a content review.
- Review the exact staged diff and commit metadata before pushing. Use the repository's generic local Git author identity; do not replace it with a personal identity by default.
- Run on the Apple GPU through PyTorch MPS by default. Select the device explicitly (`mps`, else `cpu`); never hard-code `.cuda()`. Record whether `PYTORCH_ENABLE_MPS_FALLBACK=1` was used, since CPU fallback changes timings.
- Record performance observations (time per epoch, memory) separately from validated accuracy conclusions. Compare models only under the same data split, point sampling, normalization, and evaluation metric.
- Report accuracy on the dataset's official split and metrics where one exists (for AirfRANS: the `full`, `scarce`, `reynolds`, and `aoa` tasks, field errors, and lift/drag coefficient errors). State any subsampling, reduced model size, or reduced epoch count next to the result.
- Never present a surrogate prediction as a validated flow solution. State the training range and mark extrapolation in Reynolds number, angle of attack, or geometry.

- User preference for recording research results in this repository: use Markdown, reviewed figures, and compact CSV/JSON tables with reproduction instructions; apply those research records directly to the remote default branch without a PR when publication is requested. This preference is limited to research records, does not authorize unrelated code changes, force pushes, or changing branch protections, and does not override tool permissions or approval requirements.

- Research record layout: put each investigation's record in `docs/reports/<YYYYMMDD>-<name>/` (add a `.gitignore` exception when it tracks CSV tables), keep topic conclusions in `docs/<topic>/`, and add the record and any changed conclusion to `docs/README.md`.

- User preference for research case tables: use one row per case (training run) and columns for settings, state common data and training conditions first, separate executed runs from plans, and match run IDs between settings and result tables.
