"""Extract all surface points in VTP patch order; never modify source meshes."""

import argparse
import json
from pathlib import Path
import time

import numpy as np


def preprocess_case(dataset, name):
    from airfrans.simulation import Simulation
    from airfrans.reorganize import reorganize

    start = time.perf_counter()
    sim = Simulation(root=str(dataset), name=name)
    pos, normal = sim.airfoil_position, sim.airfoil_normals
    packed = np.asarray(sim.airfoil.lines).reshape(-1, 3)
    if not np.all(packed[:, 0] == 2) or len(packed) != sim.airfoil.n_cells:
        raise ValueError(f"{name}: expected only two-point line cells")
    p = reorganize(sim.position[sim.surface], pos, sim.pressure[sim.surface]).ravel()
    tau = sim.wallshearstress(over_airfoil=True, reference=True)
    u, alpha = float(sim.inlet_velocity), float(sim.angle_of_attack)
    q = 0.5 * u ** 2
    tangent = np.stack((-normal[:, 1], normal[:, 0]), axis=1)
    reference = np.asarray(sim.force_coefficient(reference=True)).ravel()
    # VTK ptc overwrites the original cell Normals with point-averaged Normals.
    airfrans_normal = np.asarray(sim.airfoil.ptc().cell_data["Normals"])[:, :2]
    arrays = dict(
        pos=pos, normal=normal, uinf=u * np.array([np.cos(alpha), np.sin(alpha)]),
        p=p, tau=tau, cp=p / q, cf=(tau * tangent).sum(axis=1) / q,
        cf_n=(tau * normal).sum(axis=1) / q,
        lines=packed[:, 1:].astype(np.int64), length=np.asarray(sim.airfoil.cell_data["Length"]),
        cell_normal=airfrans_normal, reference=reference,
        q=q, aoa_rad=alpha,
    )
    meta = dict(case=name, uinf=u, aoa_deg=float(np.rad2deg(alpha)),
                nu=float(sim.NU), n_points=len(pos), n_cells=len(packed),
                preprocessing_seconds=time.perf_counter() - start)
    return arrays, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/airfrans/Dataset"))
    parser.add_argument("--out", type=Path, default=Path("data/airfrans_surface"))
    parser.add_argument("--splits", nargs="+", default=["scarce_train", "full_test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    names = list(dict.fromkeys(name for split in args.splits for name in manifest[split]))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        names = names[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"cases": {}}
    # Store split membership locally so train/evaluate need no volume dataset path.
    index.update(schema_version=2, cell_normal_convention="airfoil.ptc", manifest=manifest)
    start = time.perf_counter()
    processed = 0
    for i, name in enumerate(names, 1):
        path = args.out / f"{name}.npz"
        if path.exists() and not args.overwrite:
            with np.load(path, allow_pickle=False) as data:
                index["cases"][name] = json.loads(str(data["metadata"]))
            print(f"[{i}/{len(names)}] {name} skipped", flush=True)
        else:
            arrays, meta = preprocess_case(args.dataset, name)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays, metadata=json.dumps(meta))
            temporary.replace(path)
            index["cases"][name] = meta
            processed += 1
            print(f"[{i}/{len(names)}] {name} N={meta['n_points']} "
                  f"{meta['preprocessing_seconds']:.3f}s", flush=True)
        temporary_index = index_path.with_suffix(".tmp.json")
        temporary_index.write_text(json.dumps(index, indent=2) + "\n")
        temporary_index.replace(index_path)
    print(json.dumps(dict(selected=len(names), processed=processed,
                          elapsed_seconds=time.perf_counter() - start)), flush=True)


if __name__ == "__main__":
    main()
