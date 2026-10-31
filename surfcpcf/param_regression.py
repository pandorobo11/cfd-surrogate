"""Case-level NACA parameter MLP baseline for official reference Cl and Cd."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .data import Normalization, load_case, read_splits
from .geometry import COEFFICIENTS, relative_error
from .metrics import Extrapolation, coefficient_metrics, coefficient_row, training_ranges
from .model import select_device
from .train import seed_all, synchronize

INPUTS = ("uinf", "aoa_deg", "naca_series_4_digit_0_5_digit_1", "p1", "p2", "p3_or_zero", "thickness")
TARGETS = ("cl", "cd")


def parameter_features(name, metadata):
    """Parse the case name; verify its flow conditions against index metadata."""
    parts = name.split("_")
    if parts[:2] != ["airFoil2D", "SST"] or len(parts) not in (7, 8):
        raise ValueError(f"Invalid NACA case name: {name}")
    values = np.asarray([float(value) for value in parts[2:]], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite case parameters: {name}")
    if not np.allclose(values[:2], [metadata["uinf"], metadata["aoa_deg"]], rtol=0, atol=1e-8):
        raise ValueError(f"Case name and index flow conditions disagree: {name}")
    parameters = values[2:]
    series = len(parameters) - 3
    if series == 0:
        parameters = [parameters[0], parameters[1], 0., parameters[2]]
    elif parameters[2] not in (0., 1.):
        raise ValueError(f"Five-digit reflex parameter must be 0 or 1: {name}")
    return np.asarray([metadata["uinf"], metadata["aoa_deg"], series, *parameters], dtype=np.float64)


class ParameterMLP(nn.Module):
    def __init__(self, n_hidden=128, n_layers=3):
        super().__init__()
        if n_hidden < 1 or n_layers < 1:
            raise ValueError("Hidden width and layer count must be positive")
        layers = []
        for i in range(n_layers):
            layers.extend((nn.Linear(7 if i == 0 else n_hidden, n_hidden), nn.GELU()))
        self.network = nn.Sequential(*layers, nn.Linear(n_hidden, 2))

    def forward(self, x):
        return self.network(x)


def load_arrays(surface, names, metadata):
    x = np.stack([parameter_features(name, metadata[name]) for name in names])
    y = np.stack([load_case(surface, name)["reference"][[COEFFICIENTS.index(k) for k in TARGETS]]
                  for name in names])
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite reference coefficients")
    return x, y


def evaluate(model, x, truth, names, metadata, norm, config, device, out):
    model.eval()
    extrapolation = Extrapolation(config["splits"]["train"],
                                  [metadata[n] for n in config["splits"]["train"]],
                                  config["training_range"])
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        prediction = norm.inverse_targets(model(x).cpu().numpy())
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite test predictions")
    rows = []
    for name, pred, true in zip(names, prediction, truth):
        rows.append({"case": name, **coefficient_row(dict(zip(TARGETS, pred)), dict(zip(TARGETS, true))),
                     **extrapolation.flags(name, metadata[name])})
    synchronize(device)
    elapsed = time.perf_counter() - start
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = dict(task=config["task"],
                   official_test_split="full_test" if config["task"] == "scarce" else config["task"] + "_test",
                   cases=len(rows), epochs=config["epochs"],
                   model={key: config[key] for key in ("n_hidden", "n_layers")},
                   setting=config["setting"], device=str(device), torch_version=torch.__version__,
                   PYTORCH_ENABLE_MPS_FALLBACK=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                   inference_and_integration_seconds=elapsed,
                   training_range=dict(config["training_range"], reynolds_unit_chord=extrapolation.re_range),
                   **coefficient_metrics(rows, TARGETS))
    metrics["model"]["name"] = "param_regression"
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("full", "scarce", "reynolds", "aoa"), default="full")
    parser.add_argument("--surface", type=Path, default=Path("data/airfrans_surface"))
    parser.add_argument("--out", type=Path, default=Path("results/surface/SURF-10"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("mps", "cpu"))
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.n_hidden, args.n_layers) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch-size, hidden width, layers and lr must be positive and finite")
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / name).exists() for name in ("model.pt", "norm.npz", "config.json", "log.csv", "eval")):
        parser.error("Output contains an existing run; choose a fresh --out")
    started = datetime.now(timezone.utc).isoformat()
    device = select_device(args.device)
    synchronize(device)
    start = time.perf_counter()
    seed_all(args.seed)
    splits = read_splits(args.surface, args.task)
    if not all(splits.values()):
        parser.error("Training, validation and test splits must be nonempty")
    metadata = json.loads((args.surface / "index.json").read_text())["cases"]
    arrays = {split: load_arrays(args.surface, names, metadata) for split, names in splits.items()}
    x, y = arrays["train"]
    norm = Normalization(x.mean(0), x.std(0) + 1e-8, y.mean(0), y.std(0) + 1e-8)
    norm.save(args.out / "norm.npz")
    tensors = {split: tuple(torch.tensor(a, dtype=torch.float32, device=device)
                            for a in norm.transform(*pair)) for split, pair in arrays.items()}
    model = ParameterMLP(args.n_hidden, args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(len(x) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model="param_regression", parameter_count=sum(p.numel() for p in model.parameters()),
                  input_definition=list(INPUTS), target_definition=list(TARGETS),
                  target_source="official reference (full shear vector)",
                  normalization="training cases only; population standard deviation plus 1e-8",
                  input_training_range={key: [float(x[:, i].min()), float(x[:, i].max())] for i, key in enumerate(INPUTS)},
                  training_range=training_ranges([metadata[n] for n in splits["train"]]),
                  splits=splits, device=str(device), torch_version=torch.__version__, started_at=started,
                  PYTORCH_ENABLE_MPS_FALLBACK=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                  scheduler_total_steps=total_steps, checkpoint_selection="final epoch; no validation selection",
                  setting="case-level NACA parameter regression; surrogate, not a validated flow solution")
    fields = ("epoch", "train_loss", "val_mse", "val_cl_rel_median", "val_cd_rel_median",
              "epoch_seconds", "mps_driver_allocated_bytes")
    with (args.out / "log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            synchronize(device)
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(len(x))
            loss_sum = 0.
            for ids in order.split(args.batch_size):
                ids = ids.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(tensors["train"][0][ids]) - tensors["train"][1][ids]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_sum += loss.item() * len(ids)
            row = dict(epoch=epoch, train_loss=loss_sum / len(x))
            # Also validate the final epoch for bounded runs shorter than 100 epochs.
            if epoch % 100 == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    pred = model(tensors["val"][0])
                    row["val_mse"] = (pred - tensors["val"][1]).square().mean().item()
                    errors = relative_error(norm.inverse_targets(pred.cpu().numpy()), arrays["val"][1])
                    row.update({f"val_{key}_rel_median": float(np.median(errors[:, i])) for i, key in enumerate(TARGETS)})
            synchronize(device)
            row.update(epoch_seconds=time.perf_counter() - epoch_start,
                       mps_driver_allocated_bytes=torch.mps.driver_allocated_memory() if device.type == "mps" else None)
            writer.writerow(row)
            stream.flush()
            if "val_mse" in row:
                print(json.dumps(row), flush=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    synchronize(device)
    config["training_seconds"] = time.perf_counter() - start
    metrics = evaluate(model, tensors["test"][0], arrays["test"][1], splits["test"], metadata,
                       norm, config, device, args.out / "eval")
    config.update(finished_at=datetime.now(timezone.utc).isoformat(), total_seconds=time.perf_counter() - start,
                  optimizer_steps=total_steps)
    (args.out / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(device=str(device), parameters=config["parameter_count"],
                         train_cases=len(x), validation_cases=len(splits["val"]),
                         total_seconds=config["total_seconds"],
                         coefficient_relative_error=metrics["coefficient_relative_error"])), flush=True)


if __name__ == "__main__":
    main()
