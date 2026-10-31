"""Train surface-only Cp/Cf Transolver, one complete airfoil per optimizer step."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import Normalization, SurfaceDataset, load_case, read_splits, to_device
from .model import MODELS, build_model, select_device
from .geometry import COEFFICIENTS, relative_error
from .metrics import training_ranges
from .torch_forces import coefficients_from_cp_cf


def parse_force_coeffs(value):
    keys = [key.strip() for key in value.split(",")]
    if not keys or any(key not in COEFFICIENTS for key in keys) or len(set(keys)) != len(keys):
        raise argparse.ArgumentTypeError(f"Choose distinct coefficients from {COEFFICIENTS}")
    return keys


def force_loss_scales(dataset, keys):
    # Population standard deviation across training cases; floor only degenerate scales.
    return {key: max(float(np.std([float(c[key]) for c in dataset.target_coefficients])), 1e-8)
            for key in keys}


def integrated_prediction(prediction, geometry, y_mean, y_std):
    physical = prediction * y_std + y_mean
    geometry = {key: value.to(prediction.device) for key, value in geometry.items()}
    return coefficients_from_cp_cf(physical[:, 0], physical[:, 1], **geometry)


def training_loss(prediction, target, weight=0., *, geometry=None, coefficients=None,
                  scales=None, y_mean=None, y_std=None):
    field_loss = (prediction - target).square().mean()
    # Preserve the exact old graph and arithmetic, and skip integration at weight zero.
    if weight == 0:
        return field_loss, field_loss, prediction.new_zeros(())
    predicted = integrated_prediction(prediction, geometry, y_mean, y_std)
    force_loss = torch.stack([((predicted[key] - coefficients[key].to(prediction.device)) / scale).square()
                              for key, scale in scales.items()]).mean()
    return field_loss + weight * force_loss, field_loss, force_loss


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validation_mse(model, dataset, device, norm=None):
    model.eval()
    losses = []
    errors = []
    if norm is not None:
        y_mean = torch.tensor(norm.y_mean, dtype=torch.float32, device=device)
        y_std = torch.tensor(norm.y_std, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, (data, target) in enumerate(dataset):
            prediction = model(to_device(data, device))
            losses.append((prediction - target.to(device)).square().mean(0).cpu().numpy())
            if norm is not None:
                coefficients = integrated_prediction(prediction, dataset.force_data[i], y_mean, y_std)
                errors.append([float(relative_error(float(coefficients[key]), dataset.reference_coefficients[i][COEFFICIENTS.index(key)]))
                               for key in ("cl", "cd")])
    if not losses:
        raise ValueError("Validation split is empty")
    mse = np.mean(losses, axis=0)
    return mse if norm is None else (mse, np.median(errors, axis=0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("scarce", "full", "reynolds", "aoa"), default="scarce")
    parser.add_argument("--surface", type=Path, default=Path("data/airfrans_surface"))
    parser.add_argument("--transolver-dir", type=Path, default=Path("external/Transolver/Airfoil-Design-AirfRANS"))
    parser.add_argument("--out", type=Path, default=Path("results/surface/SURF-01"))
    parser.add_argument("--model", choices=MODELS, default="transolver")
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--slice-num", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("mps", "cpu"))
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--force-loss-weight", type=float, default=0.)
    parser.add_argument("--force-loss-coeffs", type=parse_force_coeffs, default="cl,cd")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.val_every < 1 or args.lr <= 0:
        parser.error("epochs, val-every and lr must be positive")
    if not np.isfinite(args.force_loss_weight) or args.force_loss_weight < 0:
        parser.error("force-loss-weight must be finite and nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / name).exists() for name in ("model.pt", "config.json", "log.csv", "norm.npz")):
        parser.error("Output contains an existing run; choose a fresh --out")
    started = datetime.now(timezone.utc).isoformat()
    device = select_device(args.device)
    synchronize(device)
    start = time.perf_counter()
    seed_all(args.seed)
    splits = read_splits(args.surface, args.task)
    if not splits["train"] or not splits["val"]:
        parser.error("Training and validation splits must be nonempty")
    norm = Normalization.fit(args.surface, splits["train"])
    norm.save(args.out / "norm.npz")
    train = SurfaceDataset(args.surface, splits["train"], norm, include_forces=True)
    val = SurfaceDataset(args.surface, splits["val"], norm, include_forces=True)
    scales = force_loss_scales(train, args.force_loss_coeffs)
    y_mean = torch.tensor(norm.y_mean, dtype=torch.float32, device=device)
    y_std = torch.tensor(norm.y_std, dtype=torch.float32, device=device)
    model = build_model(args.model, args.transolver_dir, args.n_hidden, args.n_layers, args.slice_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = (len(train) // 1 + 1) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    conditions = [json.loads(str(load_case(args.surface, name)["metadata"])) for name in splits["train"]]
    config.update(parameter_count=sum(p.numel() for p in model.parameters()), device=str(device),
                  PYTORCH_ENABLE_MPS_FALLBACK=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                  torch_version=torch.__version__, started_at=started, splits=splits,
                  scheduler_total_steps=total_steps, batch_size=1, point_subsampling=False,
                  validation_reduction="mean over points per case, then equal mean over cases",
                  training_range=training_ranges(conditions), force_loss_scales=scales,
                  force_loss_target="integrated ground-truth Cp and tangential Cf; fp32 inverse normalization",
                  force_loss_scale_floor=1e-8,
                  validation_force_target="official reference (full shear vector)",
                  setting="surface only Cp/Cf; surrogate, not a validated flow solution")
    fields = ("epoch", "train_loss", "train_field_loss", "train_force_loss", "val_cp_mse", "val_cf_mse",
              "val_cl_rel_median", "val_cd_rel_median", "epoch_seconds", "mps_driver_allocated_bytes")
    with (args.out / "log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            synchronize(device)
            epoch_start = time.perf_counter()
            model.train()
            order = list(range(len(train)))
            random.shuffle(order)
            loss_sum = 0.
            field_sum = 0.
            force_sum = 0.
            for i in order:
                data, target = train[i]
                optimizer.zero_grad(set_to_none=True)
                loss, field_loss, force_loss = training_loss(
                    model(to_device(data, device)), target.to(device), args.force_loss_weight,
                    geometry=train.force_data[i], coefficients=train.target_coefficients[i],
                    scales=scales, y_mean=y_mean, y_std=y_std)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_sum += loss.detach().item()
                field_sum += field_loss.detach().item()
                force_sum += force_loss.detach().item()
            val_loss, val_errors = validation_mse(model, val, device, norm) if epoch == 1 or epoch % args.val_every == 0 else (None, None)
            synchronize(device)
            row = dict(epoch=epoch, train_loss=loss_sum / len(train),
                       train_field_loss=field_sum / len(train),
                       train_force_loss=force_sum / len(train),
                       val_cl_rel_median=float(val_errors[0]) if val_errors is not None else None,
                       val_cd_rel_median=float(val_errors[1]) if val_errors is not None else None,
                       val_cp_mse=float(val_loss[0]) if val_loss is not None else None,
                       val_cf_mse=float(val_loss[1]) if val_loss is not None else None,
                       epoch_seconds=time.perf_counter() - epoch_start,
                       mps_driver_allocated_bytes=torch.mps.driver_allocated_memory() if device.type == "mps" else None)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    synchronize(device)
    config.update(finished_at=datetime.now(timezone.utc).isoformat(), total_seconds=time.perf_counter() - start,
                  optimizer_steps=len(train) * args.epochs)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"device": str(device), "parameters": config["parameter_count"],
                      "train_cases": len(train), "validation_cases": len(val),
                      "total_seconds": config["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
