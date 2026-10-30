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


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validation_mse(model, dataset, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for data, target in dataset:
            losses.append((model(to_device(data, device)) - target.to(device)).square().mean(0).cpu().numpy())
    if not losses:
        raise ValueError("Validation split is empty")
    return np.mean(losses, axis=0)


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
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.val_every < 1 or args.lr <= 0:
        parser.error("epochs, val-every and lr must be positive")
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
    train = SurfaceDataset(args.surface, splits["train"], norm)
    val = SurfaceDataset(args.surface, splits["val"], norm)
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
                  training_range={key: [min(c[key] for c in conditions), max(c[key] for c in conditions)]
                                  for key in ("uinf", "aoa_deg", "nu")},
                  setting="surface only Cp/Cf; surrogate, not a validated flow solution")
    fields = ("epoch", "train_loss", "val_cp_mse", "val_cf_mse", "epoch_seconds", "mps_driver_allocated_bytes")
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
            for i in order:
                data, target = train[i]
                optimizer.zero_grad(set_to_none=True)
                loss = (model(to_device(data, device)) - target.to(device)).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_sum += loss.detach().item()
            val_loss = validation_mse(model, val, device) if epoch == 1 or epoch % args.val_every == 0 else None
            synchronize(device)
            row = dict(epoch=epoch, train_loss=loss_sum / len(train),
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
