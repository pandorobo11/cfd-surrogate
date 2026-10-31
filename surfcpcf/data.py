"""Official case splits, point-weighted train-only statistics, batch size one."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from .torch_forces import coefficients_from_cp_cf


def split_cases(manifest, task):
    train = list(manifest[task + "_train"])
    n = int(0.1 * len(train))
    # Avoid Python's [:-0] returning an empty training set on tiny fixtures.
    return {"train": train[:-n] if n else train,
            "val": train[-n:] if n else [],
            "test": list(manifest["full_test" if task == "scarce" else task + "_test"])}


def load_case(surface, name):
    with np.load(Path(surface) / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def features(case):
    x = np.concatenate((case["pos"], np.broadcast_to(case["uinf"], case["pos"].shape),
                        case["normal"]), axis=1)
    y = np.stack((case["cp"], case["cf"]), axis=1)
    return x, y


class Normalization:
    def __init__(self, x_mean, x_std, y_mean, y_std):
        self.x_mean, self.x_std = np.asarray(x_mean), np.asarray(x_std)
        self.y_mean, self.y_std = np.asarray(y_mean), np.asarray(y_std)

    @classmethod
    def fit(cls, surface, train_names):
        if not train_names:
            raise ValueError("No training cases")
        xy = [features(load_case(surface, name)) for name in train_names]
        x, y = (np.concatenate([pair[i] for pair in xy], axis=0) for i in range(2))
        return cls(x.mean(0), x.std(0) + 1e-8, y.mean(0), y.std(0) + 1e-8)

    def transform(self, x, y):
        return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std

    def inverse_targets(self, y):
        return np.asarray(y) * self.y_std + self.y_mean

    def save(self, path):
        np.savez(path, x_mean=self.x_mean, x_std=self.x_std,
                 y_mean=self.y_mean, y_std=self.y_std)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{key: data[key] for key in ("x_mean", "x_std", "y_mean", "y_std")})


class SurfaceDataset(Dataset):
    def __init__(self, surface, names, norm, include_forces=False):
        self.names = list(names)
        self.samples = []
        self.force_data = []
        self.target_coefficients = []
        self.reference_coefficients = []
        for name in self.names:
            case = load_case(surface, name)
            x, y = norm.transform(*features(case))
            self.samples.append((torch.tensor(x, dtype=torch.float32),
                                 torch.tensor(case["pos"], dtype=torch.float32),
                                 torch.tensor(y, dtype=torch.float32)))
            if include_forces:
                geometry = {key: torch.tensor(case[key], dtype=torch.long if key == "lines" else torch.float32)
                            for key in ("normal", "lines", "length", "cell_normal", "aoa_rad")}
                # Use the same fp32 inverse transform as predictions, including rounding.
                truth = self.samples[-1][2] * torch.tensor(norm.y_std, dtype=torch.float32) + torch.tensor(norm.y_mean, dtype=torch.float32)
                self.force_data.append(geometry)
                self.target_coefficients.append(coefficients_from_cp_cf(truth[:, 0], truth[:, 1], **geometry))
                self.reference_coefficients.append(case["reference"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, pos, y = self.samples[index]
        return SimpleNamespace(x=x, pos=pos), y


def read_splits(surface, task):
    index = json.loads((Path(surface) / "index.json").read_text())
    splits = split_cases(index["manifest"], task)
    missing = [name for names in splits.values() for name in names
               if not (Path(surface) / f"{name}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"Preprocess all requested splits first; missing {len(set(missing))} cases")
    return splits


def to_device(data, device):
    return SimpleNamespace(x=data.x.to(device), pos=data.pos.to(device))
