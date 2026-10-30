import numpy as np
import pytest
import torch

from surfcpcf.data import Normalization, SurfaceDataset, features, load_case, split_cases


@pytest.mark.parametrize("task", ["scarce", "full", "reynolds", "aoa"])
def test_official_splits(task):
    manifest = {task + "_train": list(range(23)), task + "_test": [24], "full_test": [25]}
    result = split_cases(manifest, task)
    assert result["train"] == list(range(21))
    assert result["val"] == [21, 22]
    assert result["test"] == manifest["full_test" if task == "scarce" else task + "_test"]


def test_small_split_does_not_drop_training():
    assert split_cases({"scarce_train": [1], "full_test": [2]}, "scarce")["train"] == [1]


def test_train_only_point_weighted_normalization(tmp_path):
    for name, n, offset in (("train1", 2, 0.), ("train2", 4, 10.), ("val", 3, 10000.)):
        pos = np.arange(n * 2).reshape(n, 2) + offset
        np.savez(tmp_path / f"{name}.npz", pos=pos, uinf=np.array([3., 4.]),
                 normal=np.tile([1., 0.], (n, 1)), cp=pos[:, 0], cf=pos[:, 1])
    names = ["train1", "train2"]
    norm = Normalization.fit(tmp_path, names)
    arrays = [features(load_case(tmp_path, name)) for name in names]
    x = np.concatenate([a[0] for a in arrays])
    y = np.concatenate([a[1] for a in arrays])
    np.testing.assert_allclose(norm.x_mean, x.mean(0))
    np.testing.assert_allclose(norm.x_std, x.std(0) + 1e-8)
    xn, yn = norm.transform(x, y)
    np.testing.assert_allclose(xn.mean(0), 0., atol=1e-14)
    np.testing.assert_allclose(yn.mean(0), 0., atol=1e-14)
    np.testing.assert_allclose(norm.inverse_targets(yn), y)
    norm.save(tmp_path / "norm.npz")
    loaded = Normalization.load(tmp_path / "norm.npz")
    np.testing.assert_array_equal(loaded.y_mean, norm.y_mean)
    dataset = SurfaceDataset(tmp_path, names, loaded)
    data, target = dataset[0]
    assert data.x.dtype == target.dtype == torch.float32
    assert data.x.shape == (2, 6) and target.shape == (2, 2)
    np.testing.assert_array_equal(data.pos.numpy(), load_case(tmp_path, names[0])["pos"])
    assert dataset[1][0].x.shape == (4, 6)
