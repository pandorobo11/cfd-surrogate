import csv
import json

import numpy as np
import pytest
import torch

from surfcpcf.data import Normalization, SurfaceDataset
from surfcpcf.param_regression import main, parameter_features
from surfcpcf.torch_forces import coefficients_from_cp_cf


@pytest.mark.parametrize('name, expected', [
    ('airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32', [31.283, -4.156, 0, .919, 6.98, 0, 14.32]),
    ('airFoil2D_SST_31.382_3.588_1.994_6.206_0.0_13.271', [31.382, 3.588, 1, 1.994, 6.206, 0, 13.271]),
])
def test_parameter_features(name, expected):
    metadata = dict(uinf=expected[0], aoa_deg=expected[1])
    np.testing.assert_array_equal(parameter_features(name, metadata), expected)
    with pytest.raises(ValueError, match='disagree'):
        parameter_features(name, dict(uinf=0., aoa_deg=0.))


@pytest.fixture
def synthetic_surface(tmp_path):
    surface = tmp_path / 'surface'
    surface.mkdir()
    names, metadata = [], {}
    for i in range(14):
        u, a = 30 + i, -2 + i
        name = f'airFoil2D_SST_{u}_{a}_1.2_4.0_{10+i}'
        names.append(name)
        meta = dict(uinf=u, aoa_deg=a, nu=1.5e-5)
        metadata[name] = meta
        pos = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
        np.savez(surface / f'{name}.npz', pos=pos, normal=pos * 2,
                 cp=pos[:, 0] * (i + 1) + pos[:, 1] * 2, cf=pos[:, 1] * .01,
                 uinf=np.array([u, 0.]), lines=np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
                 length=np.ones(4), cell_normal=np.array([[0., -1.], [1., 0.], [0., 1.], [-1., 0.]]),
                 aoa_rad=np.deg2rad(a), q=.5 * u**2, reference=np.array([.01+i*.001, .005, .005, .2+i*.1, .1, .1]),
                 metadata=json.dumps(meta))
    manifest = dict(full_train=names[:10], full_test=names[10:], scarce_train=names[:10])
    (surface / 'index.json').write_text(json.dumps(dict(cases=metadata, manifest=manifest)))
    return surface, names


def test_dataset_target_is_tangential_integral(synthetic_surface):
    surface, names = synthetic_surface
    norm = Normalization.fit(surface, names[:9])
    dataset = SurfaceDataset(surface, names[:9], norm, include_forces=True)
    for i, (_, y) in enumerate(dataset):
        physical = y * torch.tensor(norm.y_std, dtype=torch.float32) + torch.tensor(norm.y_mean, dtype=torch.float32)
        expected = coefficients_from_cp_cf(physical[:, 0], physical[:, 1], **dataset.force_data[i])
        for key in expected:
            torch.testing.assert_close(dataset.target_coefficients[i][key], expected[key], rtol=0, atol=0)
        assert float(expected['cd']) != dataset.reference_coefficients[i][0]


def test_parameter_training_and_metrics(synthetic_surface, tmp_path):
    surface, names = synthetic_surface
    out = tmp_path / 'param'
    main(['--surface', str(surface), '--out', str(out), '--epochs', '3', '--batch-size', '4',
          '--n-hidden', '8', '--n-layers', '2', '--device', 'cpu'])
    assert all((out / p).is_file() for p in ['model.pt', 'norm.npz', 'config.json', 'log.csv', 'eval/metrics.json', 'eval/per_case.csv'])
    config = json.loads((out / 'config.json').read_text())
    assert config['splits'] == dict(train=names[:9], val=names[9:10], test=names[10:])
    assert config['optimizer_steps'] == 9
    norm = Normalization.load(out / 'norm.npz')
    assert norm.x_mean[0] == 34.
    assert norm.y_mean[0] == pytest.approx(.6)
    metrics = json.loads((out / 'eval/metrics.json').read_text())
    assert metrics['cases'] == 4
    assert set(metrics['coefficient_relative_error']) == {'cl', 'cd'}
    assert metrics['extrapolation_counts']['aoa_extrapolation'] == 4
    assert 'field_relative_l2' not in metrics
    with (out / 'eval/per_case.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r['case'] for r in rows] == names[10:]
    for key in ['cl', 'cd']:
        errors = [abs(float(r[f'pred_{key}']) - float(r[f'true_{key}'])) / abs(float(r[f'true_{key}'])) for r in rows]
        assert metrics['coefficient_relative_error'][key]['median'] == pytest.approx(np.median(errors))


def test_old_config_evaluation_and_zero_weight_validation(synthetic_surface, tmp_path, monkeypatch):
    from surfcpcf import train, evaluate
    surface, _ = synthetic_surface
    out = tmp_path / 'surface_run'
    train.main(['--task', 'scarce', '--surface', str(surface), '--out', str(out), '--model', 'mlp',
                '--epochs', '1', '--val-every', '1', '--n-hidden', '8', '--n-layers', '1', '--device', 'cpu'])
    with (out / 'log.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['train_force_loss'] == '0.0'
    assert rows[0]['val_cl_rel_median'] and rows[0]['val_cd_rel_median']
    config = json.loads((out / 'config.json').read_text())
    for key in list(config):
        if key.startswith('force_loss'):
            del config[key]
    (out / 'config.json').write_text(json.dumps(config))
    monkeypatch.setattr(evaluate, 'plot_cases', lambda *args: None)
    evaluate.main(['--run', str(out), '--surface', str(surface), '--device', 'cpu'])
    metrics = json.loads((out / 'eval/metrics.json').read_text())
    assert metrics['cases'] == 4 and 'cdp' in metrics['coefficient_relative_error']
