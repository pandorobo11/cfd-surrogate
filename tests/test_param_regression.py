import csv
import json

import numpy as np
import pytest
import torch

from surfcpcf.data import Normalization, SurfaceDataset
from surfcpcf.param_regression import fit_normalization, geometry_features, load_arrays, main, parameter_features
from surfcpcf.torch_forces import coefficients_from_cp_cf


def test_constant_input_normalization_bounds_shifted_test_features():
    x = np.array([[0., 2., -1e-7, -1e-6], [0., 4., 1e-7, 1e-6]])
    y = np.array([[.2, .01], [.4, .01]])
    norm, constant = fit_normalization(x, y)
    np.testing.assert_array_equal(constant, [True, False, True, False])
    np.testing.assert_array_equal(norm.x_std[constant], [1., 1.])
    np.testing.assert_array_equal(norm.x_std[~constant], (x.std(0) + 1e-8)[~constant])
    np.testing.assert_array_equal(norm.x_mean, x.mean(0))
    np.testing.assert_array_equal(norm.y_mean, y.mean(0))
    np.testing.assert_array_equal(norm.y_std, y.std(0) + 1e-8)
    test_x = np.array([[1., 5., 1., 2e-6]])
    normalized, _ = norm.transform(test_x, y[:1])
    assert np.isfinite(normalized).all()
    assert np.abs(normalized).max() < 100
    np.testing.assert_array_equal(normalized[:, constant], (test_x - x.mean(0))[:, constant])


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


@pytest.mark.parametrize('feature_mode', ['naca', 'geometry'])
def test_parameter_training_and_metrics(synthetic_surface, tmp_path, feature_mode):
    surface, names = synthetic_surface
    out = tmp_path / 'param'
    feature_args = [] if feature_mode == 'naca' else ['--features', 'geometry', '--stations', '5']
    if feature_mode == 'geometry':
        # Use inward normals for the geometry test; preserve the force fixtures.
        for name in names:
            with np.load(surface / f'{name}.npz') as data:
                case = dict(data)
            case['normal'] *= -1
            np.savez(surface / f'{name}.npz', **case)
    main(feature_args + ['--surface', str(surface), '--out', str(out), '--epochs', '3', '--batch-size', '4',
          '--n-hidden', '8', '--n-layers', '2', '--device', 'cpu'])
    assert all((out / p).is_file() for p in ['model.pt', 'norm.npz', 'config.json', 'log.csv', 'eval/metrics.json', 'eval/per_case.csv'])
    config = json.loads((out / 'config.json').read_text())
    assert config['splits'] == dict(train=names[:9], val=names[9:10], test=names[10:])
    assert config['optimizer_steps'] == 9
    assert config['features'] == feature_mode
    assert config['stations'] == (20 if feature_mode == 'naca' else 5)
    assert len(config['input_definition']) == (7 if feature_mode == 'naca' else 12)
    assert config['constant_input_std_threshold'] == 1e-6
    expected_constant = (['naca_series_4_digit_0_5_digit_1', 'p1', 'p2', 'p3_or_zero']
                         if feature_mode == 'naca' else config['input_definition'][2:])
    assert config['constant_input_features'] == expected_constant
    if feature_mode == 'geometry':
        assert config['input_definition'] == ['uinf', 'aoa_deg'] + [f'thickness_{k}' for k in range(1, 6)] + [f'camber_{k}' for k in range(1, 6)]
        assert len(config['station_x']) == 5
    norm = Normalization.load(out / 'norm.npz')
    constant_indices = [config['input_definition'].index(key) for key in expected_constant]
    np.testing.assert_array_equal(norm.x_std[constant_indices], np.ones(len(constant_indices)))
    assert norm.x_mean[0] == 34.
    assert norm.y_mean[0] == pytest.approx(.6)
    metrics = json.loads((out / 'eval/metrics.json').read_text())
    assert metrics['cases'] == 4
    assert metrics['model']['features'] == feature_mode
    assert set(metrics['coefficient_relative_error']) == {'cl', 'cd'}
    assert metrics['extrapolation_counts']['aoa_extrapolation'] == 4
    assert 'field_relative_l2' not in metrics
    with (out / 'eval/per_case.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r['case'] for r in rows] == names[10:]
    for key in ['cl', 'cd']:
        errors = [abs(float(r[f'pred_{key}']) - float(r[f'true_{key}'])) / abs(float(r[f'true_{key}'])) for r in rows]
        assert metrics['coefficient_relative_error'][key]['median'] == pytest.approx(np.median(errors))


def test_default_and_explicit_naca_inputs(synthetic_surface):
    surface, names = synthetic_surface
    metadata = json.loads((surface / 'index.json').read_text())['cases']
    expected = np.stack([parameter_features(n, metadata[n]) for n in names])
    np.testing.assert_array_equal(load_arrays(surface, names, metadata)[0], expected)
    np.testing.assert_array_equal(load_arrays(surface, names, metadata, 'naca')[0], expected)


@pytest.mark.parametrize('camber_scale', [0., .03])
def test_geometry_features_analytic_and_duplicate_order(camber_scale):
    k = 20
    stations = .5 * (1 - np.cos(np.pi * np.arange(1, k + 1) / (k + 1)))
    x = np.r_[0., stations, 1.]
    half_thickness = .4 * x * (1 - x)
    camber = camber_scale * np.sin(np.pi * x)
    upper = np.column_stack((2 * x - .01, camber + half_thickness))
    lower = np.column_stack((2 * x - .01, camber - half_thickness))
    pos = np.vstack((upper, lower))
    normal = np.repeat([[0., -1.], [0., 1.]], len(x), axis=0)
    # Unequal duplicates near LE must select each side's outer envelope.
    pos = np.vstack((pos, upper[1] - [0., .001], lower[1] + [0., .001]))
    normal = np.vstack((normal, [0., -1.], [0., 1.]))
    order = np.random.default_rng(0).permutation(len(pos))
    thickness, measured_camber = geometry_features(pos[order], normal[order], k)
    np.testing.assert_allclose(thickness, 2 * half_thickness[1:-1], atol=1e-14)
    np.testing.assert_allclose(measured_camber, camber[1:-1], atol=1e-14)


def test_geometry_interpolates_and_preserves_negative_thickness():
    pos = np.array([[0., .1], [1., .2], [0., -.1], [1., -.2], [.5, 99.]])
    normals = np.array([[0., -1.], [0., -1.], [0., 1.], [0., 1.], [1., 0.]])
    t, c = geometry_features(pos, normals, 3)
    x = .5 * (1 - np.cos(np.pi * np.arange(1, 4) / 4))
    np.testing.assert_allclose(t, .2 + .2 * x)
    np.testing.assert_allclose(c, 0.)
    np.testing.assert_allclose(geometry_features(pos, -normals, 3)[0], -t)
    for stations in (0, -1, 1.5):
        with pytest.raises(ValueError, match='stations'):
            geometry_features(pos, normals, stations)
    with pytest.raises(ValueError, match='surface'):
        geometry_features(pos, np.abs(normals))
    with pytest.raises(ValueError, match='Chord'):
        geometry_features(np.zeros_like(pos), normals)


def test_geometry_inputs_do_not_parse_case_name(synthetic_surface, monkeypatch):
    from surfcpcf import param_regression
    surface, names = synthetic_surface
    name = 'arbitrary_shape'
    (surface / f'{name}.npz').write_bytes((surface / f'{names[0]}.npz').read_bytes())
    def forbidden(*args):
        raise AssertionError('Geometry must not parse NACA parameters')
    monkeypatch.setattr(param_regression, 'parameter_features', forbidden)
    metadata = {name: dict(uinf=30., aoa_deg=-2.)}
    x, _ = load_arrays(surface, [name], metadata, 'geometry', 3)
    assert x.shape == (1, 8)
    np.testing.assert_array_equal(x[0, :2], [30., -2.])


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
    evaluate.main(['--task', 'scarce', '--run', str(out), '--surface', str(surface), '--device', 'cpu'])
    with pytest.raises(SystemExit):
        evaluate.main(['--task', 'naca4to5', '--run', str(out), '--surface', str(surface), '--device', 'cpu'])
