import argparse
import copy

import numpy as np
import pytest
import torch

from surfcpcf.geometry import coefficients_from_cp_cf as numpy_forces
from surfcpcf.torch_forces import coefficients_from_cp_cf as torch_forces
from surfcpcf.train import force_loss_scales, parse_force_coeffs, training_loss


def square_geometry():
    return dict(normal=torch.tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]], dtype=torch.float64),
                lines=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
                length=torch.ones(4, dtype=torch.float64),
                cell_normal=torch.tensor([[0., -1.], [1., 0.], [0., 1.], [-1., 0.]], dtype=torch.float64),
                aoa_rad=torch.tensor(0., dtype=torch.float64))


def test_torch_closed_square():
    geometry = square_geometry()
    uniform = torch_forces(torch.full((4,), 3., dtype=torch.float64), torch.zeros(4, dtype=torch.float64), **geometry)
    assert all(abs(float(value)) < 1e-14 for value in uniform.values())
    geometry['aoa_rad'] = torch.tensor(np.pi / 2, dtype=torch.float64)
    result = torch_forces(torch.tensor([-.5, .5, .5, -.5], dtype=torch.float64), torch.zeros(4, dtype=torch.float64), **geometry)
    assert float(result['cd']) == pytest.approx(0., abs=1e-14)
    assert float(result['cl']) == pytest.approx(-1.)


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_torch_matches_numpy_and_gradient(seed):
    rng = np.random.default_rng(seed)
    geometry = square_geometry()
    # Deliberately nonunit normals and unequal lengths: never normalize or flip.
    geometry['normal'] = torch.tensor(rng.normal(size=(4, 2)), dtype=torch.float64)
    geometry['length'] = torch.tensor(rng.uniform(.2, 2, size=4), dtype=torch.float64)
    geometry['aoa_rad'] = torch.tensor(rng.uniform(-3, 3), dtype=torch.float64)
    cp = torch.tensor(rng.normal(size=4), dtype=torch.float64, requires_grad=True)
    cf = torch.tensor(rng.normal(size=4), dtype=torch.float64, requires_grad=True)
    expected = numpy_forces(cp.detach().numpy(), cf.detach().numpy(), **{k: v.numpy() for k, v in geometry.items()}, q=37.)
    actual = torch_forces(cp, cf, **geometry)
    for key in expected:
        np.testing.assert_allclose(actual[key].detach().numpy(), expected[key], rtol=1e-10, atol=0)
    sum(value.square() for value in actual.values()).backward()
    assert torch.isfinite(cp.grad).all() and torch.isfinite(cf.grad).all()
    assert cp.grad.abs().sum() > 0 and cf.grad.abs().sum() > 0
    assert torch.autograd.gradcheck(lambda p, f: tuple(torch_forces(p, f, **geometry).values()), (cp, cf))


def test_zero_weight_preserves_loss_gradients_and_adam_step():
    torch.manual_seed(12)
    baseline = torch.nn.Linear(6, 2)
    extended = copy.deepcopy(baseline)
    x, target = torch.randn(4, 6), torch.randn(4, 2)
    old_optimizer = torch.optim.Adam(baseline.parameters(), lr=.001)
    new_optimizer = torch.optim.Adam(extended.parameters(), lr=.001)
    old_loss = (baseline(x) - target).square().mean()
    # Omitted geometry proves no force computation occurs even with invalid force inputs.
    new_loss, field, force = training_loss(extended(x), target, weight=0.)
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    torch.testing.assert_close(field, new_loss, rtol=0, atol=0)
    assert force.item() == 0.
    old_loss.backward()
    new_loss.backward()
    for old, new in zip(baseline.parameters(), extended.parameters()):
        torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
    old_optimizer.step()
    new_optimizer.step()
    for old, new in zip(baseline.parameters(), extended.parameters()):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_force_loss_uses_selected_scales_and_zero_at_truth():
    geometry = square_geometry()
    mean, std = torch.tensor([2., .01]), torch.tensor([.2, .001])
    truth = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=torch.float64)
    physical = truth * std + mean
    coefficients = torch_forces(physical[:, 0], physical[:, 1], **geometry)
    scales = {'cdp': .5, 'clv': .002}
    kwargs = dict(geometry=geometry, coefficients=coefficients, scales=scales, y_mean=mean, y_std=std)
    total, field, force = training_loss(truth, truth, 2., **kwargs)
    assert total == field == force == 0
    pred = truth.clone().requires_grad_(True)
    perturbed = pred * 1.1
    total, field, force = training_loss(perturbed, truth, 2., **kwargs)
    pp = perturbed * std + mean
    actual = torch_forces(pp[:, 0], pp[:, 1], **geometry)
    expected = sum(((actual[k] - coefficients[k]) / scales[k]) ** 2 for k in scales) / len(scales)
    torch.testing.assert_close(force, expected)
    torch.testing.assert_close(total, (perturbed - truth).square().mean() + 2 * expected)
    total.backward()
    assert torch.isfinite(pred.grad).all()


def test_positive_weight_total_combines_field_and_force_losses():
    geometry = square_geometry()
    target = torch.zeros(4, 2, dtype=torch.float64)
    prediction = torch.tensor([[-.5, 0.], [.5, 0.], [.5, 0.], [-.5, 0.]], dtype=torch.float64)
    coefficients = torch_forces(target[:, 0], target[:, 1], **geometry)
    weight = .7
    total, field, force = training_loss(
        prediction, target, weight, geometry=geometry, coefficients=coefficients,
        scales={'cl': .3, 'cd': .2}, y_mean=torch.zeros(2), y_std=torch.ones(2))
    torch.testing.assert_close(field, (prediction - target).square().mean(), rtol=0, atol=0)
    assert field > 0 and force > 0
    torch.testing.assert_close(total, field + weight * force, rtol=0, atol=0)


def test_scale_population_std_and_floor():
    from types import SimpleNamespace
    data = SimpleNamespace(target_coefficients=[{'cl': torch.tensor(v), 'cd': torch.tensor(0.)} for v in (1., 3.)])
    assert force_loss_scales(data, ['cl', 'cd']) == {'cl': 1., 'cd': 1e-8}
    assert parse_force_coeffs('cl, cd,cdp') == ['cl', 'cd', 'cdp']
    for value in ('', 'cl,cl', 'unknown', 'cl,'):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_force_coeffs(value)
