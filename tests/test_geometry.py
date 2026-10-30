import numpy as np
import pytest

from surfcpcf.geometry import coefficients_from_cp_cf, force_coefficients, reconstruct, tangent


@pytest.fixture
def square():
    pos = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    normals = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    return pos, dict(lines=lines, length=np.ones(4), cell_normal=normals, aoa_rad=0., q=1.)


def test_uniform_pressure_zero_force(square):
    pos, args = square
    result = force_coefficients(np.full(4, 3.), np.zeros_like(pos), **args)
    assert all(abs(value) < 1e-14 for value in result.values())


def test_linear_pressure_analytic_force(square):
    pos, args = square
    result = force_coefficients(pos[:, 0], np.zeros_like(pos), **args)
    assert result["cdp"] == pytest.approx(1.)
    assert result["clp"] == pytest.approx(0.)
    assert result["cdv"] == result["clv"] == 0.


def test_rotation(square):
    pos, args = square
    args["aoa_rad"] = np.pi / 2
    result = force_coefficients(pos[:, 0], np.zeros_like(pos), **args)
    assert result["cd"] == pytest.approx(0., abs=1e-14)
    assert result["cl"] == pytest.approx(-1.)


def test_signed_tangent_and_reconstruction(square):
    pos, args = square
    normals = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    expected = np.array([[0., 1.], [-1., 0.], [0., -1.], [1., 0.]])
    np.testing.assert_array_equal(tangent(normals), expected)
    cp, cf, cf_n, q = np.arange(4.), np.array([-1., 2., -3., 4.]), np.full(4, 0.5), 2.
    p, tau = reconstruct(cp, cf, normals, q, cf_n)
    np.testing.assert_allclose(p, cp * q)
    np.testing.assert_allclose(tau, (cf[:, None] * expected + cf_n[:, None] * normals) * q)
    args["q"] = q
    assert coefficients_from_cp_cf(cp, cf, normals, **args, cf_n=cf_n) == force_coefficients(p, tau, **args)
    _, tau_t = reconstruct(cp, cf, normals, q)
    np.testing.assert_allclose((tau_t * normals).sum(1), 0.)


def test_invalid_pressure_scale(square):
    pos, args = square
    args["q"] = 0.
    with pytest.raises(ValueError, match="q"):
        force_coefficients(pos[:, 0], np.zeros_like(pos), **args)
