import numpy as np
import pytest

import lumicks.pyoptics.mathutils.integration as integration


@pytest.mark.parametrize("method", ("Gauss-Legendre", "Clenshaw-Curtis", "Lebedev-Laikov"))
def test_determine_integration_order_wrong_method(method):
    with pytest.raises(ValueError, match=f"Wrong type of integration method specified: {method}"):
        integration.determine_integration_order(method, 1)


@pytest.mark.parametrize("n_orders", (-10, -1.0, -0.0001))
def test_determine_integration_order_wrong_order(n_orders):

    with pytest.raises(
        ValueError, match=f"Invalid value for n_orders. Must be > 0, got {n_orders}"
    ):
        integration.determine_integration_order("gauss-legendre", n_orders)


@pytest.mark.parametrize("n_orders", np.arange(1, 50))
@pytest.mark.parametrize("method", ("lebedev-laikov", "gauss-legendre", "clenshaw-curtis"))
def test_determine_integration_order(method, n_orders):

    assert integration.determine_integration_order(method, n_orders) > n_orders


@pytest.mark.parametrize(
    "integration_order", ([3 + n * 2 for n in range(15)] + [35 + n * 6 for n in range(17)])
)
@pytest.mark.parametrize("method", ("lebedev-laikov", "gauss-legendre", "clenshaw-curtis"))
def test_get_integration_locations(integration_order: int, method: str):
    x, y, z, w = integration.get_integration_locations(integration_order, method)
    R = np.hypot(np.hypot(x, y), z)
    np.testing.assert_allclose(R, np.ones(R.shape))

    if method in ["lebedev-laikov", "clenshaw-curtis"]:
        assert w.sum() == pytest.approx(1.0)

    if method == "clenshaw-curtis":
        np.testing.assert_allclose((np.min(z), np.max(z)), (-1.0, 1.0))


def test_clenshaw_curtis_weights():
    # Numbers from on "On the Method for Numerical Integration of Clenshaw and Curtis" by J. P.
    # Imhof
    _, w = integration.clenshaw_curtis_weights(16)

    np.testing.assert_allclose(
        w[1:9],
        (0.0373687, 0.07548231, 0.108906, 0.138956, 0.163173, 0.181474, 0.192514, 0.196410),
        atol=1e-6,
    )
    np.testing.assert_allclose(w[1:8], w[15:8:-1])


@pytest.mark.parametrize("integration_order", np.arange(-2, 2))
def test_clenshaw_curtis_weights_raises(integration_order: int):
    with pytest.raises(ValueError, match="Only integration orders of 2 and higher are supported"):
        integration.clenshaw_curtis_weights(integration_order)


@pytest.mark.parametrize("method", ("lebedev-laikov", "gauss-legendre", "clenshaw-curtis"))
@pytest.mark.parametrize("order", (7, 11))
def test_integration_result(method: str, order: int):
    x, y, z, w = integration.get_integration_locations(order, method)

    # Fornberg, B., Martel, J.M. On spherical harmonics based numerical quadrature over the surface
    # of a sphere. Adv Comput Math 40, 1169–1184 (2014). https://doi.org/10.1007/s10444-014-9346-3
    integral = (1 + x + y**2 + x**2 * y + x**4 + y**5 + x**2 * y**2 * z**2) * w
    assert 4 * np.pi * integral.sum() == pytest.approx(216 * np.pi / 35)

    integral = (1 + np.arctan(-9 * x - 9 * y + 9 * z)) * w / 9.0
    assert integral.sum() == pytest.approx(1.0 / 9.0)

    # Simple check
    integrals = [(ax**2 * w).sum() for ax in (x, y, z)]
    np.testing.assert_allclose(integrals, [1.0 / 3.0] * 3)


@pytest.mark.parametrize(
    "r_inner, r_outer, result",
    [(0.0, 1.0, np.pi / 2), (0.0, 2.0, 8 * np.pi), (1.0, 2.0, 15 * np.pi / 2)],
)
def test_annular_rule_exact(r_inner, r_outer, result):
    x, y, w = integration.annulus_rule(2, r_inner=r_inner, r_outer=r_outer)
    assert ((x**2 + y**2) * w).sum() == pytest.approx(result)


def test_annular_rule_sin():
    result = -0.0213167  # Wolfram Alpha
    x, y, w = integration.annulus_rule(13, r_inner=0, r_outer=1)
    assert ((x * np.sin(10 * np.pi * x)) * w).sum() == pytest.approx(result)
    assert ((y * np.sin(10 * np.pi * y)) * w).sum() == pytest.approx(result)


@pytest.mark.parametrize("r_inner, r_outer", [(0.0, 0.5), (0, 1), (1, 2), (np.exp(1), np.pi)])
def test_annular_rule_cos(r_inner, r_outer):
    def integral(r):
        return (2 * np.pi) * (
            r**2 / 4
            + r * np.sin(4 * np.pi * r) / (8 * np.pi)
            + np.cos(4 * np.pi * r) / (32 * np.pi**2)
        )

    x, y, w = integration.annulus_rule(8, r_inner=r_inner, r_outer=r_outer)
    r = np.hypot(x, y)
    assert (np.cos(2 * np.pi * r) ** 2 * w).sum() == pytest.approx(
        integral(r_outer) - integral(r_inner)
    )


@pytest.mark.parametrize(
    "r_inner, r_outer, result",
    [
        (0.0, 1.0, (np.exp(1) - 1) * np.pi / np.exp(1)),
        (1.0, 2.0, (np.exp(3) - 1) * np.pi / np.exp(4)),
        (0.0, np.exp(1.0), np.pi * (1 - np.exp(-np.exp(2)))),
    ],
)
def test_annular_rule_exp(r_inner, r_outer, result):
    x, y, w = integration.annulus_rule(6, r_inner=r_inner, r_outer=r_outer)
    assert ((np.exp(-(x**2) - y**2)) * w).sum() == pytest.approx(result)
