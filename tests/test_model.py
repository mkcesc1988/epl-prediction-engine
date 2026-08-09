import math

from model import poisson_over_25


def test_poisson_over_25_bounds():
    p = poisson_over_25(2.5)
    assert 0 < p < 1


def test_poisson_over_25_increases_with_lambda():
    assert poisson_over_25(3.0) > poisson_over_25(2.0)


def test_poisson_over_25_known_value():
    lam = 2.5
    expected = 1 - math.exp(-lam) * (1 + lam + lam**2 / 2)
    assert abs(poisson_over_25(lam) - expected) < 1e-12
