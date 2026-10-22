from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import ndtri
from scipy.stats import qmc

from basket import Basket, sector_correlation


def test_correlation_is_a_valid_correlation_matrix() -> None:
    r = sector_correlation(12)
    np.testing.assert_allclose(np.diag(r), 1.0)
    np.testing.assert_allclose(r, r.T)
    assert np.linalg.eigvalsh(r).min() > 0


def test_pre_integration_matches_adaptive_quadrature_from_the_kink() -> None:
    # integrate the kinked payoff over z_1 numerically, starting exactly at the kink found by bisection,
    # independently of the Newton solve inside Basket.smoothed
    bk = Basket(5)
    for row in np.random.default_rng(1).standard_normal((6, 4)):

        def integrand(t: float, row: NDArray[np.float64] = row) -> float:
            density = float(np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi))
            return float(bk.payoff(np.concatenate([[t], row])[None, :])[0]) * density

        lo, hi = -40.0, 40.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if integrand(mid) > 0 else (mid, hi)
        value, _ = quad(integrand, hi, 40.0, epsabs=0, epsrel=1e-12, limit=200)
        assert bk.smoothed(row[None, :])[0] == pytest.approx(value, rel=1e-8)


def test_smoothing_preserves_the_price() -> None:
    bk = Basket(5)
    u4 = qmc.Sobol(4, scramble=True, seed=0).random_base2(16)
    u5 = qmc.Sobol(5, scramble=True, seed=0).random_base2(18)
    assert float(bk.smoothed(ndtri(u4)).mean()) == pytest.approx(float(bk.payoff(ndtri(u5)).mean()), rel=2e-4)


def test_geometric_closed_form_against_quasi_monte_carlo() -> None:
    bk = Basket(5)
    u = qmc.Sobol(5, scramble=True, seed=1).random_base2(18)
    assert float(bk.geometric_payoff(ndtri(u)).mean()) == pytest.approx(bk.geometric_price(), rel=1e-3)
