"""A d-asset Black-Scholes basket call under the risk-neutral measure Q.

    log S_i(T) = log S_i(0) + (r - sigma_i^2 / 2) T + sigma_i sqrt(T) Y_i,   Y ~ N(0, R)
    price      = e^{-rT} E_Q[(sum_i w_i S_i(T) - K)^+] = integral of f(z) phi_d(z) dz

Coordinates are principal components, Y = V Lambda^{1/2} z, so z_1 moves the market.
Every asset loads positively on z_1, so for fixed z_2..z_d the basket is increasing in
z_1 and E[payoff | z_2..z_d] has a closed form once the exercise boundary is found
(Bayer, Siebenmorgen and Tempone, Quant. Finance 18(3), 2018).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

Float = NDArray[np.float64]


def sector_correlation(
    d: int,
    *,
    n_sectors: int = 3,
    rho_in: float = 0.6,
    rho_out: float = 0.3,
    jitter: float = 0.05,
    seed: int = 7,
) -> Float:
    """Equity-like correlation: sector blocks plus a small symmetric perturbation, kept positive definite."""
    rng = np.random.default_rng(seed)
    sector = np.arange(d) % n_sectors
    r: Float = np.where(sector[:, None] == sector[None, :], rho_in, rho_out).astype(np.float64)
    e = rng.normal(size=(d, d)) * jitter
    r = r + (e + e.T) / 2
    np.fill_diagonal(r, 1.0)
    lam, v = np.linalg.eigh(r)
    r = (v * np.maximum(lam, 1e-3)) @ v.T
    s = np.sqrt(np.diag(r))
    normalised: Float = r / np.outer(s, s)
    return normalised


class Basket:
    def __init__(
        self,
        d: int,
        *,
        corr: Float | None = None,
        sigma: Float | None = None,
        s0: float = 100.0,
        strike: float = 100.0,
        maturity: float = 1.0,
        rate: float = 0.03,
        weights: Float | None = None,
    ) -> None:
        self.d = d
        self.strike = strike
        self.maturity = maturity
        self.rate = rate
        self.corr: Float = sector_correlation(d) if corr is None else corr
        self.sigma: Float = np.linspace(0.15, 0.40, d) if sigma is None else np.asarray(sigma, dtype=np.float64)
        self.w: Float = np.full(d, 1.0 / d) if weights is None else np.asarray(weights, dtype=np.float64)
        self.s0: Float = np.full(d, s0)
        lam, v = np.linalg.eigh(self.corr)
        order = np.argsort(lam)[::-1]
        lam, v = lam[order], v[:, order]
        v[:, 0] *= np.sign(v[:, 0].sum())
        self.lam: Float = lam
        self.mu: Float = np.log(self.s0) + (rate - 0.5 * self.sigma**2) * maturity
        self.b: Float = (self.sigma * np.sqrt(maturity))[:, None] * (v * np.sqrt(lam))  # log S_T = mu + b z
        self.disc = float(np.exp(-rate * maturity))
        if not np.all(self.b[:, 0] > 0):
            raise ValueError("every asset must rise with the market factor")

    def payoff(self, z: Float) -> Float:
        """Discounted payoff at standard-normal points z of shape (N, d). Kinked at the strike."""
        s = np.exp(self.mu + z @ self.b.T)
        result: Float = self.disc * np.maximum(s @ self.w - self.strike, 0.0)
        return result

    def smoothed(self, rest: Float, iters: int = 60) -> Float:
        """E[discounted payoff | z_2..z_d] at points `rest` of shape (N, d - 1): analytic in z_1, smooth in the rest."""
        b1 = self.b[:, 0]
        log_c = np.log(self.w) + self.mu + rest @ self.b[:, 1:].T
        log_k = np.log(self.strike)
        t = np.zeros(len(rest))
        for _ in range(iters):  # Newton on the convex log-sum-exp
            e = log_c + b1 * t[:, None]
            m = e.max(axis=1, keepdims=True)
            ex = np.exp(e - m)
            s = ex.sum(axis=1)
            step = (np.log(s) + m[:, 0] - log_k) / ((ex * b1).sum(axis=1) / s)
            t -= step
            if np.max(np.abs(step)) < 1e-13:
                break
        value = (np.exp(log_c + 0.5 * b1**2) * ndtr(b1 - t[:, None])).sum(axis=1) - self.strike * ndtr(-t)
        result: Float = self.disc * np.maximum(value, 0.0)
        return result

    def geometric_price(self) -> float:
        """Closed-form price of the geometric-basket call, a check on the integrators."""
        m = float(self.w @ self.mu)
        cov = np.outer(self.sigma, self.sigma) * self.corr * self.maturity
        var = float(self.w @ cov @ self.w)
        d1 = (m - np.log(self.strike) + var) / np.sqrt(var)
        return float(self.disc * (np.exp(m + var / 2) * ndtr(d1) - self.strike * ndtr(d1 - np.sqrt(var))))

    def geometric_payoff(self, z: Float) -> Float:
        result: Float = self.disc * np.maximum(np.exp((self.mu + z @ self.b.T) @ self.w) - self.strike, 0.0)
        return result
