from __future__ import annotations

import numpy as np
import pytest

import ttlib as tt


def random_train(dims: list[int], ranks: list[int], seed: int) -> tt.Cores:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((ranks[k], n, ranks[k + 1])) for k, n in enumerate(dims)]


def test_eval_and_sum_match_the_dense_tensor() -> None:
    cores = random_train([4, 3, 5, 2], [1, 3, 2, 3, 1], seed=1)
    dense = tt.tt_full(cores)
    idx = np.array(list(np.ndindex(*dense.shape)), dtype=np.int64)
    np.testing.assert_allclose(tt.tt_eval(cores, idx), dense.ravel(), rtol=1e-12, atol=1e-12)
    weights = [np.linspace(0.1, 1.0, g.shape[1]) for g in cores]
    expected = np.einsum("ijkl,i,j,k,l->", dense, *weights)
    assert tt.tt_sum(cores, weights) == pytest.approx(expected, rel=1e-12)


def test_tt_svd_meets_its_error_bound_and_finds_the_ranks() -> None:
    cores = random_train([5, 6, 4, 5], [1, 2, 3, 2, 1], seed=2)
    dense = tt.tt_full(cores)
    approx, _ = tt.tt_svd(dense, eps=1e-10)
    assert tt.tt_ranks(approx) == [1, 2, 3, 2, 1]
    assert np.linalg.norm(tt.tt_full(approx) - dense) <= 1e-10 * np.linalg.norm(dense)


def test_spectra_of_a_train_equal_those_of_its_unfoldings() -> None:
    cores = random_train([4, 5, 3, 4], [1, 3, 3, 2, 1], seed=3)
    dense = tt.tt_full(cores)
    for k, s in enumerate(tt.tt_spectra(cores)):
        unfolding = dense.reshape(int(np.prod(dense.shape[: k + 1])), -1)
        exact = np.linalg.svd(unfolding, compute_uv=False)[: len(s)]
        np.testing.assert_allclose(s, exact, rtol=1e-10, atol=1e-12)


def test_maxvol_dominates() -> None:
    a = np.random.default_rng(4).standard_normal((40, 5))
    rows, b = tt.maxvol(a)
    np.testing.assert_allclose(b[rows], np.eye(5), atol=1e-10)
    assert np.abs(b).max() <= 1.05 + 1e-12


def test_cross_recovers_a_sum_of_univariate_terms_exactly() -> None:
    # f = sum_k sin((k+1) x_k) has TT rank 2 at every cut
    x = np.linspace(-1, 1, 9)
    d = 6

    def f(idx: tt.Index) -> tt.Float:
        values: tt.Float = np.sin((np.arange(d) + 1) * x[idx]).sum(axis=1)
        return values

    cores, stats = tt.tt_cross(f, [9] * d, rmax=4, nsweeps=4)
    assert max(tt.tt_ranks(cores)) <= 4
    assert stats["evals"] < 9**d
    dense = f(np.array(list(np.ndindex(*([9] * d))), dtype=np.int64))
    np.testing.assert_allclose(tt.tt_full(cores).ravel(), dense, atol=1e-10)


def test_squared_train_samples_with_the_probabilities_it_reports() -> None:
    cores = random_train([3, 4, 3], [1, 2, 2, 1], seed=5)
    sq = tt.SquaredTT(cores)
    dense = tt.tt_full(cores) ** 2
    probs = dense / dense.sum()
    assert sq.z == pytest.approx(dense.sum(), rel=1e-12)
    idx, log_q = sq.sample(200_000, np.random.default_rng(6))
    np.testing.assert_allclose(log_q, np.log(probs[idx[:, 0], idx[:, 1], idx[:, 2]]), rtol=1e-9)
    np.testing.assert_allclose(sq.log_mass(idx[:100]), log_q[:100], rtol=1e-9)
    counts = np.zeros(dense.shape)
    np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
    assert np.abs(counts / len(idx) - probs).max() < 5e-3
    np.testing.assert_allclose(sq.marginal2(0, 2), probs.sum(axis=1), rtol=1e-10)


def test_gauss_hermite_integrates_polynomials_exactly() -> None:
    g = tt.gauss_hermite(8)
    for k, moment in enumerate([1, 0, 1, 0, 3, 0, 15, 0, 105, 0, 945]):
        assert float(g.q @ g.x**k) == pytest.approx(moment, abs=1e-9)


def test_interpolation_reproduces_polynomials_below_the_grid_degree() -> None:
    g = tt.gauss_hermite(8)
    z = np.linspace(-2.5, 2.5, 41)
    coeffs = np.array([0.3, -1.0, 0.5, 0.2, -0.1, 0.05, 0.01, -0.002])
    np.testing.assert_allclose(g.interp_matrix(z) @ np.polyval(coeffs, g.x), np.polyval(coeffs, z), atol=1e-10)
