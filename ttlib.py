"""Tensor trains: TT-SVD, maxvol, TT-cross, contraction, exact sampling from a
squared train, and interpolation on one-dimensional quadrature grids.

A train is a list of cores G_k of shape (r_{k-1}, n_k, r_k) with r_0 = r_d = 1:

    T[i_1, ..., i_d] = G_1[:, i_1, :] @ G_2[:, i_2, :] @ ... @ G_d[:, i_d, :]

References
    Oseledets, Tensor-train decomposition, SIAM J. Sci. Comput. 33(5), 2011.
    Oseledets and Tyrtyshnikov, TT-cross approximation for multidimensional arrays,
        Linear Algebra Appl. 432(1), 2010.
    Goreinov et al., How to find a good submatrix, 2010.
    Dolgov, Anaya-Izquierdo, Fox and Scheichl, Stat. Comput. 30, 2020.
    Cui and Dolgov, Found. Comput. Math. 22(6), 2022.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

Float = NDArray[np.float64]
Index = NDArray[np.int64]
Cores = list[Float]
BlackBox = Callable[[Index], Float]


# ---------------------------------------------------------------------------- basics


def tt_eval(cores: Sequence[Float], idx: Index) -> Float:
    """The train at integer multi-indices `idx` of shape (B, d)."""
    v: Float = cores[0][0, idx[:, 0], :]
    for k in range(1, len(cores)):
        v = np.einsum("br,brs->bs", v, cores[k][:, idx[:, k], :].transpose(1, 0, 2))
    return v[:, 0]


def tt_full(cores: Sequence[Float]) -> Float:
    """The dense tensor. Small cases only."""
    t: Float = cores[0][0]
    for g in cores[1:]:
        t = np.tensordot(t, g, axes=([-1], [0]))
    return t[..., 0]


def tt_ranks(cores: Sequence[Float]) -> list[int]:
    return [1] + [int(g.shape[2]) for g in cores]


def tt_storage(cores: Sequence[Float]) -> int:
    return sum(int(g.size) for g in cores)


def tt_sum(cores: Sequence[Float], weights: Sequence[Float] | None = None) -> float:
    """sum over i of w_1(i_1) ... w_d(i_d) T[i], in O(d n r^2): collapse each core, then multiply."""
    v: Float = np.ones(1)
    for k, g in enumerate(cores):
        w = np.ones(g.shape[1]) if weights is None else weights[k]
        v = v @ np.einsum("ais,i->as", g, w)
    return float(v[0])


def tt_svd(a: Float, eps: float = 1e-10, rmax: int | None = None) -> tuple[Cores, list[Float]]:
    """TT-SVD (Oseledets 2011, algorithm 1), with ||A - T||_F <= eps ||A||_F.

    Returns the cores and the singular values of each unfolding.
    """
    shape, d = a.shape, a.ndim
    delta = eps / np.sqrt(d - 1) * float(np.linalg.norm(a))
    cores: Cores = []
    spectra: list[Float] = []
    r = 1
    c = a.reshape(1, -1)
    for k in range(d - 1):
        c = c.reshape(r * shape[k], -1)
        u, s, vt = np.linalg.svd(c, full_matrices=False)
        spectra.append(s)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]  # tail[j] = ||s[j:]||
        rk = max(1, int(np.sum(tail > delta)))
        if rmax is not None:
            rk = min(rk, rmax)
        cores.append(u[:, :rk].reshape(r, shape[k], rk))
        c = s[:rk, None] * vt[:rk]
        r = rk
    cores.append(c.reshape(r, shape[-1], 1))
    return cores, spectra


def tt_spectra(cores: Sequence[Float]) -> list[Float]:
    """Singular values of every unfolding, in O(d n r^3) and without forming the tensor."""
    g = [core.copy() for core in cores]
    d = len(g)
    for k in range(d - 1):  # left-orthogonalise
        r0, n, r1 = g[k].shape
        q, rr = np.linalg.qr(g[k].reshape(r0 * n, r1))
        g[k] = q.reshape(r0, n, -1)
        g[k + 1] = np.einsum("ab,bis->ais", rr, g[k + 1])
    spectra: list[Float] = [np.empty(0)] * (d - 1)
    for k in range(d - 1, 0, -1):  # sweep back, reading each bond's singular values
        r0, n, r1 = g[k].shape
        u, s, vt = np.linalg.svd(g[k].reshape(r0, n * r1), full_matrices=False)
        spectra[k - 1] = s
        g[k] = vt.reshape(-1, n, r1)
        g[k - 1] = np.einsum("ais,sb->aib", g[k - 1], u * s)
    return spectra


# ---------------------------------------------------------------------------- maxvol


def maxvol(a: Float, tol: float = 1.05, max_iters: int = 200) -> tuple[Index, Float]:
    """Rows I of a tall (N, r) matrix whose r x r submatrix has locally maximal volume.

    Returns I and A @ inv(A[I]), every entry of which is at most `tol` in modulus.
    """
    n_rows, r = a.shape
    if n_rows <= r:
        return np.arange(n_rows, dtype=np.int64), np.eye(n_rows)
    _, _, piv = qr(a.T, pivoting=True, mode="economic")
    rows: Index = piv[:r].astype(np.int64)
    b: Float = np.linalg.solve(a[rows].T, a.T).T
    for _ in range(max_iters):
        i, j = np.unravel_index(int(np.argmax(np.abs(b))), b.shape)
        if abs(b[i, j]) <= tol:
            break
        rows[j] = i
        bj = b[:, j].copy()
        bi = b[i, :].copy()
        bi[j] -= 1.0
        b -= np.outer(bj, bi / b[i, j])
    return rows, b


# ---------------------------------------------------------------------------- TT-cross


class Sweep(TypedDict):
    evals: int
    ranks: list[int]
    integral: float


class CrossStats(TypedDict):
    evals: int
    sweeps: list[Sweep]


def tt_cross(
    fun: BlackBox,
    n: Sequence[int],
    rmax: int,
    *,
    nsweeps: int = 6,
    rank_tol: float = 1e-14,
    conv_tol: float = 1e-6,
    weights: Sequence[Float] | None = None,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> tuple[Cores, CrossStats]:
    """Alternating one-site TT-cross with maxvol pivots, SVD rank truncation and random kicks.

    `fun` evaluates the black box at integer multi-indices of shape (B, d). A sweep asks for
    O(d n rmax^2) entries and never the full n^d. Iteration stops once the weighted sum, the
    integral estimate, changes by less than `conv_tol` relative.
    """
    d = len(n)
    rng = np.random.default_rng(seed)
    stats: CrossStats = {"evals": 0, "sweeps": []}

    def fibers(left: Index, k: int, right: Index) -> Float:
        idx = np.empty((len(left), n[k], len(right), d), dtype=np.int64)
        if k > 0:
            idx[..., :k] = left[:, None, None, :]
        idx[..., k] = np.arange(n[k])[None, :, None]
        if k < d - 1:
            idx[..., k + 1 :] = right[None, None, :, :]
        flat = idx.reshape(-1, d)
        stats["evals"] += len(flat)
        return fun(flat).reshape(len(left), n[k], len(right))

    def kick(right: Index, k: int, target: int) -> Index:
        """Pad the right index set of bond k with random multi-indices up to `target` rows."""
        extra = target - len(right)
        if extra <= 0:
            return right
        fresh = np.column_stack([rng.integers(0, n[j], size=extra) for j in range(k, d)])
        if len(right) == 0:
            return fresh
        merged: Index = np.unique(np.vstack([right, fresh]), axis=0)
        return merged[:target]

    def truncate(m: Float) -> Float:
        u, s, _ = np.linalg.svd(m, full_matrices=False)
        r = max(1, int(np.sum(s > rank_tol * s[0]))) if s[0] > 0 else 1
        return u[:, :r]

    empty = np.zeros((1, 0), np.int64)
    lefts: list[Index] = [empty] * d
    rights: list[Index] = [empty] * (d + 1)
    for k in range(d - 1, 0, -1):
        rights[k] = kick(np.zeros((0, d - k), np.int64), k, rmax)
    cores: Cores = [np.empty(0)] * d
    prev: float | None = None

    for sweep in range(nsweeps):
        for k in range(d - 1):  # left to right
            c = fibers(lefts[k], k, rights[k + 1])
            r0, nk, r1 = c.shape
            sel, b = maxvol(truncate(c.reshape(r0 * nk, r1)))
            cores[k] = b.reshape(r0, nk, -1)
            lefts[k + 1] = np.column_stack([lefts[k][sel // nk], sel % nk])
        cores[d - 1] = fibers(lefts[d - 1], d - 1, rights[d])
        for k in range(d - 1, 0, -1):  # right to left
            c = fibers(lefts[k], k, rights[k + 1])
            r0, nk, r1 = c.shape
            sel, b = maxvol(truncate(c.reshape(r0, nk * r1).T))
            cores[k] = b.T.reshape(-1, nk, r1)
            rights[k] = np.column_stack([sel // r1, rights[k + 1][sel % r1]])
        cores[0] = fibers(lefts[0], 0, rights[1])
        value = tt_sum(cores, weights)
        stats["sweeps"].append({"evals": stats["evals"], "ranks": tt_ranks(cores), "integral": value})
        if log:
            log(f"  sweep {sweep}: ranks={tt_ranks(cores)} evals={stats['evals']:,} integral={value:.6e}")
        if prev is not None and abs(value - prev) <= conv_tol * abs(value):
            break
        prev = value
        if sweep < nsweeps - 1:
            for k in range(1, d):
                rights[k] = kick(rights[k], k, rmax)
    return cores, stats


# ---------------------------------------------------------------------------- exact sampling


class SquaredTT:
    """The density proportional to (T[i] s_1(i_1) ... s_d(i_d))^2 on a product grid of cells.

    Squaring makes it nonnegative by construction (Cui and Dolgov 2022), so every
    conditional p(i_k | i_<k) is an exact contraction: a draw is an exact sample from
    the approximation and its probability is known, which is what importance weights
    and Metropolis-Hastings acceptance ratios need.
    """

    def __init__(self, cores: Sequence[Float], scale: Sequence[Float] | None = None) -> None:
        self.d = len(cores)
        self.cores: Cores = [g * (1.0 if scale is None else scale[k][None, :, None]) for k, g in enumerate(cores)]
        # P[k] = sum over i_k..i_d of (G_k .. G_d)(G_k .. G_d)^T, rescaled against overflow
        self.right: list[Float] = [np.ones((1, 1))] * (self.d + 1)
        log_scale = 0.0
        for k in range(self.d - 1, -1, -1):
            g = self.cores[k]
            pk: Float = np.einsum("ais,st,bit->ab", g, self.right[k + 1], g)
            c = float(np.abs(pk).max())
            self.right[k] = pk / c
            log_scale += float(np.log(c))
        self.log_z = float(np.log(self.right[0][0, 0])) + log_scale

    @property
    def z(self) -> float:
        """The normalising constant: the sum of (T s)^2 over all cells."""
        return float(np.exp(self.log_z))

    def log_mass(self, idx: Index) -> Float:
        """log of the normalised probability (T s)^2 / Z of the cells `idx`."""
        v: Float = self.cores[0][0, idx[:, 0], :]
        log_norm = np.zeros(len(idx))
        for k in range(1, self.d):
            v = np.einsum("br,brs->bs", v, self.cores[k][:, idx[:, k], :].transpose(1, 0, 2))
            size = np.abs(v).max(axis=1, keepdims=True) + 1e-300
            v = v / size
            log_norm += np.log(size[:, 0])
        result: Float = 2 * (np.log(np.abs(v[:, 0]) + 1e-300) + log_norm) - self.log_z
        return result

    def sample(self, count: int, rng: np.random.Generator) -> tuple[Index, Float]:
        """`count` cells by the conditional-distribution method, O(count d n r^2).

        Returns the cells (count, d) and the log of each cell's probability.
        """
        v: Float = np.ones((count, 1))
        idx = np.zeros((count, self.d), dtype=np.int64)
        log_q = np.zeros(count)
        rows = np.arange(count)
        for k in range(self.d):
            u: Float = np.einsum("br,rns->bns", v, self.cores[k])
            p: Float = np.maximum(np.einsum("bns,bns->bn", u @ self.right[k + 1], u), 0.0)
            p /= p.sum(axis=1, keepdims=True)
            i = np.minimum((np.cumsum(p, axis=1) < rng.random((count, 1))).sum(axis=1), p.shape[1] - 1)
            idx[:, k] = i
            log_q += np.log(p[rows, i])
            v = u[rows, i, :]
            v /= np.linalg.norm(v, axis=1, keepdims=True)
        return idx, log_q

    def marginal2(self, a: int, b: int) -> Float:
        """The exact two-dimensional marginal over coordinates a < b, shape (n_a, n_b)."""
        env: Float = np.ones((1, 1))
        for k in range(a):
            g = self.cores[k]
            env = np.einsum("ab,ais,bit->st", env, g, g)
        ga = self.cores[a]
        x: Float = np.einsum("iist->ist", np.einsum("ab,ais,bjt->ijst", env, ga, ga))
        for k in range(a + 1, b):
            g = self.cores[k]
            x = np.einsum("ist,sku,tkv->iuv", x, g, g)
        gb = self.cores[b]
        m: Float = np.einsum("ijuv,uv->ij", np.einsum("ist,sju,tjv->ijuv", x, gb, gb), self.right[b + 1])
        return m / m.sum()


# ---------------------------------------------------------------------------- grids


class Grid1D:
    """Nodes, Gaussian quadrature weights (integrals against phi) and barycentric weights."""

    def __init__(self, nodes: Float, weights: Float) -> None:
        self.x: Float = np.asarray(nodes, dtype=np.float64)
        self.q: Float = np.asarray(weights, dtype=np.float64)
        self.n = len(self.x)
        gaps = self.x[:, None] - self.x[None, :]
        np.fill_diagonal(gaps, 1.0)
        log_w = -np.log(np.abs(gaps)).sum(axis=1)
        self.bw: Float = np.prod(np.sign(gaps), axis=1) * np.exp(log_w - log_w.max())

    def interp_matrix(self, z: Float) -> Float:
        """Rows are the Lagrange basis l_j(z) through the nodes, in barycentric form: (len(z), n)."""
        diff = z[:, None] - self.x[None, :]
        exact = np.abs(diff) < 1e-14
        diff[exact] = 1.0
        m: Float = self.bw[None, :] / diff
        m /= m.sum(axis=1, keepdims=True)
        hit = exact.any(axis=1)
        m[hit] = exact[hit].astype(np.float64)
        return m


def gauss_hermite(n: int) -> Grid1D:
    """The probabilists' Gauss-Hermite rule: sum_j q_j f(x_j) ~ E f(Z), Z ~ N(0, 1), exact to degree 2n - 1."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return Grid1D(x, w / w.sum())


def tt_interp_eval(cores: Sequence[Float], z: Float, grids: Sequence[Grid1D]) -> Float:
    """The train's interpolant at arbitrary points z of shape (B, d), in O(B d n r^2)."""
    v: Float = grids[0].interp_matrix(z[:, 0]) @ cores[0][0]
    for k in range(1, len(cores)):
        v = np.einsum("br,bi,ris->bs", v, grids[k].interp_matrix(z[:, k]), cores[k], optimize=True)
    return v[:, 0]
