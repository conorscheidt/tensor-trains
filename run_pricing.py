"""Every measured number in "Tensor trains vs. Monte Carlo" (crs.so/writing/tensor-trains).

    E1  accuracy against cost for MC, RQMC and TT, raw and smoothed, d = 5, 10, 20;
        tensor-train cost includes construction
    E2  unfolding spectra of the raw and smoothed integrands, and the rank needed for
        1e-4 as d grows under three correlation structures
    E3  a tensor train as a control variate: bias, variance reduction, coverage
    E4  the 10-asset train itself: cores, grid and model parameters

Results go to results/e{1,2,3,4}*.json. A full run takes about ten minutes on an
Apple M4; `--quick` shrinks every sample size for a smoke test.

    uv run python run_pricing.py [e1 e2 e3 e4] [--quick]
"""

from __future__ import annotations

import json
import platform
import sys
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import ttlib as tt
from basket import Basket, Float, sector_correlation

Json = dict[str, Any]
Integrand = Callable[[Float], Float]

OUT = Path(__file__).resolve().parent / "results"
QUICK = "--quick" in sys.argv


def log(*parts: object) -> None:
    print(*parts, flush=True)


def stable_seed(label: str) -> int:
    """A seed that is the same in every process (str hashes are randomised per process)."""
    return zlib.crc32(label.encode()) % 1000


@dataclass(frozen=True)
class Priced:
    value: float
    evals: int
    secs: float
    cores: tt.Cores
    grid: tt.Grid1D
    stats: tt.CrossStats


def tt_price(fun: Integrand, dim: int, n: int, r: int, *, nsweeps: int = 12, seed: int = 0) -> Priced:
    """TT-cross on the sqrt(w)-scaled integrand over an n^dim Gauss-Hermite grid, then contract.

    The scaling makes the Frobenius error of the train bound the error of the integral, and
    makes maxvol pivot on the entries that carry the integral rather than on extreme nodes.
    The returned cores are unscaled, so they interpolate the integrand itself.
    """
    grid = tt.gauss_hermite(n)
    sq = np.sqrt(grid.q)

    def scaled(idx: tt.Index) -> Float:
        values: Float = fun(grid.x[idx]) * np.prod(sq[idx], axis=1)
        return values

    t0 = time.perf_counter()
    cores, stats = tt.tt_cross(scaled, [n] * dim, r, nsweeps=nsweeps, weights=[sq] * dim, conv_tol=1e-12, seed=seed)
    return Priced(
        value=tt.tt_sum(cores, [sq] * dim),
        evals=stats["evals"],
        secs=time.perf_counter() - t0,
        cores=[g / sq[None, :, None] for g in cores],
        grid=grid,
        stats=stats,
    )


def rqmc(fun: Integrand, dim: int, m: int, reps: int, seed: int = 0) -> tuple[Float, float]:
    """`reps` independent scramblings of a 2^m-point Sobol rule; returns the estimates and seconds per run."""
    estimates = []
    t0 = time.perf_counter()
    for s in range(reps):
        u = qmc.Sobol(dim, scramble=True, seed=seed + 1000 * s).random_base2(m)
        estimates.append(float(fun(ndtri(u)).mean()))
    return np.array(estimates), (time.perf_counter() - t0) / reps


def mc_stats(fun: Integrand, dim: int, n: int, seed: int, chunk: int = 1 << 20) -> tuple[float, float, float]:
    """Mean and standard deviation of fun(Z), Z ~ N(0, I), and seconds per evaluation."""
    rng = np.random.default_rng(seed)
    s1 = s2 = 0.0
    done = 0
    t0 = time.perf_counter()
    while done < n:
        y = fun(rng.standard_normal((min(chunk, n - done), dim)))
        s1 += float(y.sum())
        s2 += float((y * y).sum())
        done += len(y)
    mean = s1 / n
    return mean, float(np.sqrt(max(s2 / n - mean**2, 0.0))), (time.perf_counter() - t0) / n


def cv_estimate(bk: Basket, priced: Priced, n: int, seed: int, chunk: int = 1 << 16) -> tuple[float, float]:
    """Unbiased control-variate estimate I_TT + mean(f - f_TT) over Z ~ N(0, I), with its standard error."""
    rng = np.random.default_rng(seed)
    s1 = s2 = 0.0
    for start in range(0, n, chunk):
        z = rng.standard_normal((min(chunk, n - start), bk.d - 1))
        residual = bk.smoothed(z) - tt.tt_interp_eval(priced.cores, z, [priced.grid] * (bk.d - 1))
        s1 += float(residual.sum())
        s2 += float((residual * residual).sum())
    m = s1 / n
    return priced.value + m, float(np.sqrt((s2 / n - m * m) / (n - 1)))


def reference(bk: Basket) -> tuple[float, Json]:
    """The reference price: a rank-24 train used as a control variate (unbiased) blended by inverse
    variance with randomized QMC, after checking that the two agree within four standard errors."""
    d = bk.d
    priced = tt_price(bk.smoothed, d - 1, 16, 24, nsweeps=16)
    est, se = cv_estimate(bk, priced, 1 << (16 if QUICK else 21), seed=4242)
    runs, _ = rqmc(bk.smoothed, d - 1, 16 if QUICK else 18, 16 if QUICK else 48, seed=99)
    rq, rq_se = float(runs.mean()), float(runs.std(ddof=1) / np.sqrt(len(runs)))
    if abs(est - rq) >= 4 * np.hypot(se, rq_se):
        raise RuntimeError(f"reference disagrees with RQMC at d={d}: {est} vs {rq}")
    wa, wb = 1 / se**2, 1 / rq_se**2
    ref, ref_se = (wa * est + wb * rq) / (wa + wb), (wa + wb) ** -0.5
    log(f"  reference d={d}: TT+CV={est:.10f} +- {se:.1e}  RQMC={rq:.10f} +- {rq_se:.1e}")
    log(f"    combined {ref:.10f} +- {ref_se:.1e}")
    info = {
        "tt": priced.value,
        "tt_cv": est,
        "tt_cv_se": se,
        "rqmc": rq,
        "rqmc_se": rq_se,
        "se": ref_se,
        "relerr_floor": 2 * ref_se / ref,
    }
    return ref, info


# ---------------------------------------------------------------------------- E1


def e1(d: int) -> Json:
    log(f"E1 d={d}")
    bk = Basket(d)
    ref, ref_info = reference(bk)
    methods: Json = {}
    out: Json = {
        "d": d,
        "price": ref,
        "reference": ref_info,
        "geometric_closed_form": bk.geometric_price(),
        "eigenvalues": bk.lam.tolist(),
        "methods": methods,
    }

    # Monte Carlo: the expected RMSE sigma / sqrt(N) of an unbiased estimator, from a large pilot
    counts = [2**k for k in range(8, 25)]
    for key, fun, dim in (("mc_raw", bk.payoff, d), ("mc_smooth", bk.smoothed, d - 1)):
        mean, sd, per = mc_stats(fun, dim, 1 << (20 if QUICK else 22), seed=stable_seed(key))
        methods[key] = {
            "sigma": sd,
            "sec_per_eval": per,
            "pilot_mean": mean,
            "points": [{"evals": n, "relerr": sd / np.sqrt(n) / ref, "secs": per * n} for n in counts],
        }
        log(f"  {key}: sigma/price={sd / ref:.3f}, pilot={mean:.6f}, {per * 1e6:.2f} us/eval")

    # randomized QMC: empirical RMSE over independent scramblings
    for key, fun, dim in (("rqmc_raw", bk.payoff, d), ("rqmc_smooth", bk.smoothed, d - 1)):
        points = []
        for m in range(8, 17 if QUICK else 21, 2):
            est, secs = rqmc(fun, dim, m, 16)
            points.append({"evals": 2**m, "relerr": float(np.sqrt(np.mean((est - ref) ** 2))) / ref, "secs": secs})
        methods[key] = {"points": points}
        log(f"  {key}: " + ", ".join(f"2^{int(np.log2(p['evals']))}:{p['relerr']:.1e}" for p in points))

    # tensor trains: one construction per grid size and rank, every evaluation counted
    smooth_grid = [(6, 2), (6, 4), (8, 4), (8, 6), (10, 6), (10, 8), (12, 8), (12, 10), (14, 12), (16, 16)]
    if d >= 10:
        smooth_grid += [(16, 20), (16, 24), (20, 32)]
    if d >= 20:
        smooth_grid += [(24, 40)]
    raw_grid = [(8, 4), (12, 6), (16, 8), (24, 8), (32, 8), (48, 8), (64, 8)]
    for key, fun, dim, grid in (("tt_smooth", bk.smoothed, d - 1, smooth_grid), ("tt_raw", bk.payoff, d, raw_grid)):
        points = []
        for n, r in grid:
            p = tt_price(fun, dim, n, r)
            points.append(
                {
                    "n": n,
                    "r": r,
                    "evals": p.evals,
                    "secs": p.secs,
                    "value": p.value,
                    "relerr": abs(p.value - ref) / ref,
                    "ranks": tt.tt_ranks(p.cores),
                    "storage": tt.tt_storage(p.cores),
                    "sweeps": len(p.stats["sweeps"]),
                }
            )
            log(f"  {key} n={n} r={r}: rel={abs(p.value - ref) / ref:.1e} evals={p.evals:,} {p.secs:.2f}s")
        methods[key] = {"points": points}
    return out


# ---------------------------------------------------------------------------- E2


def eps_ranks(spectra: list[Float], eps: float) -> list[int]:
    """At each bond, the smallest rank whose discarded tail is at most eps of the whole spectrum."""
    ranks = []
    for s in spectra:
        tail = np.sqrt(np.cumsum((s**2)[::-1]))[::-1] / np.linalg.norm(s)
        ranks.append(int(np.sum(tail > eps)))
    return ranks


def weighted_spectra(p: Priced) -> list[Float]:
    sq = np.sqrt(p.grid.q)
    return tt.tt_spectra([g * sq[None, :, None] for g in p.cores])


def random_corr(d: int, k: int, seed: int = 3) -> Float:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.1, 0.8, size=(d, k))  # positive loadings: every asset rises with the market
    s = w @ w.T + np.diag(rng.uniform(0.2, 1.0, d))
    scale = np.sqrt(np.diag(s))
    result: Float = s / np.outer(scale, scale)
    return result


def e2() -> Json:
    log("E2 rank structure")
    d = 10
    bk = Basket(d)
    cases = []
    for key, label, fun, dim in (("smooth", "smoothed", bk.smoothed, d - 1), ("raw", "raw payoff", bk.payoff, d)):
        spectra = weighted_spectra(tt_price(fun, dim, 16, 24))
        cases.append(
            {
                "key": key,
                "label": label,
                "bond_spectra": [(s / s[0]).tolist() for s in spectra],
                "eps_ranks": {str(e): eps_ranks(spectra, e) for e in (1e-2, 1e-4, 1e-6)},
            }
        )
        log(f"  {key}: eps-ranks at 1e-4 {eps_ranks(spectra, 1e-4)}")

    structures: dict[str, Callable[[int], Float]] = {
        "sector": sector_correlation,
        "one_factor": lambda dd: np.full((dd, dd), 0.5) + 0.5 * np.eye(dd),
        "random_factors": lambda dd: random_corr(dd, k=max(2, dd // 2)),
    }
    scaling: Json = {}
    for name, make in structures.items():
        rows = []
        for dd in [4, 8] if QUICK else [4, 6, 8, 10, 14, 20]:
            p = tt_price(Basket(dd, corr=make(dd)).smoothed, dd - 1, 12, 28)
            spectra = weighted_spectra(p)
            rows.append(
                {
                    "d": dd,
                    "max_rank_1e-4": max(eps_ranks(spectra, 1e-4)),
                    "max_rank_1e-6": max(eps_ranks(spectra, 1e-6)),
                    "evals": p.evals,
                    "secs": p.secs,
                }
            )
            log(f"  {name} d={dd}: rank for 1e-4 = {rows[-1]['max_rank_1e-4']}, evals={p.evals:,}")
        scaling[name] = rows
    return {"d": d, "cases": cases, "scaling": scaling}


# ---------------------------------------------------------------------------- E3


def e3(d: int = 10) -> Json:
    log("E3 control variate")
    bk = Basket(d)
    ref, _ = reference(bk)
    rng = np.random.default_rng(2026)
    n = 4096
    runs = []
    for grid_n, r in ((6, 2), (8, 4), (10, 6), (12, 8), (16, 16)):
        p = tt_price(bk.smoothed, d - 1, grid_n, r)
        z = rng.standard_normal((n, d - 1))
        f = bk.smoothed(z)
        residual = f - tt.tt_interp_eval(p.cores, z, [p.grid] * (d - 1))
        est = p.value + float(residual.mean())
        se = float(residual.std(ddof=1) / np.sqrt(n))
        runs.append(
            {
                "n": grid_n,
                "r": r,
                "setup_evals": p.evals,
                "tt_value": p.value,
                "tt_relerr": abs(p.value - ref) / ref,
                "N": n,
                "estimate": est,
                "se": se,
                "plain_mc_se": float(f.std(ddof=1) / np.sqrt(n)),
                "variance_reduction": float(f.var(ddof=1) / residual.var(ddof=1)),
                "covered": bool(abs(est - ref) < 1.96 * se),
            }
        )
        reduction = runs[-1]["variance_reduction"]
        log(f"  rank {r}: estimate {est:.8f} +- {se:.1e} (reference {ref:.8f}), variance / {reduction:.2e}")
    return {"d": d, "price": ref, "runs": runs}


# ---------------------------------------------------------------------------- E4


def e4(d: int = 10, n: int = 10, r: int = 8) -> Json:
    log("E4 the 10-asset train")
    bk = Basket(d)
    ref, _ = reference(bk)
    p = tt_price(bk.smoothed, d - 1, n, r)
    return {
        "d": d,
        "price": ref,
        "tt_value": p.value,
        "tt_relerr": abs(p.value - ref) / ref,
        "setup_evals": p.evals,
        "setup_secs": p.secs,
        "model": {
            "S0": bk.s0.tolist(),
            "K": bk.strike,
            "T": bk.maturity,
            "r": bk.rate,
            "sigma": bk.sigma.tolist(),
            "w": bk.w.tolist(),
            "mu": bk.mu.tolist(),
            "B": bk.b.tolist(),
            "disc": bk.disc,
            "R": bk.corr.tolist(),
            "eigenvalues": bk.lam.tolist(),
        },
        "grid": {"nodes": p.grid.x.tolist(), "qweights": p.grid.q.tolist(), "bary": p.grid.bw.tolist()},
        "cores": [{"shape": list(g.shape), "data": g.ravel().tolist()} for g in p.cores],
        "sweeps": p.stats["sweeps"],
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta = {
        "machine": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "quick": QUICK,
    }
    chosen = [a for a in sys.argv[1:] if not a.startswith("--")] or ["e1", "e2", "e3", "e4"]
    if "e1" in chosen:
        dims = [5, 10] if QUICK else [5, 10, 20]
        (OUT / "e1_convergence.json").write_text(json.dumps({"meta": meta, "runs": [e1(d) for d in dims]}))
    if "e2" in chosen:
        (OUT / "e2_ranks.json").write_text(json.dumps({"meta": meta, **e2()}))
    if "e3" in chosen:
        (OUT / "e3_control_variate.json").write_text(json.dumps({"meta": meta, **e3()}))
    if "e4" in chosen:
        (OUT / "e4_train.json").write_text(json.dumps({"meta": meta, **e4()}))
    log("done")


if __name__ == "__main__":
    main()
