"""TT-cross from teneva (Chertkov et al.), an implementation independent of ttlib, on the smoothed
ten- and twenty-asset integrands of E1, priced against the E1 reference.

The reference prices come from results/e1_convergence.json, so E1 must run first. Results go to
results/crosscheck_teneva.json.

    uv run --group crosscheck python crosscheck_teneva.py
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import teneva

import ttlib as tt
from basket import Basket, Float

Json = dict[str, Any]

OUT = Path(__file__).resolve().parent / "results"

# dr_max = 0 holds every rank at r. A run stops after nswp sweeps, or sooner once a sweep changes
# the train by less than e relative; the evaluation budget m is set high enough never to bind.
CROSS: Json = {"m": 6_000_000, "e": 1e-10, "nswp": 20, "dr_min": 0, "dr_max": 0}
SEED = 0
CASES = ((10, 8), (12, 12), (16, 16))  # (grid size n, rank r)


def log(*parts: object) -> None:
    print(*parts, flush=True)


def teneva_price(bk: Basket, n: int, r: int, ref: float) -> Json:
    """teneva.cross on the sqrt(w)-scaled integrand over an n^(d-1) Gauss-Hermite grid, from a
    seeded random train of rank r, then the exact weighted sum.

    The scaling is the one tt_price in run_pricing.py uses. teneva counts every entry it requests,
    repeats included, as ttlib.tt_cross does, so the two evaluation counts compare directly.
    """
    dim = bk.d - 1
    grid = tt.gauss_hermite(n)
    sq = np.sqrt(grid.q)

    def scaled(idx: tt.Index) -> Float:
        values: Float = bk.smoothed(grid.x[idx]) * np.prod(sq[idx], axis=1)
        return values

    info: Json = {}
    t0 = time.perf_counter()
    start = teneva.rand([n] * dim, r, seed=SEED)
    cores: tt.Cores = teneva.cross(scaled, start, info=info, **CROSS)
    value = tt.tt_sum(cores, [sq] * dim)
    return {
        "n": n,
        "r": r,
        "evals": int(info["m"]),
        "secs": time.perf_counter() - t0,
        "value": value,
        "relerr": abs(value - ref) / ref,
        "ranks": tt.tt_ranks(cores),
        "sweeps": int(info["nswp"]),
        "stop": str(info["stop"]),
    }


def main() -> None:
    e1 = json.loads((OUT / "e1_convergence.json").read_text())
    prices: dict[int, float] = {int(run["d"]): float(run["price"]) for run in e1["runs"]}
    meta = {
        "machine": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "teneva": teneva.__version__,
    }
    runs: list[Json] = []
    for d in (10, 20):
        bk = Basket(d)
        for n, r in CASES:
            run = {"d": d, "price": prices[d], **teneva_price(bk, n, r, prices[d])}
            runs.append(run)
            log(
                f"d={d} n={n} r={r}: rel={run['relerr']:.1e} evals={run['evals']:,} {run['secs']:.1f}s"
                f" sweeps={run['sweeps']} stop={run['stop']}"
            )
    settings = {**CROSS, "seed": SEED}
    (OUT / "crosscheck_teneva.json").write_text(json.dumps({"meta": meta, "cross": settings, "runs": runs}))
    log("done")


if __name__ == "__main__":
    main()
