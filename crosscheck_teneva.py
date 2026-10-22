"""An independent TT-cross (teneva, Chertkov et al.) on the same smoothed integrands, so the
slowdown at d = 20 can be attributed to the problem rather than to ttlib.

    uv run --group crosscheck python crosscheck_teneva.py
"""

from __future__ import annotations

import time

import numpy as np
import teneva  # type: ignore[import-untyped]

import ttlib as tt
from basket import Basket, Float

# reference prices from results/e1_convergence.json
REFERENCE = {10: 8.729465156, 20: 8.552177701}


def main() -> None:
    for d, ref in REFERENCE.items():
        bk = Basket(d)
        dim = d - 1
        for n, r in ((10, 8), (12, 12), (16, 16)):
            grid = tt.gauss_hermite(n)
            sq = np.sqrt(grid.q)

            def scaled(idx: tt.Index, bk: Basket = bk, grid: tt.Grid1D = grid, sq: Float = sq) -> Float:
                index = np.asarray(idx)
                values: Float = bk.smoothed(grid.x[index]) * np.prod(sq[index], axis=1)
                return values

            t0 = time.perf_counter()
            start = teneva.rand([n] * dim, r, seed=0)
            cores = teneva.cross(scaled, start, m=6_000_000, e=1e-10, nswp=20, dr_min=0, dr_max=0)
            value = tt.tt_sum(cores, [sq] * dim)
            print(f"d={d} n={n} r={r}: relative error {abs(value - ref) / ref:.1e} ({time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
