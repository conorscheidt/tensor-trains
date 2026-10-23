# tensor-trains

Experiments for [Tensor trains vs. Monte Carlo](https://crs.so/writing/tensor-trains). The code prices a basket call by building a tensor train of its integrand with TT-cross and summing the train exactly, then compares that with Monte Carlo and randomized quasi-Monte Carlo run on the same integrands.

| File | Contents |
| --- | --- |
| `ttlib.py` | TT-SVD, `maxvol`, TT-cross, contraction, exact sampling from a squared train, interpolation on Gauss–Hermite grids |
| `basket.py` | The d-asset basket under Q in principal-component coordinates: the payoff, its pre-integrated form, and the geometric basket in closed form |
| `run_pricing.py` | Experiments E1–E4, written to `results/` |
| `crosscheck_teneva.py` | The d = 10 and d = 20 constructions repeated with [teneva](https://github.com/AndreiChertkov/teneva), written to `results/crosscheck_teneva.json` |
| `results/` | Output of a full run on an Apple M4 |
| `tests/` | Exactness and error-bound checks for `ttlib` and `basket` |

## Running

The flake provides uv, and `uv.lock` pins the Python dependencies.

```sh
nix develop                                # or: direnv allow
uv run pytest
uv run python run_pricing.py               # E1–E4, about ten minutes
uv run python run_pricing.py e3 --quick    # one experiment, reduced sample sizes
uv run --group crosscheck python crosscheck_teneva.py   # after E1, whose reference prices it uses
```

Every random stream has a fixed seed, so on the same platform a rerun reproduces `results/` apart from the timing fields.

## Experiments

**E1.** Relative error against the number of integrand evaluations for d = 5, 10 and 20, for Monte Carlo, randomized QMC and tensor trains, each on the raw and on the pre-integrated payoff.

**E2.** Singular values of every unfolding of the ten-asset integrand, raw and smoothed, and the rank needed for a relative error of 10⁻⁴ as d grows under three correlation structures.

**E3.** Trains of rank 2 to 16 used as control variates for Monte Carlo: estimates, standard errors, variance reduction and coverage of the reference.

**E4.** The ten-asset train shown in the post: its cores, grid and model parameters.

**Cross-check.** teneva's TT-cross on the smoothed integrands for d = 10 and 20 at fixed ranks 8, 12 and 16, with at most 20 sweeps, priced against the E1 reference.

## Conventions

Cost is counted in evaluations of the integrand, and for tensor trains it includes every evaluation requested during construction, repeats included, since neither `ttlib` nor teneva as called here caches entries it has already requested. The Monte Carlo error is the expected root-mean-square error σ/√N, with σ estimated from a 2²²-sample pilot. The quasi-Monte Carlo error is the root-mean-square error over 16 independent scramblings of a Sobol sequence. The tensor-train error is that of a single construction at each grid size and rank.

The reference price combines a rank-24 train used as a control variate with randomized QMC, weighted by inverse variance, once the two agree within four standard errors. Its relative standard error is below 3 × 10⁻⁷ for every d.

## License

[MIT](LICENSE)
