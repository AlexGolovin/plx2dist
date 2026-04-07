# plx2dist

**Bayesian parallax-to-distance pipeline for stellar catalogues.**

Trigonometric parallax is an unbiased distance estimator only at high S/N. At
moderate or low S/N — or for the many Gaia stars with *negative* parallaxes —
naive inversion (d = 1/ϖ) fails badly. `plx2dist` solves this by computing
the full posterior p(r | ϖ, σ) on an adaptive grid, combining a Gaussian
parallax likelihood with a physically motivated distance prior (EDSD or
uniform volume density). It was developed for CNS6 (The Sixth Catalogue of Nearby Stars) but is general-purpose for any Gaia-era parallax catalogue.

## Installation
```bash
pip install plx2dist               # core
pip install plx2dist[progress]     # + tqdm progress bar
```

## Quick start (Python API)
```python
import pandas as pd
from plx2dist import derive_distances

df = pd.read_csv("my_catalogue.csv")   # must contain 'parallax', 'parallax_error'

if __name__ == "__main__":
    result_df = derive_distances(df, plx_col="parallax", err_col="parallax_error", n_jobs=4)
```

## Quick start (command line)
```bash
plx2dist --input catalogue.vot \
         --output-prefix results \
         --priors edsd \
         --n-jobs 8
```

## Output columns

For each prior (e.g. `edsd`), the following columns are added:

| Column | Description |
|---|---|
| `distance_edsd_mode` | Posterior mode (pc) |
| `distance_edsd_q05` | 5th percentile (pc) |
| `distance_edsd_q16` | 16th percentile — lower 1σ bound (pc) |
| `distance_edsd_q50` | Median distance (pc) — **recommended point estimate** |
| `distance_edsd_q84` | 84th percentile — upper 1σ bound (pc) |
| `distance_edsd_q95` | 95th percentile (pc) |
| `distance_edsd_err_lo` | q50 − q16 (pc) |
| `distance_edsd_err_hi` | q84 − q50 (pc) |
| `distance_edsd_p_within_25pc` | P(r < 25 pc \| ϖ, σ) — membership probability; threshold can be specified by user |
| `distance_edsd_r_grid_min_used` | Diagnostic: refined grid lower bound (pc) |
| `distance_edsd_r_grid_max_used` | Diagnostic: refined grid upper bound (pc) |
| `distance_edsd_prior_L_pc` | EDSD scale length used (pc) |

Use `q50` as the point estimate and `err_lo`/`err_hi` as the asymmetric uncertainty.
`mode` tends to be slightly closer to the likelihood peak; prefer `q50` for catalogue work.

## Method

The pipeline uses a two-stage adaptive log-r grid:
1. A broad safety pass locates posterior support, with automatic boundary expansion for low-S/N stars.
2. A refined pass concentrates grid points around the posterior mass for accurate quantile estimation.

No MCMC. Fully deterministic and parallelised via Python `multiprocessing`.

## Priors

| Name | Formula | Default L |
|---|---|---|
| `edsd` | p(r) ∝ r² exp(−r/L) | 250 pc (tuned for CNS6 volume) |
| `volume` | p(r) ∝ r² (uniform space density, truncated) | — |

For the solar neighbourhood (e.g., ≲ 25 pc) the EDSD prior with L = 250 pc is recommended.

## Reference

If you use `plx2dist`, please cite:

- Golovin et al. (in prep.) — CNS6 catalogue and distance methodology

BibTeX for CNS6:
```bibtex
@article{Golovin2026,
  author  = {Golovin, A. et al.},
  title   = {{The Sixth Catalogue of Nearby Stars (CNS6)}},
  journal = {A\&A},
  year    = {2026},
  volume  = {NNN},
  pages   = {ANN},
  doi     = {TBD}
}
```