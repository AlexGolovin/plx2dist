import numpy as np
from astropy import units as u
from astropy.units import Quantity

from scipy.integrate import cumulative_trapezoid
# from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


def parallax_to_distance(parallax: Quantity) -> Quantity:
    """Convert parallax to distance.

    Parameters
    ----------
    parallax:
        Parallax angle with angular units (e.g. milliarcseconds).

    Returns
    -------
    astropy.units.Quantity
        Distance (parsecs).

    Notes
    -----
    This is a simple 1/p conversion (distance in parsecs for parallax in arcseconds).
    It does not apply priors or correct for measurement uncertainties.
    """
    parallax_arcsec = parallax.to(u.arcsecond)
    return (1.0 / parallax_arcsec.value) * u.parsec

# -----------------------------------------------------------------------------
# 1. MATHEMATICAL MODEL
# -----------------------------------------------------------------------------

def log_prior(r: np.ndarray, prior_type: str = "edsd", L: float = 250.0, r_max: float = 20000.0) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    lp = np.full_like(r, -np.inf, dtype=float)
    valid = (r > 0.0) & (r < r_max)
    if not np.any(valid):
        return lp

    rv = r[valid]
    if prior_type == "edsd":
        # p(r|L) = (1 / 2L^3) r^2 exp(-r/L), r > 0
        lp[valid] = 2.0 * np.log(rv) - (rv / L) - np.log(2.0) - 3.0 * np.log(L)
    elif prior_type == "volume":
        # Truncated constant space-density prior on (0, r_max)
        lp[valid] = 2.0 * np.log(rv) + np.log(3.0) - 3.0 * np.log(r_max)
    else:
        raise ValueError(f"Unknown prior: {prior_type}")
    return lp


def log_likelihood(r: np.ndarray, w_obs: float, w_err: float) -> np.ndarray:
    # Gaussian likelihood in parallax space, parallax in mas, distance in pc.
    w_true = 1000.0 / np.asarray(r, dtype=float)
    return -0.5 * ((w_obs - w_true) / w_err) ** 2 - np.log(w_err) - 0.5 * np.log(2.0 * np.pi)


def log_posterior_grid(r: np.ndarray, w_obs: float, w_err: float, prior_type: str, L: float, r_max: float) -> np.ndarray:
    return log_prior(r, prior_type=prior_type, L=L, r_max=r_max) + log_likelihood(r, w_obs, w_err)


# -----------------------------------------------------------------------------
# 2. GRID HELPERS
# -----------------------------------------------------------------------------

def _initial_r_upper(w_obs: float, w_err: float, L: float, user_r_max: float) -> float:
    scales = [user_r_max, 10.0 * L, 100.0]
    if np.isfinite(w_err) and w_err > 0:
        scales.append(20.0 * (1000.0 / w_err))
    if np.isfinite(w_obs) and w_obs > 0:
        scales.append(10.0 * (1000.0 / w_obs))
    return max(scales)


def _make_log_grid(r_min: float, r_max: float, grid_size: int) -> np.ndarray:
    return np.geomspace(float(r_min), float(r_max), int(grid_size))


def _normalize_posterior(r: np.ndarray, log_post: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    finite = np.isfinite(log_post)
    if not np.any(finite):
        raise RuntimeError("Posterior is non-finite everywhere.")

    shifted = np.full_like(log_post, -np.inf, dtype=float)
    shifted[finite] = log_post[finite] - np.max(log_post[finite])
    unnorm = np.exp(shifted)
    norm = np.trapezoid(unnorm, r)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Posterior normalization failed.")

    pdf = unnorm / norm
    cdf = np.zeros_like(pdf)
    cdf[1:] = cumulative_trapezoid(pdf, r)
    cdf /= cdf[-1]
    return pdf, cdf, norm


def _interp_quantile(r: np.ndarray, cdf: np.ndarray, q: float) -> float:
    return float(np.interp(float(np.clip(q, 0.0, 1.0)), cdf, r))


def _p_less_than(r: np.ndarray, cdf: np.ndarray, threshold_pc: float) -> float:
    if threshold_pc <= r[0]:
        return 0.0
    if threshold_pc >= r[-1]:
        return 1.0
    return float(np.interp(threshold_pc, r, cdf))


def _boundary_is_problematic(pdf: np.ndarray, cdf: np.ndarray) -> bool:
    peak = float(np.nanmax(pdf))
    tail_density_ratio = (pdf[-1] / peak) if peak > 0 else np.inf
    tail_mass_last_decade = 1.0 - cdf[int(0.9 * len(cdf))]
    return (tail_density_ratio > 1e-5) or (tail_mass_last_decade > 1e-4)


def _refine_bounds_from_cdf(r: np.ndarray, cdf: np.ndarray) -> Tuple[float, float]:
    """Choose narrower bounds that safely contain essentially all posterior mass.

    We start from extreme quantiles and then pad multiplicatively. This is much
    more accurate for narrow high-S/N posteriors than reusing one giant global
    grid, while still keeping low-S/N tails if they are genuinely present.
    """
    q_lo = _interp_quantile(r, cdf, 1e-6)
    q_hi = _interp_quantile(r, cdf, 1.0 - 1e-6)

    # multiplicative padding on log scale
    r_lo = max(r[0], q_lo / 2.5)
    r_hi = min(r[-1], q_hi * 2.5)

    # numerical safeguard in case the posterior is extremely narrow
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_lo <= 0 or r_hi <= r_lo:
        r_mode = r[np.argmax(np.diff(np.r_[0.0, cdf]))]
        r_lo = max(r[0], r_mode / 10.0)
        r_hi = min(r[-1], r_mode * 10.0)

    # Prevent over-collapse for moderate/poor data.
    min_span_dex = 0.5  # at least factor ~3.16 in total span
    current_span_dex = np.log10(r_hi) - np.log10(r_lo)
    if current_span_dex < min_span_dex:
        mid = np.sqrt(r_lo * r_hi)
        half = min_span_dex / 2.0
        r_lo = max(r[0], mid / (10.0 ** half))
        r_hi = min(r[-1], mid * (10.0 ** half))

    return float(r_lo), float(r_hi)

def _threshold_label(threshold_pc: float) -> str:
    if float(threshold_pc).is_integer():
        return f"{int(threshold_pc)}pc"
    return f"{threshold_pc:g}pc"


def summarize_distance_posterior(
    w_obs: float,
    w_err: float,
    prior_type: str = "edsd",
    L: float = 250.0,
    r_max: float = 20000.0,
    grid_size: int = 6000,
    threshold_pc: float = 25.0,
    max_expand: int = 4,
) -> Dict[str, float]:
    if not np.isfinite(w_obs) or not np.isfinite(w_err) or (w_err <= 0):
        raise ValueError("Invalid parallax or parallax uncertainty.")

    current_r_max = float(_initial_r_upper(w_obs, w_err, L=L, user_r_max=r_max))
    r_broad = None
    pdf_broad = None
    cdf_broad = None

    for _ in range(max_expand + 1):
        r_broad = _make_log_grid(1e-3, current_r_max, max(2000, grid_size // 2))
        log_post_broad = log_posterior_grid(
            r_broad,
            w_obs=w_obs,
            w_err=w_err,
            prior_type=prior_type,
            L=L,
            r_max=current_r_max,
        )
        pdf_broad, cdf_broad, _ = _normalize_posterior(r_broad, log_post_broad)
        if not _boundary_is_problematic(pdf_broad, cdf_broad):
            break
        current_r_max *= 5.0

    r_lo, r_hi = _refine_bounds_from_cdf(r_broad, cdf_broad)
    r_refined = _make_log_grid(r_lo, r_hi, grid_size)
    log_post_refined = log_posterior_grid(
        r_refined,
        w_obs=w_obs,
        w_err=w_err,
        prior_type=prior_type,
        L=L,
        r_max=current_r_max,
    )
    pdf, cdf, _ = _normalize_posterior(r_refined, log_post_refined)

    lost_low = cdf[0]
    lost_high = 1.0 - cdf[-1]
    if (lost_low > 1e-10) or (lost_high > 1e-10):
        r_refined, pdf, cdf = r_broad, pdf_broad, cdf_broad

    mode = float(r_refined[np.argmax(pdf)])
    q05 = _interp_quantile(r_refined, cdf, 0.05)
    q16 = _interp_quantile(r_refined, cdf, 0.16)
    q50 = _interp_quantile(r_refined, cdf, 0.50)
    q84 = _interp_quantile(r_refined, cdf, 0.84)
    q95 = _interp_quantile(r_refined, cdf, 0.95)
    threshold_label = _threshold_label(threshold_pc)
    p_within = _p_less_than(r_refined, cdf, threshold_pc)

    return {
        "mode": mode,
        "q05": q05,
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "q95": q95,
        "err_lo": q50 - q16,
        "err_hi": q84 - q50,
        f"p_within_{threshold_label}": p_within,
        "p_within_threshold_pc": p_within,
        "r_grid_min_used": float(r_refined[0]),
        "r_grid_max_used": float(r_refined[-1]),
    }

# -----------------------------------------------------------------------------
# 4. WORKER FUNCTIONS
# -----------------------------------------------------------------------------

def _plot_single_diagnostic(star_id, w_obs, w_err, prior, summary, L, threshold_pc):
    r = _make_log_grid(summary["r_grid_min_used"], max(summary["r_grid_max_used"], summary["r_grid_min_used"] * 1.001), 5000)
    log_post = log_posterior_grid(r, w_obs=w_obs, w_err=w_err, prior_type=prior, L=L, r_max=summary["r_grid_max_used"])
    pdf, cdf, _ = _normalize_posterior(r, log_post)
    threshold_label = _threshold_label(threshold_pc)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(r, pdf, color="k", lw=1.3)
    for key, ls in [("mode", "-"), ("q16", "--"), ("q50", "-"), ("q84", "--")]:
        ax.axvline(summary[key], color="C1" if key in {"mode", "q50"} else "0.5", linestyle=ls, lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Distance (pc)")
    ax.set_ylabel("Posterior density")
    ax.set_title(
        f"ID {star_id} | {prior.upper()} | w={w_obs:.3f}±{w_err:.3f} mas\n"
        f"L={L:.1f} pc, P(<{threshold_pc}pc)={summary[f'p_within_{threshold_label}']:.3e}"
    )
    plt.tight_layout()
    plt.savefig(f"distance_plots/object_id_{star_id}_{prior}.png", dpi=120)
    plt.close(fig)


def _process_star_chunk(args: Tuple) -> Dict:
    (chunk_idx, star_ids, parallaxes, parallax_errs, priors, L, r_max, grid_size, threshold_pc, verbose, plot_diagnostics) = args

    n_stars = len(star_ids)
    threshold_label = _threshold_label(threshold_pc)
    results = {
        prior: {
            "mode": np.full(n_stars, np.nan),
            "q05": np.full(n_stars, np.nan),
            "q16": np.full(n_stars, np.nan),
            "q50": np.full(n_stars, np.nan),
            "q84": np.full(n_stars, np.nan),
            "q95": np.full(n_stars, np.nan),
            "err_lo": np.full(n_stars, np.nan),
            "err_hi": np.full(n_stars, np.nan),
            f"p_within_{threshold_label}": np.full(n_stars, np.nan),
            "p_within_threshold_pc": np.full(n_stars, np.nan),
            "r_grid_min_used": np.full(n_stars, np.nan),
            "r_grid_max_used": np.full(n_stars, np.nan),
        }
        for prior in priors
    }
    log_buffer = []

    for i in range(n_stars):
        t0 = time.time()
        s_id = star_ids[i]
        w = parallaxes[i]
        w_err = parallax_errs[i]

        is_invalid = np.ma.is_masked(w) or np.ma.is_masked(w_err) or np.isnan(w) or np.isnan(w_err) or (w_err <= 0)
        if is_invalid:
            if verbose:
                log_buffer.append(f"Skipping {s_id}: invalid or missing parallax data.")
            continue

        if verbose:
            log_buffer.append("-" * 60)
            log_buffer.append(f"Object ID: {s_id}")
            log_buffer.append(f"Parallax: {float(w):.6f} ± {float(w_err):.6f} mas")
            if w > 0:
                log_buffer.append(f"Naive 1000/parallax: {1000.0 / float(w):.6f} pc")
            else:
                log_buffer.append("Naive 1000/parallax: undefined (non-positive parallax)")

        for prior in priors:
            summary = summarize_distance_posterior(
                w_obs=float(w),
                w_err=float(w_err),
                prior_type=prior,
                L=L,
                r_max=r_max,
                grid_size=grid_size,
                threshold_pc=threshold_pc,
            )
            for key in results[prior]:
                results[prior][key][i] = summary[key]

            if plot_diagnostics:
                _plot_single_diagnostic(s_id, float(w), float(w_err), prior, summary, L=L, threshold_pc=threshold_pc)

            if verbose:
                log_buffer.append(f"  [{prior.upper()} prior]")
                log_buffer.append(
                    f"    q05={summary['q05']:.5f} | q16={summary['q16']:.5f} | q50={summary['q50']:.5f} | "
                    f"q84={summary['q84']:.5f} | q95={summary['q95']:.5f}"
                )
                log_buffer.append(
                    f"    mode={summary['mode']:.5f} pc | P(r<{threshold_pc} pc)={summary[f'p_within_{threshold_label}']:.5e} | "
                    f"grid=[{summary['r_grid_min_used']:.5g}, {summary['r_grid_max_used']:.5g}] pc"
                )

        if verbose:
            log_buffer.append(f"  Runtime: {time.time() - t0:.3f} s")

    return {"chunk_idx": chunk_idx, "results": results, "logs": "\n".join(log_buffer)}


# -----------------------------------------------------------------------------
# 5. PIPELINE MANAGER
# -----------------------------------------------------------------------------
def _validate_input(
    data: Union[pd.DataFrame, "Table"],
    plx_col: str,
    err_col: str,
    id_col: Optional[str],
) -> None:
    """Check that required columns exist and have numeric dtype.

    Raises
    ------
    KeyError
        If a required column is missing.
    TypeError
        If parallax or error columns are not numeric.
    """
    is_astropy = hasattr(data, "colnames")
    cols = data.colnames if is_astropy else list(data.columns)

    for col, label in [(plx_col, "parallax"), (err_col, "parallax error")]:
        if col not in cols:
            raise KeyError(
                f"Required {label} column '{col}' not found in input data. "
                f"Available columns: {cols}. "
                f"Use the plx_col= / err_col= arguments to specify the correct names."
            )
        if is_astropy:
            dtype = data[col].dtype
        else:
            dtype = data[col].dtype
        if not np.issubdtype(dtype, np.number):
            raise TypeError(
                f"Column '{col}' must be numeric, but has dtype '{dtype}'. "
                f"Convert it before calling derive_distances()."
            )

    if id_col is not None and id_col not in cols:
        import warnings
        warnings.warn(
            f"ID column '{id_col}' not found. Falling back to integer row index. "
            f"Available columns: {cols}.",
            UserWarning,
            stacklevel=3,
        )

def derive_distances(
    input_data: Union[pd.DataFrame, "Table"],
    id_col: str = "cns5_id",
    plx_col: str = "parallax",
    err_col: str = "parallax_error",
    priors: Optional[List[str]] = None,
    L: float = 250.0,
    r_max: float = 20000.0,
    threshold_pc: float = 25.0,
    n_jobs: int = 1,
    grid_size: int = 6000,
    verbose: bool = False,
    plot_diagnostics: bool = False,
    output_prefix: Optional[str] = None,
) -> Union[pd.DataFrame, "Table"]:
    out = input_data.copy()
    n_total = len(out)

    is_astropy = hasattr(out, "colnames")
    cols = out.colnames if is_astropy else out.columns

    _validate_input(out, plx_col=plx_col, err_col=err_col, id_col=id_col)

    star_ids = np.asarray(out[id_col]) if id_col in cols else np.arange(n_total)
    parallaxes = np.asarray(out[plx_col])
    parallax_errs = np.asarray(out[err_col])
    if priors is None:
        priors = ["edsd"]
    threshold_label = _threshold_label(threshold_pc)

    metrics = [
        "mode", "q05", "q16", "q50", "q84", "q95",
        "err_lo", "err_hi",
        "p_within_threshold_pc",          # stable, always present
        f"p_within_{threshold_label}",    # label-specific, for pipeline/plots
        "r_grid_min_used", "r_grid_max_used",
    ]

    # precreate columns so schema is deterministic (even for empty catalogs)
    for prior in priors:
        prefix = f"distance_{prior}"
        for metric in metrics:
            col_name = f"{prefix}_{metric}"
            if col_name not in cols:
                out[col_name] = np.nan
                if is_astropy:
                    cols = out.colnames

            prior_L_col = f"{prefix}_prior_L_pc"
            if prior_L_col not in cols:
                out[prior_L_col] = np.nan
                if is_astropy:
                    cols = out.colnames
    
    if n_total == 0:
        if output_prefix is not None:
            csv_out = f"{output_prefix}.csv"
            vot_out = f"{output_prefix}.vot"
            if is_astropy:
                out.to_pandas().to_csv(csv_out, index=False)
                out.write(vot_out, format="votable", overwrite=True)
            else:
                out.to_csv(csv_out, index=False)
                if Table is not None:
                    Table.from_pandas(out).write(vot_out, format="votable", overwrite=True)
        return out

    if plot_diagnostics:
        os.makedirs("distance_plots", exist_ok=True)
        print("Diagnostic plotting enabled. Plots will be saved to 'distance_plots/'.")

    n_jobs = min(max(int(n_jobs), 1), max(n_total, 1))
    chunk_size = math.ceil(n_total / n_jobs)
    chunks = [range(i, min(i + chunk_size, n_total)) for i in range(0, n_total, chunk_size)]

    worker_args = []
    for idx, chunk in enumerate(chunks):
        worker_args.append((
            idx,
            star_ids[chunk],
            parallaxes[chunk],
            parallax_errs[chunk],
            priors,
            L,
            r_max,
            grid_size,
            threshold_pc,
            verbose,
            plot_diagnostics,
        ))

    print(f"--- Starting CNS6 adaptive-grid distance pipeline ({n_total} stars, {n_jobs} cores) ---")
    t_start = time.time()
    completed_rows = 0

    _progress = (
        _tqdm(total=n_total, unit="star", desc="plx2dist")
        if _TQDM_AVAILABLE and not verbose
        else None
    )

    Executor = cf.ProcessPoolExecutor if n_jobs > 1 else cf.ThreadPoolExecutor
    with Executor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_process_star_chunk, args): args for args in worker_args}
        for future in cf.as_completed(futures):
            res = future.result()
            chunk_indices = chunks[res["chunk_idx"]]

            if verbose and res["logs"]:
                print(res["logs"])

            for prior in priors:
                prefix = f"distance_{prior}"
                for metric in [
                    "mode", "q05", "q16", "q50", "q84", "q95",
                    "err_lo", "err_hi", f"p_within_{threshold_label}", "r_grid_min_used", "r_grid_max_used"
                ]:
                    col_name = f"{prefix}_{metric}"
                    if col_name not in cols:
                        out[col_name] = np.nan
                        if is_astropy:
                            cols = out.colnames

                if f"{prefix}_prior_L_pc" not in cols:
                    out[f"{prefix}_prior_L_pc"] = np.nan
                    if is_astropy:
                        cols = out.colnames

                if is_astropy:
                    for metric in [
                        "mode", "q05", "q16", "q50", "q84", "q95",
                        "err_lo", "err_hi", f"p_within_{threshold_label}", "r_grid_min_used", "r_grid_max_used"
                    ]:
                        out[f"{prefix}_{metric}"][chunk_indices] = res["results"][prior][metric]
                    out[f"{prefix}_prior_L_pc"][chunk_indices] = L
                else:
                    pd_indices = out.index[chunk_indices]
                    for metric in [
                        "mode", "q05", "q16", "q50", "q84", "q95",
                        "err_lo", "err_hi", f"p_within_{threshold_label}", "r_grid_min_used", "r_grid_max_used"
                    ]:
                        out.loc[pd_indices, f"{prefix}_{metric}"] = res["results"][prior][metric]
                    out.loc[pd_indices, f"{prefix}_prior_L_pc"] = L

            completed_rows += len(chunk_indices)
            if _progress is not None:
                _progress.update(len(chunk_indices))
            elif not verbose:
                print(f"Progress: {completed_rows}/{n_total} objects processed...")

    if _progress is not None:
        _progress.close()

    print("-" * 60)
    print(f"Total runtime: {time.time() - t_start:.2f} s")

    if output_prefix is not None:
        csv_out = f"{output_prefix}.csv"
        vot_out = f"{output_prefix}.vot"
        if is_astropy:
            out.to_pandas().to_csv(csv_out, index=False)
            out.write(vot_out, format="votable", overwrite=True)
        else:
            out.to_csv(csv_out, index=False)
            if Table is not None:
                Table.from_pandas(out).write(vot_out, format="votable", overwrite=True)
        print(f"Exported final tables to:\n  -> {csv_out}\n  -> {vot_out}")

    return out


# -----------------------------------------------------------------------------
# 6. DATA I/O HELPERS
# -----------------------------------------------------------------------------

def load_table_to_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".fits", ".fit", ".fz"}:
        if Table is None:
            raise ImportError("astropy is required to read FITS tables")
        return Table.read(path).to_pandas()
    if suffix in {".vo", ".vot", ".xml"}:
        if Table is None:
            raise ImportError("astropy is required to read VOTables")
        return Table.read(path).to_pandas()
    raise ValueError(f"Unsupported input table format: {path}")


def save_dataframe_to_format(df: pd.DataFrame, output_path: str) -> None:
    """Write a DataFrame to CSV, VOTable, or FITS based on the file extension.

    Supported extensions: .csv, .vo, .vot, .xml (VOTable), .fits, .fit.
    Falls back to .csv if the extension is unrecognised.
    """
    suffix = Path(output_path).suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix in {".vo", ".vot", ".xml", ".fits", ".fit"}:
        if Table is None:
            raise ImportError("astropy is required to write astropy tables")
        t = Table.from_pandas(df)
        fmt = "votable" if suffix in {".vo", ".vot", ".xml"} else "fits"
        t.write(output_path, format=fmt, overwrite=True)
    else:
        df.to_csv(output_path + ".csv", index=False)


# -----------------------------------------------------------------------------
# 7. CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plx2dist",
        description=(
            "plx2dist: Bayesian parallax-to-distance pipeline.\n"
            "Computes geometric posterior distances from trigonometric parallax.\n"
            "Developed for CNS6; general-purpose for any parallax-based catalogue."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=str, required=True, help="Input catalog (VOTable/CSV/FITS)")
    p.add_argument("--output-prefix", type=str, required=True, help="Prefix for output files")
    p.add_argument("--priors", type=str, nargs="+", default=["edsd"], help="Priors to compute (edsd, volume)")
    p.add_argument("--L", type=float, default=250.0, help="EDSD scale length in pc (recommended CNS6 default: 250)")
    p.add_argument("--r-max", type=float, default=20000.0, help="Initial computational upper bound in pc")
    p.add_argument("--grid-size", type=int, default=6000, help="Number of grid points in refined posterior grid")
    p.add_argument(
        "--threshold-pc",
        type=float,
        default=25.0,
        help="Distance threshold in pc for P(r < threshold) output column (default: 25.0)",
    )
    p.add_argument("--n-jobs", type=int, default=1, help="Number of parallel worker processes")
    p.add_argument("--verbose", action="store_true", help="Print detailed per-star logs")
    p.add_argument("--plot-diagnostics", action="store_true", help="Save per-star posterior plots")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"Loading {args.input}...")
    df = load_table_to_dataframe(args.input)

    out = derive_distances(
        df,
        priors=args.priors,
        L=args.L,
        r_max=args.r_max,
        threshold_pc=args.threshold_pc,
        grid_size=args.grid_size,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        plot_diagnostics=args.plot_diagnostics,
        output_prefix=args.output_prefix,
    )

    
    print(f"\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())