import concurrent.futures as cf
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from astropy.table import Table
except ImportError:
    Table = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional progress bar — works without tqdm installed
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from .conversion import (
    _make_log_grid,
    _normalize_posterior,
    _threshold_label,
    log_posterior_grid,
    summarize_distance_posterior,
)


def _plot_single_diagnostic(star_id, w_obs, w_err, prior, summary, L, threshold_pc):
    r = _make_log_grid(
        summary["r_grid_min_used"],
        max(summary["r_grid_max_used"], summary["r_grid_min_used"] * 1.001),
        5000,
    )
    log_post = log_posterior_grid(
        r,
        w_obs=w_obs,
        w_err=w_err,
        prior_type=prior,
        L=L,
        r_max=summary["r_grid_max_used"],
    )
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
    (
        chunk_idx,
        star_ids,
        parallaxes,
        parallax_errs,
        priors,
        L,
        r_max,
        grid_size,
        threshold_pc,
        verbose,
        plot_diagnostics,
    ) = args

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

        is_invalid = (
            np.ma.is_masked(w)
            or np.ma.is_masked(w_err)
            or np.isnan(w)
            or np.isnan(w_err)
            or (w_err <= 0)
        )
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


def _validate_input(
    data: Union[pd.DataFrame, "Table"],
    plx_col: str,
    err_col: str,
    id_col: Optional[str],
) -> None:
    """Check that required columns exist and have numeric dtype."""
    is_astropy = hasattr(data, "colnames")
    cols = data.colnames if is_astropy else list(data.columns)

    for col, label in [(plx_col, "parallax"), (err_col, "parallax error")]:
        if col not in cols:
            raise KeyError(
                f"Required {label} column '{col}' not found in input data. "
                f"Available columns: {cols}. "
                f"Use the plx_col= / err_col= arguments to specify the correct names."
            )
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
        "p_within_threshold_pc",
        f"p_within_{threshold_label}",
        "r_grid_min_used", "r_grid_max_used",
    ]

    # Precreate columns so schema is deterministic, even for empty catalogs
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

                if is_astropy:
                    for metric in metrics:
                        out[f"{prefix}_{metric}"][chunk_indices] = res["results"][prior][metric]
                    out[f"{prefix}_prior_L_pc"][chunk_indices] = L
                else:
                    pd_indices = out.index[chunk_indices]
                    for metric in metrics:
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
    """Write a DataFrame to CSV, VOTable, or FITS based on the file extension."""
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