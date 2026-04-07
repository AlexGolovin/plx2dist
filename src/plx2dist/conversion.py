from __future__ import annotations

from astropy import units as u
from astropy.units import Quantity

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
    """Deterministically summarize the posterior p(r | parallax, sigma).

    Method
    ------
    1. Construct a broad initial log-r grid.
    2. Evaluate likelihood x prior and normalize to get pdf/cdf.
    3. Expand the upper boundary if the posterior still carries mass there.
    4. Build a refined log-r grid around the posterior support.
    5. Recompute normalized pdf/cdf on the refined grid and extract summaries.

    Returns
    -------
    mode, q05, q16, q50, q84, q95, err_lo, err_hi, p_within_threshold_pc,
    r_grid_min_used, r_grid_max_used
    """
    if not np.isfinite(w_obs) or not np.isfinite(w_err) or (w_err <= 0):
        raise ValueError("Invalid parallax or parallax uncertainty.")

    # ----- Stage 1: broad safety grid -----
    current_r_max = float(_initial_r_upper(w_obs, w_err, L=L, user_r_max=r_max))
    r_broad = None
    pdf_broad = None
    cdf_broad = None

    for _ in range(max_expand + 1):
        r_broad = _make_log_grid(1e-3, current_r_max, max(2000, grid_size // 2))
        log_post_broad = log_posterior_grid(r_broad, w_obs=w_obs, w_err=w_err, prior_type=prior_type, L=L, r_max=current_r_max)
        pdf_broad, cdf_broad, _ = _normalize_posterior(r_broad, log_post_broad)
        if not _boundary_is_problematic(pdf_broad, cdf_broad):
            break
        current_r_max *= 5.0

    # ----- Stage 2: refined local grid -----
    r_lo, r_hi = _refine_bounds_from_cdf(r_broad, cdf_broad)
    r_refined = _make_log_grid(r_lo, r_hi, grid_size)
    log_post_refined = log_posterior_grid(r_refined, w_obs=w_obs, w_err=w_err, prior_type=prior_type, L=L, r_max=current_r_max)
    pdf, cdf, _ = _normalize_posterior(r_refined, log_post_refined)

    # Sanity check: if refinement clipped non-negligible mass, fall back to broad grid
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
        "threshold_pc": threshold_pc,
        "r_grid_min_used": float(r_refined[0]),
        "r_grid_max_used": float(r_refined[-1]),
    }