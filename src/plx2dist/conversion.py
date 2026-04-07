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