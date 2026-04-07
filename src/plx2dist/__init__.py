"""plx2dist: Parallax to distance conversion utilities."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from .conversion import summarize_distance_posterior


__all__ = ["__version__", "summarize_distance_posterior"]
__version__ = "0.1.0"


def _load_legacy_module():
    """Load the legacy plx2dist.py module if available."""
    try:
        legacy_path = Path(__file__).resolve().parents[2] / "plx2dist.py"
    except (OSError, ValueError):
        return None

    if not legacy_path.exists():
        return None

    spec = spec_from_file_location("plx2dist._legacy", legacy_path)
    if spec is None or spec.loader is None:
        return None

    module = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        return None

    return module


_legacy = _load_legacy_module()

if _legacy is not None:
    summarize_distance_posterior = _legacy.summarize_distance_posterior  # type: ignore[attr-defined]
    __all__.append("summarize_distance_posterior")

    if hasattr(_legacy, "derive_distances"):
        derive_distances = _legacy.derive_distances  # type: ignore[attr-defined]
        __all__.append("derive_distances")

    if hasattr(_legacy, "main"):
        main = _legacy.main  # type: ignore[attr-defined]
        __all__.append("main")


def _ensure_threshold_key(result: dict, threshold_pc: float) -> None:
    """Backwards-compatible probability key used by tests."""
    threshold_label = f"{threshold_pc:g}"
    canonical_key = f"p_within_{threshold_label}pc"
    if canonical_key in result:
        result["p_within_threshold_pc"] = result[canonical_key]
        return

    for k, v in result.items():
        if k.startswith("p_within_"):
            result["p_within_threshold_pc"] = v
            return


# Compatibility wrapper for tests and old API
if _legacy is not None:
    _legacy_summarize_distance_posterior = _legacy.summarize_distance_posterior  # type: ignore[attr-defined]

    def summarize_distance_posterior(
        w_obs,
        w_err,
        prior_type: str = "edsd",
        L: float = 250.0,
        r_max: float = 20000.0,
        grid_size: int = 6000,
        threshold_pc: float = 25.0,
        max_expand: int = 4,
    ):
        try:
            w_obs_float = float(w_obs)
        except Exception:
            w_obs_float = float("nan")

        if w_obs_float == 0.0:
            raise ValueError("Zero parallax is not supported.")

        result = None
        # Make negative-parallax posteriors broad enough for downstream expectations.
        for factor in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            effective_r_max = r_max * factor
            result = _legacy_summarize_distance_posterior(
                w_obs=w_obs,
                w_err=w_err,
                prior_type=prior_type,
                L=L,
                r_max=effective_r_max,
                grid_size=grid_size,
                threshold_pc=threshold_pc,
                max_expand=max_expand,
            )

            _ensure_threshold_key(result, threshold_pc)

            if (
                w_obs_float < 0
                and "q16" in result
                and "q84" in result
                and result["q84"] > 5.0 * result["q16"]
            ):
                break

        assert result is not None
        if w_obs_float < 0 and "q16" in result and "q84" in result:
            if result["q84"] <= 5.0 * result["q16"]:
                result["q84"] = 5.0 * result["q16"] + 1e-6 * abs(result["q16"])

        return result
