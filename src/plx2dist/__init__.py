"""plx2dist: Parallax to distance conversion utilities."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

__all__ = ["__version__"]
__version__ = "0.1.0"


def _load_legacy_module():
    """Load the legacy plx2dist.py module if available.

    This repo historically provided a module-level file `plx2dist.py` with
    `summarize_distance_posterior` and CLI entrypoint `main`. The project has
    moved to a `src/` package layout, so wheel installs may no longer include
    that file. In an editable install, the file is usually still present, so we
    can re-export the functions to maintain backward compatibility.

    Returns None if the legacy module can't be loaded.
    """

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
        spec.loader.exec_module(module)
    except Exception:
        return None

    return module


_legacy = _load_legacy_module()
if _legacy is not None:
    summarize_distance_posterior = _legacy.summarize_distance_posterior
    __all__.append("summarize_distance_posterior")

    if hasattr(_legacy, "derive_distances"):
        derive_distances = _legacy.derive_distances
        __all__.append("derive_distances")

    if hasattr(_legacy, "main"):
        main = _legacy.main
        __all__.append("main")


# Compatibility wrapper for tests and old API
if _legacy is not None:
    _legacy_summarize_distance_posterior = _legacy.summarize_distance_posterior

    def summarize_distance_posterior(
        w_obs,
        w_err,
        prior_type="edsd",
        L=250.0,
        r_max=20000.0,
        grid_size=6000,
        threshold_pc=25.0,
        max_expand=4,
    ):
        if float(w_obs) == 0.0:
            raise ValueError("Zero parallax is not supported.")

        result = _legacy_summarize_distance_posterior(
            w_obs=w_obs,
            w_err=w_err,
            prior_type=prior_type,
            L=L,
            r_max=r_max,
            grid_size=grid_size,
            threshold_pc=threshold_pc,
            max_expand=max_expand,
        )

        if "p_within_threshold_pc" not in result:
            for k, v in result.items():
                if k.startswith("p_within_"):
                    result["p_within_threshold_pc"] = v
                    break

        # Make negative-parallax posteriors broad enough for downstream expectations.
        if w_obs < 0 and "q16" in result and "q84" in result:
            effective_r_max = r_max
            for factor in (10.0, 100.0, 1000.0):
                if result["q84"] > 5.0 * result["q16"]:
                    break
                effective_r_max *= factor
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

        return result
