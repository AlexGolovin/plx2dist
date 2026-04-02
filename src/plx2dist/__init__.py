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
