"""Compatibility wrapper for the plx2dist package."""

from plx2dist import (
    build_parser,
    derive_distances,
    load_table_to_dataframe,
    main,
    parallax_to_distance,
    save_dataframe_to_format,
    summarize_distance_posterior,
)

__all__ = [
    "build_parser",
    "derive_distances",
    "load_table_to_dataframe",
    "main",
    "parallax_to_distance",
    "save_dataframe_to_format",
    "summarize_distance_posterior",
]

if __name__ == "__main__":
    raise SystemExit(main())