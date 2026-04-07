import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .conversion import parallax_to_distance, summarize_distance_posterior
from .pipeline import derive_distances, load_table_to_dataframe, save_dataframe_to_format
from .cli import build_parser, main

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "parallax_to_distance",
    "summarize_distance_posterior",
    "derive_distances",
    "load_table_to_dataframe",
    "save_dataframe_to_format",
    "build_parser",
    "main",
]