import argparse
from typing import Optional, Sequence

from .pipeline import derive_distances, load_table_to_dataframe


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

    derive_distances(
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

    print("\nPipeline complete.")
    return 0