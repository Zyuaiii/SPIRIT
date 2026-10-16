"""Run the published experiments from one entry point."""

import argparse

from utils.experiments import EXPERIMENTS
from utils.runner import run_experiments


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run SPIRIT experiments and save images/MAT files to results/.")
    parser.add_argument(
        "--experiment", "-e", nargs="+", choices=["all", *EXPERIMENTS], default=["all"],
        help="Choose a, b, c, plane, or all (default). Multiple experiments are allowed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show the execution plan without loading data or running GPU code.")
    args = parser.parse_args(argv)
    if "all" in args.experiment and len(args.experiment) != 1:
        parser.error("Use all by itself, or list individual experiments.")
    names = list(EXPERIMENTS) if args.experiment == ["all"] else list(dict.fromkeys(args.experiment))
    return run_experiments(names, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
