"""Run one unchanged experiment, then export its results (internal entry point)."""

import argparse
import os
from pathlib import Path
import runpy

from .experiments import EXPERIMENTS, ROOT, validate_inputs
from .outputs import photoacoustic_parameters, save_outputs, write_reconstruction_log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    experiment = EXPERIMENTS[args.experiment]
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    validate_inputs([experiment.name])
    # Preserve the original relative data paths, including when launched elsewhere.
    os.chdir(ROOT)
    os.environ["MPLBACKEND"] = "Agg"
    namespace = runpy.run_path(str(experiment.script_path), run_name="__main__")
    parameters = photoacoustic_parameters(namespace)
    reconstruction_time = namespace["t_start"].elapsed_time(namespace["t_end"]) / 1000.0
    write_reconstruction_log(destination, parameters, reconstruction_time)
    save_outputs(namespace, destination, experiment.name)
    print(f"Saved reconstruction.mat, reconstruction.png, original figures, and console.log to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
