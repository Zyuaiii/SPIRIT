"""Sequential subprocess execution with per-experiment result directories."""

from datetime import datetime, timezone
import os
import subprocess
import sys

from .experiments import EXPERIMENTS, RESULTS, ROOT, validate_inputs


def run_experiments(names, dry_run=False):
    if dry_run:
        for name in names:
            experiment = EXPERIMENTS[name]
            print(f"{name}: {experiment.script} -> {RESULTS / name / '<run-id>'}")
        return 0

    try:
        validate_inputs(names)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        print("Place signals and detector locations in ./data and the skull mask in ./data/skull; see Readme.md.", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONIOENCODING"] = "utf-8"
    for index, name in enumerate(names, 1):
        destination = RESULTS / name / run_id
        destination.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", "-m", "utils.worker", "--experiment", name, "--output", str(destination)]
        print(f"[{index}/{len(names)}] Running {name}; results: {destination}", flush=True)
        try:
            return_code = subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode
        except KeyboardInterrupt:
            print(f"Interrupted {name}", file=sys.stderr)
            return 130
        if return_code != 0:
            print(f"Experiment {name} failed (exit {return_code})", file=sys.stderr)
            return return_code if return_code > 0 else 1
        print(f"[{index}/{len(names)}] Completed {name}", flush=True)
    return 0
