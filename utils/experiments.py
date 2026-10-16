"""Experiment paths; numerical parameters remain in the original scripts."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Experiment:
    name: str
    script: str
    signal: str

    @property
    def script_path(self):
        return ROOT / self.script

    @property
    def input_paths(self):
        return (
            ROOT / "data" / self.signal,
            ROOT / "data" / "sensor_location.mat",
            ROOT / "data" / "skull" / "skull_mask.mat",
        )


EXPERIMENTS = {
    "a": Experiment("a", "experiments/depth/depth_a.py", "signal_ws_fig4_a.mat"),
    "b": Experiment("b", "experiments/depth/depth_b.py", "signal_ws_fig4_b.mat"),
    "c": Experiment("c", "experiments/depth/depth_c.py", "signal_ws_fig4_c.mat"),
    "plane": Experiment("plane", "experiments/plane/plane.py", "filtered_top_ws.mat"),
}


def validate_inputs(names):
    """Reject missing data before the original scripts can fall back to random inputs."""
    paths = dict.fromkeys(path for name in names for path in EXPERIMENTS[name].input_paths)
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment data:\n  " + "\n  ".join(missing))
