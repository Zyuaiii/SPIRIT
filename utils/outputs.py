"""Save completed reconstructions without changing the experiment's timed region."""

def write_reconstruction_log(destination, parameters, reconstruction_time_s=None):
    lines = [f"{name}: {value}" for name, value in parameters.items()]
    if reconstruction_time_s is not None:
        lines.append(f"reconstruction_time_s: {reconstruction_time_s:.6f}")
    (destination / "console.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def photoacoustic_parameters(namespace):
    return {
        "num_detectors": int(namespace["num_detectors"]),
        "num_time_samples": int(namespace["num_times"]),
        "grid_shape_xyz": [int(namespace[f"num_{axis}"]) for axis in "xyz"],
        "num_voxels": int(namespace["num_voxels"]),
        "sampling_rate_hz": float(namespace["fs"]),
        "voxel_size_m": float(namespace["res"]),
        **{
            f"{axis}_range_m": [float(namespace[f"{axis}_start"]), float(namespace[f"{axis}_end"])]
            for axis in "xyz"
        },
    }


def save_outputs(namespace, destination, experiment_name):
    # All dependencies and tensor transfers are used only after run_path has returned.
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.io as sio

    shape = (namespace["num_x"], namespace["num_y"], namespace["num_z"])
    volume = namespace["x_reconstructed"].reshape(*shape).detach().cpu().numpy()
    mip = np.max(volume, axis=2)
    center = volume[:, :, shape[2] // 2]
    resolution = namespace["res"]
    axes_m = {
        axis: namespace[f"{axis}_start"] + np.arange(length) * resolution
        for axis, length in zip("xyz", shape)
    }
    reconstruction_time = namespace["t_start"].elapsed_time(namespace["t_end"]) / 1000.0

    # Preserve the original figures, which were created after the first reconstruction.
    original_figures = []
    for index, number in enumerate(plt.get_fignums(), 1):
        filename = f"original_figure_{index}.png"
        plt.figure(number).savefig(destination / filename, dpi=200, bbox_inches="tight")
        original_figures.append(filename)

    mat_contents = {
        "reconstruction": volume,
        "mip_reconstruction": mip,
        "center_slice": center,
        "x_m": axes_m["x"], "y_m": axes_m["y"], "z_m": axes_m["z"],
        "res_m": resolution,
        "fs_hz": namespace["fs"],
        "reconstruction_time_s": reconstruction_time,
        "losses": np.asarray(namespace["losses"]),
        "experiment": experiment_name,
        "original_mip_recon": namespace["mip_recon"],
        "original_center_slice": namespace["center_slice"],
    }
    sio.savemat(destination / "reconstruction.mat", mat_contents, do_compression=True, oned_as="row")

    # This separate figure and the MAT file consistently use the second reconstruction.
    extent = [namespace[f"{axis}_{bound}"] * 1e3 for axis in "xy" for bound in ("start", "end")]
    figure, panels = plt.subplots(1, 2, figsize=(16, 6))
    for panel, image, title in zip(
        panels, (mip, center),
        (f"Experiment {experiment_name}: Z-MIP", f"Center slice (z={axes_m['z'][shape[2] // 2] * 1e3:.1f} mm)"),
    ):
        shown = panel.imshow(image.T, cmap="gray", aspect="equal", origin="lower", extent=extent)
        panel.set_title(title)
        panel.set_xlabel("x (mm)")
        panel.set_ylabel("y (mm)")
        figure.colorbar(shown, ax=panel)
    figure.tight_layout()
    figure.savefig(destination / "reconstruction.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    return {
        "shape_xyz": list(shape),
        "reconstruction_time_s": reconstruction_time,
        "files": ["reconstruction.mat", "reconstruction.png", *original_figures],
    }
