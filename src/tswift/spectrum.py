"""Final-spectrum I/O and plotting.

The combine step returns a dict of arrays (pure data). This module persists that dict
to human-readable files + a review figure. Always call `save_spectrum(...)` at the
end of a pipeline run — it's the handoff point between the pipeline and the scientist.

Outputs (writes into `outdir`):
    spectrum_native.txt     — wvl_um  depth_ppm  depth_err_ppm  rp  rp_err
    spectrum_<N>nm.txt      — wvl_um  depth_ppm  depth_err_ppm    (one per binning)
    spectrum.png            — native (faint) + all binnings overlaid, one color per bin
    spectrum_summary.json   — one-line machine-readable summary (native N, per-bin N,
                              mean depth, median error) — useful for agents.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_spectrum(
    combined: dict,
    *,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
    show_native: bool = True,
):
    """Plot native + binned spectra on one axes, return (fig, ax)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if show_native:
        native = combined["native"]
        mask = native["mask"]
        ax.errorbar(
            native["wvl_um"][mask],
            native["depth_ppm"][mask],
            yerr=native["depth_err_ppm"][mask],
            fmt=".",
            color="0.7",
            ecolor="0.85",
            ms=2,
            alpha=0.5,
            elinewidth=0.3,
            label="native",
            zorder=1,
        )

    # Use a perceptually-ordered palette that reads well at print sizes.
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange"]
    for i, (name, b) in enumerate(combined["binned"].items()):
        ax.errorbar(
            b["wvl_um"], b["depth_ppm"], yerr=b["depth_err_ppm"],
            fmt="o", color=colors[i % len(colors)], ms=4, capsize=2,
            elinewidth=1.0, label=name, zorder=2 + i,
        )

    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Transit depth (ppm)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)
    return fig, ax


def save_spectrum(
    combined: dict,
    outdir: str | Path,
    *,
    planet_name: str | None = None,
    dpi: int = 150,
) -> dict[str, Path]:
    """Persist the combined spectrum to disk (txt + png + json summary).

    Parameters
    ----------
    combined : dict
        Return value of `tswift.combine_spectrum` (or `combine_detectors`).
    outdir : path-like
        Destination directory. Created if missing.
    planet_name : str, optional
        Shown in the figure title.
    dpi : int

    Returns
    -------
    dict[str, Path]
        Map of written file keys → absolute paths. Keys: "native_txt",
        "binned_<N>nm_txt", "png", "summary_json".
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # --- Native resolution text ---
    native = combined["native"]
    mask = native["mask"]
    native_txt = outdir / "spectrum_native.txt"
    header = "wvl_um  depth_ppm  depth_err_ppm  rp  rp_err"
    np.savetxt(
        native_txt,
        np.column_stack([
            native["wvl_um"][mask],
            native["depth_ppm"][mask],
            native["depth_err_ppm"][mask],
            native["rp"][mask],
            native["rp_err"][mask],
        ]),
        header=header, fmt="%.6e",
    )
    paths["native_txt"] = native_txt.resolve()
    logger.info(f"wrote {native_txt.name}: {int(mask.sum())} points")

    # --- Binned text files ---
    summary = {
        "planet": planet_name,
        "native": {
            "n_points": int(mask.sum()),
            "mean_depth_ppm": float(np.mean(native["depth_ppm"][mask])) if mask.any() else None,
            "median_err_ppm": float(np.median(native["depth_err_ppm"][mask])) if mask.any() else None,
            "wvl_range_um": [
                float(np.min(native["wvl_um"][mask])) if mask.any() else None,
                float(np.max(native["wvl_um"][mask])) if mask.any() else None,
            ],
        },
        "binned": {},
    }
    for name, b in combined["binned"].items():
        path = outdir / f"spectrum_{name}.txt"
        np.savetxt(
            path,
            np.column_stack([b["wvl_um"], b["depth_ppm"], b["depth_err_ppm"]]),
            header="wvl_um  depth_ppm  depth_err_ppm",
            fmt="%.6e",
        )
        paths[f"binned_{name}_txt"] = path.resolve()
        summary["binned"][name] = {
            "n_points": len(b["wvl_um"]),
            "mean_depth_ppm": float(np.mean(b["depth_ppm"])) if len(b["depth_ppm"]) else None,
            "median_err_ppm": float(np.median(b["depth_err_ppm"])) if len(b["depth_err_ppm"]) else None,
            "wvl_range_um": [
                float(np.min(b["wvl_um"])) if len(b["wvl_um"]) else None,
                float(np.max(b["wvl_um"])) if len(b["wvl_um"]) else None,
            ],
        }
        logger.info(f"wrote {path.name}: {len(b['wvl_um'])} bins")

    # --- Figure ---
    title = f"{planet_name} transmission spectrum" if planet_name else "Transmission spectrum"
    fig, _ = plot_spectrum(combined, title=title)
    png = outdir / "spectrum.png"
    fig.tight_layout()
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    paths["png"] = png.resolve()
    logger.info(f"wrote {png.name}")

    # --- JSON summary (for agents) ---
    jpath = outdir / "spectrum_summary.json"
    jpath.write_text(json.dumps(summary, indent=2))
    paths["summary_json"] = jpath.resolve()

    return paths
