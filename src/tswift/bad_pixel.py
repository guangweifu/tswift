"""Per-pixel MAD-robust bad-pixel masking.

Why MAD and not an absolute ADU threshold
------------------------------------------
Absolute ADU thresholds are chromatically biased. At the stellar SED peak, a 100-ADU
threshold may be 3σ; in the wings it's 15σ. So the flagging rate varies with wavelength
for no good reason, and replacing flagged pixels with the time-median silently erases
transit signal at bright channels (because the time-median is dominated by out-of-transit
frames). Per-pixel MAD clipping fixes both problems.

The `min_sigma` floor prevents flagging on artificially quiet pixels (e.g. reference
pixels or masked regions).

Diagnostic plot
---------------
`plot_bad_pixel(...)` writes a 4-panel review figure: median frame, per-pixel flag
rate heatmap (look for wavelength-dependent rates — chromatic bias warning), per-pixel
σ vs signal hexbin (confirms the threshold is nominal for the actual noise law), and
one corrected frame. Call it right after `mad_clip()` with the same inputs plus an
outdir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# 1.4826 · MAD is the standard robust estimator of σ for Gaussian noise.
_MAD_TO_SIGMA = 1.4826


def mad_clip(
    data_all: np.ndarray,
    *,
    n_sigma: float = 5.0,
    min_sigma: float = 2.0,
    replace_with_nan: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify and replace bad pixels via per-pixel time-MAD robust clipping.

    Parameters
    ----------
    data_all : np.ndarray, shape (n_frames, n_rows, n_cols)
        Time series of 2D frames (ramp-fit output).
    n_sigma : float, default 5.0
        Threshold in σ units.
    min_sigma : float, default 2.0
        ADU floor to avoid flagging pixels whose MAD is artificially small
        (reference pixels, zero regions, etc.).
    replace_with_nan : bool, default True
        If True, flagged values → NaN. Downstream nansum/nanmedian handles this
        correctly. Set False to replace with the per-pixel time-median (legacy;
        biases transit depth at bright channels — avoid).

    Returns
    -------
    data_fixed : np.ndarray
        Copy of input with flagged pixels replaced.
    bad_mask : np.ndarray, bool, same shape
        True where a sample was flagged.
    sigma_pix : np.ndarray, shape (n_rows, n_cols)
        The per-pixel σ estimate used for thresholding.
    """
    median = np.nanmedian(data_all, axis=0)
    diff = data_all - median[None, :, :]

    # MAD across time. As long as in-transit samples are a minority (< 40% for
    # any JWST transit observation), the MAD is unaffected by transit depth.
    mad = np.nanmedian(np.abs(diff), axis=0)
    sigma_pix = np.maximum(_MAD_TO_SIGMA * mad, min_sigma)
    bad_mask = np.abs(diff) > n_sigma * sigma_pix[None, :, :]

    data_fixed = np.array(data_all, copy=True)
    if replace_with_nan:
        data_fixed[bad_mask] = np.nan
    else:
        data_fixed[bad_mask] = np.broadcast_to(median, data_all.shape)[bad_mask]

    return data_fixed, bad_mask, sigma_pix


def plot_bad_pixel(
    data_all: np.ndarray,
    data_fixed: np.ndarray,
    bad_mask: np.ndarray,
    sigma_pix: np.ndarray,
    outdir: str | Path,
    *,
    n_sigma: Optional[float] = None,
    detector: Optional[str] = None,
) -> Path:
    """Write a 4-panel diagnostic for `mad_clip` outputs; return the png path.

    Reading the plot
    ----------------
    - Top-left: median frame. Sanity: should look like a focused spectrum.
    - Top-right: per-pixel flag rate. Look for wavelength-dependent banding —
      that indicates chromatic bias (threshold is either too aggressive at the SED
      peak or too loose in the wings).
    - Bottom-left: per-pixel robust σ vs signal (hexbin, log-log). Noise should
      scale ~ √signal (Poisson). A flat floor at the bottom is the `min_sigma`
      kicking in for quiet pixels.
    - Bottom-right: one corrected frame chosen to contain at least one flag,
      shown on the same scale as the median.
    """
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    median = np.nanmedian(data_all, axis=0)
    per_pix_flag = bad_mask.mean(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"bad_pixel diagnostic — {detector or ''}"
        f"{' — n_sigma=' + str(n_sigma) if n_sigma is not None else ''}"
        f"  total flagged: {bad_mask.mean() * 100:.3f}%"
    )

    vmin = float(np.nanpercentile(median, 5))
    vmax = float(np.nanpercentile(median, 98))

    ax = axes[0, 0]
    im = ax.imshow(median, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title("Median frame")
    ax.set_xlabel("column"); ax.set_ylabel("row")
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[0, 1]
    im = ax.imshow(per_pix_flag * 100, aspect="auto", origin="lower", vmin=0, vmax=5, cmap="magma")
    ax.set_title("Per-pixel flag rate (%)  ←  chromatic banding = biased threshold")
    ax.set_xlabel("column"); ax.set_ylabel("row")
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[1, 0]
    sig_flat = median.ravel()
    std_flat = sigma_pix.ravel()
    m = np.isfinite(sig_flat) & np.isfinite(std_flat) & (sig_flat > 0) & (std_flat > 0)
    if m.sum() > 10:
        ax.hexbin(sig_flat[m], std_flat[m], bins="log", gridsize=60, xscale="log", yscale="log")
    ax.set_xlabel("pixel signal (median, ADU/s)")
    ax.set_ylabel("per-pixel σ (1.4826·MAD, floored)")
    ax.set_title("Noise vs signal  ←  should trend with √signal")

    ax = axes[1, 1]
    flagged_frames = np.where(np.any(bad_mask, axis=(1, 2)))[0]
    si = int(flagged_frames[0]) if flagged_frames.size else 0
    ax.imshow(data_fixed[si], aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"Corrected frame {si}")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    plt.tight_layout()
    png = outdir / "bad_pixel.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png
