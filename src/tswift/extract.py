"""Trace finding, 2D aperture extraction, and aperture optimization via OOT scatter.

The whole extraction pipeline in one module. Steps, in order:

1. **Find trace.** Per-column argmax + parabolic sub-pixel refinement, then a robust
   polynomial fit. SOSS needs poly_order ≥ 5 because order-1 curves by ~50 rows across
   the detector; NIRSpec is ~3 rows and needs only poly_order 3.

2. **Extract wide aperture.** A generous ±half_width window around the polynomial trace.
   Wider than the final aperture — the next step brute-forces the true extraction window.

3. **Optimize aperture.** Brute-force every contiguous `(up, down)` sub-aperture within
   the wide window; pick the one that minimizes out-of-transit (OOT) scatter on either
   the white-light curve or the median per-channel light curve. OOT baseline is detected
   from the flux drop at ingress.

4. **Collapse + clean.** Sum the best aperture to produce a 2D (time, wavelength) cube
   and apply per-channel rolling-median outlier rejection.

Key design decisions worth preserving
-------------------------------------
- **Trace finder is argmax + parabolic, not Gaussian.** A Gaussian fit is dominated by
  the PSF wings and drifts toward the mean. Argmax-of-smoothed + parabolic sub-pixel is
  robust and reproducible.
- **Aperture optimizer is brute-force over OOT scatter.** Any analytic "optimal aperture"
  assumes a PSF model the real data doesn't match. Brute-force is O(n²) on ~40 pixels;
  cheap enough.
- **Per-channel criterion (not wl_rms) for SOSS.** SOSS order-2 contamination contributes
  signal but hurts per-channel scatter at 1.0–1.4 μm. `wl_rms` would happily include it.

The module is functional (no classes): composable functions + a top-level
`run_extract(...)` orchestrator.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def fill_nan_with_nanmedian(array: np.ndarray, kernel_size: int = 10) -> np.ndarray:
    """Fill NaN values with local nanmedian (used before trace finding)."""
    result = np.copy(array)
    nan_mask = np.isnan(result)
    if not np.any(nan_mask):
        return result
    half = kernel_size // 2
    for i, j in zip(*np.where(nan_mask)):
        r_lo, r_hi = max(0, i - half), min(array.shape[0], i + half)
        c_lo, c_hi = max(0, j - half), min(array.shape[1], j + half)
        med = np.nanmedian(array[r_lo:r_hi, c_lo:c_hi])
        if np.isfinite(med):
            result[i, j] = med
    return result


def remove_outliers_1d(arr: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """Rolling-median / MAD outlier rejection on a 1-D light curve.

    Preserves length; replaces flagged samples with the local rolling median.
    """
    series = pd.Series(arr)
    rolling_median = series.rolling(window, center=True, min_periods=1).median()
    deviation = (series - rolling_median).abs()
    mad = deviation.median()
    if mad == 0:
        return arr
    outliers = deviation > (threshold * mad)
    series[outliers] = rolling_median[outliers]
    return series.values


# ----------------------------------------------------------------------------
# OOT baseline detection
# ----------------------------------------------------------------------------

def find_ingress_index(wl_nor: np.ndarray, n_seed: int | None = None) -> int:
    """First frame where the white-light flux drops > 5 MAD below the pre-ingress baseline."""
    n = len(wl_nor)
    n_seed = n_seed or max(10, int(n * 0.10))
    seed = wl_nor[:n_seed]
    med = np.nanmedian(seed)
    mad = np.nanmedian(np.abs(seed - med))
    if mad == 0:
        mad = np.nanstd(seed) or 1e-9
    threshold = med - 5 * mad
    rolling = pd.Series(wl_nor).rolling(5, center=True, min_periods=1).median()
    for i in range(n_seed, n):
        if rolling.iloc[i] < threshold:
            return i
    return n


def build_pretransit_oot_mask(wl_nor: np.ndarray, sigma: float = 5.0) -> tuple[np.ndarray, int]:
    """Pre-ingress-only baseline, with outlier rejection against the seed median.

    Returns the boolean OOT mask (same length as wl_nor) and the ingress frame index.
    """
    ingress_idx = find_ingress_index(wl_nor)
    pre = wl_nor[:ingress_idx]
    med = np.nanmedian(pre)
    mad = np.nanmedian(np.abs(pre - med))
    if mad == 0:
        mad = np.nanstd(pre) or 1e-9
    good = np.abs(pre - med) < sigma * 1.4826 * mad
    oot_mask = np.zeros(len(wl_nor), dtype=bool)
    oot_mask[:ingress_idx] = good
    return oot_mask, ingress_idx


# ----------------------------------------------------------------------------
# Trace finding
# ----------------------------------------------------------------------------

def find_trace(
    median_frame: np.ndarray,
    *,
    mode: str,
    detector: str | None = None,
    poly_order: int = 5,
    outlier_clip: float = 4.0,
) -> np.ndarray:
    """Per-column trace row (argmax + parabolic sub-pixel + robust polynomial).

    Parameters
    ----------
    median_frame : (n_rows, n_cols)
        Time-median of the 2D cube after bad-pixel fixing.
    mode : {"SOSS", "G395H", "PRISM"}
    detector : "nis" | "nrs1" | "nrs2" | None
    poly_order : int
        Polynomial order for the final smoothing fit. SOSS needs ≥5; NIRSpec 3 is fine.
    outlier_clip : float
        Flag per-column centroids further than this many rows from a rough polynomial
        before the final fit.

    Returns
    -------
    np.ndarray, shape (n_cols,)
        Polynomial trace center for every column (float).
    """
    median = fill_nan_with_nanmedian(median_frame, kernel_size=10)
    n_rows, n_cols = median.shape

    # Light vertical smoothing stabilizes argmax against single-pixel noise peaks.
    smoothed = uniform_filter1d(
        np.where(np.isfinite(median), median, 0.0), size=3, axis=0, mode="nearest"
    )

    # Pass 1: coarse argmax per column.
    trace_argmax = np.full(n_cols, np.nan)
    for i in range(n_cols):
        col = smoothed[:, i]
        if np.all(col <= 0) or not np.any(np.isfinite(median[:, i])):
            continue
        trace_argmax[i] = float(np.argmax(col))

    # Pass 2: parabolic sub-pixel refinement around each argmax.
    trace_y = trace_argmax.copy()
    for i in range(n_cols):
        k = trace_y[i]
        if not np.isfinite(k):
            continue
        k = int(k)
        if k < 1 or k >= n_rows - 1:
            continue
        y0, y1, y2 = smoothed[k - 1, i], smoothed[k, i], smoothed[k + 1, i]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            if -1 <= delta <= 1:
                trace_y[i] = k + delta

    # Pass 3: robust outlier rejection.
    trace_clean = trace_y.copy()
    x = np.arange(n_cols)
    if poly_order >= 5:
        # For wide curved traces (SOSS) use a provisional fit to clip deviants.
        valid0 = np.isfinite(trace_clean)
        if valid0.sum() > poly_order + 1:
            c0 = np.polyfit(x[valid0], trace_clean[valid0], min(poly_order, 5))
            smooth = np.polyval(c0, x)
            trace_clean[np.abs(trace_clean - smooth) > outlier_clip] = np.nan
    else:
        # For near-flat traces (NIRSpec) clip against the median.
        med_trace = np.nanmedian(trace_clean)
        trace_clean[np.abs(trace_clean - med_trace) > outlier_clip] = np.nan

    valid = np.isfinite(trace_clean)
    if valid.sum() < poly_order + 1:
        logger.warning("Not enough valid trace centroids; using median row.")
        return np.full(n_cols, np.nanmedian(trace_clean))

    coeffs = np.polyfit(x[valid], trace_clean[valid], poly_order)
    trace_fit = np.polyval(coeffs, x)
    return trace_fit


# ----------------------------------------------------------------------------
# 2D aperture extraction
# ----------------------------------------------------------------------------

def extract_trace_2d(
    data_all: np.ndarray, trace_fit: np.ndarray, half_width: int
) -> np.ndarray:
    """Extract a 2D cube of shape (n_frames, 2*half_width, n_cols) around `trace_fit`.

    Rows outside the detector are filled with NaN.
    """
    n_frames, n_rows, n_cols = data_all.shape
    ap_height = 2 * half_width
    extract_all = np.full((n_frames, ap_height, n_cols), np.nan)

    for j in range(n_cols):
        lo = int(np.round(trace_fit[j] - half_width))
        hi = lo + ap_height
        src_lo, src_hi = max(0, lo), min(n_rows, hi)
        dst_lo = src_lo - lo
        dst_hi = dst_lo + (src_hi - src_lo)
        extract_all[:, dst_lo:dst_hi, j] = data_all[:, src_lo:src_hi, j]
    return extract_all


# ----------------------------------------------------------------------------
# Aperture optimization
# ----------------------------------------------------------------------------

ApertureCriterion = Literal["wl_rms", "per_channel"]


def _measure_wl_rms(cl_2D: np.ndarray, oot_mask: np.ndarray,
                    wl_left: int, wl_right: int) -> float:
    wl = np.nansum(cl_2D[:, wl_left:wl_right], axis=1)
    baseline = np.nanmedian(wl[oot_mask])
    if not np.isfinite(baseline) or baseline == 0:
        return np.inf
    return float(np.nanstd(wl[oot_mask] / baseline))


def _measure_per_channel(cl_2D: np.ndarray, oot_mask: np.ndarray,
                         wl_left: int, wl_right: int) -> float:
    sub = cl_2D[:, wl_left:wl_right]
    oot_median = np.nanmedian(sub[oot_mask, :], axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        sub_nor = sub / oot_median
    return float(np.nanmedian(np.nanstd(sub_nor[oot_mask, :], axis=0)))


def optimize_aperture(
    extract_2D: np.ndarray,
    *,
    criterion: ApertureCriterion = "per_channel",
    wavelength_left: int = 0,
    wavelength_right: int | None = None,
    min_width: int = 3,
) -> tuple[int, int, list[tuple[int, int, float]], np.ndarray, int]:
    """Brute-force every contiguous `(up, down)` sub-aperture; pick the one that
    minimizes out-of-transit scatter.

    Returns
    -------
    best_up, best_down : int
        Optimal extraction rows [best_up, best_down).
    all_results : list of (up, down, scatter)
        Every tried sub-aperture for diagnostic plots.
    oot_mask : np.ndarray of bool
        The out-of-transit frame mask used.
    ingress_idx : int
        Detected ingress frame index.
    """
    n_frames, n_spatial, n_cols = extract_2D.shape
    wl_left = max(0, wavelength_left)
    wl_right = min(wavelength_right or n_cols, n_cols)
    if wl_left >= wl_right:
        raise ValueError(f"bad wavelength window: [{wl_left}, {wl_right})")

    # Detect OOT using the widest possible aperture.
    wl_full = np.nansum(extract_2D[:, :, wl_left:wl_right], axis=(1, 2))
    wl_nor_full = wl_full / np.nanmedian(wl_full)
    oot_mask, ingress_idx = build_pretransit_oot_mask(wl_nor_full)
    logger.info(
        f"Ingress at frame {ingress_idx}/{len(wl_nor_full)}, "
        f"OOT baseline: {int(oot_mask.sum())} frames"
    )

    if criterion == "wl_rms":
        metric = _measure_wl_rms
    elif criterion == "per_channel":
        metric = _measure_per_channel
    else:
        raise ValueError(f"unknown aperture criterion {criterion!r}")

    results: list[tuple[int, int, float]] = []
    for up in range(n_spatial):
        for down in range(up + min_width, n_spatial + 1):
            cl_2D = np.nansum(extract_2D[:, up:down, :], axis=1)
            s = metric(cl_2D, oot_mask, wl_left, wl_right)
            results.append((up, down, s))

    best_up, best_down, _ = min(results, key=lambda x: x[2])
    logger.info(f"Optimal aperture rows {best_up}:{best_down} (width {best_down - best_up})")
    return best_up, best_down, results, oot_mask, ingress_idx


# ----------------------------------------------------------------------------
# Per-channel cleaning
# ----------------------------------------------------------------------------

def clean_per_channel(cl_2D: np.ndarray, *, window: int = 20, threshold: float = 4.0) -> np.ndarray:
    """Rolling-median outlier rejection on each wavelength's light curve.

    Returns a new array of the same shape with outliers replaced by rolling median.
    """
    clean = np.empty_like(cl_2D)
    for i in range(cl_2D.shape[1]):
        clean[:, i] = remove_outliers_1d(np.copy(cl_2D[:, i]), window=window, threshold=threshold)
    return clean


# ----------------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------------

def run_extract(
    data_all: np.ndarray,
    *,
    mode: str,
    detector: str | None = None,
    trace_half_width: int = 22,
    trace_poly_order: int = 5,
    trace_outlier_clip: float = 4.0,
    aperture_criterion: ApertureCriterion = "per_channel",
    wavelength_left: int = 0,
    wavelength_right: int | None = None,
    outlier_window: int = 20,
    outlier_threshold: float = 4.0,
) -> dict:
    """Run the full extraction pipeline from a bad-pixel-fixed 3D cube.

    Parameters
    ----------
    data_all : (n_frames, n_rows, n_cols)
        Input cube. Usually the output of `bad_pixel.mad_clip`.
    mode : {"SOSS", "G395H", "PRISM"}
    detector : "nis" | "nrs1" | "nrs2" | None

    Returns
    -------
    dict with keys:
        trace_fit, extract_2D, clean_2D, aperture=(up, down), oot_mask, ingress_idx,
        all_aperture_results
    """
    if mode not in ("SOSS", "G395H", "PRISM"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "PRISM":
        raise NotImplementedError(
            "PRISM mode uses row slicing, not polynomial trace finding. "
            "Will be ported separately."
        )

    median = np.nanmedian(data_all, axis=0)
    logger.info(f"Finding trace (mode={mode}, detector={detector}, poly_order={trace_poly_order})")
    trace_fit = find_trace(
        median,
        mode=mode,
        detector=detector,
        poly_order=trace_poly_order,
        outlier_clip=trace_outlier_clip,
    )
    logger.info(
        f"Trace median row {np.nanmedian(trace_fit):.1f}, "
        f"range {np.nanmin(trace_fit):.1f}–{np.nanmax(trace_fit):.1f}"
    )

    logger.info(f"Extracting wide aperture (±{trace_half_width} rows)")
    extract_2D = extract_trace_2d(data_all, trace_fit, trace_half_width)

    logger.info(f"Optimizing aperture (criterion={aperture_criterion})")
    best_up, best_down, opt_results, oot_mask, ingress_idx = optimize_aperture(
        extract_2D,
        criterion=aperture_criterion,
        wavelength_left=wavelength_left,
        wavelength_right=wavelength_right,
    )

    cl_2D = np.nansum(extract_2D[:, best_up:best_down, :], axis=1)

    logger.info(f"Cleaning per-channel (window={outlier_window}, threshold={outlier_threshold})")
    clean_2D = clean_per_channel(cl_2D, window=outlier_window, threshold=outlier_threshold)

    return {
        "trace_fit": trace_fit,
        "extract_2D": extract_2D,
        "clean_2D": clean_2D,
        "aperture": (int(best_up), int(best_down)),
        "oot_mask": oot_mask,
        "ingress_idx": int(ingress_idx),
        "all_aperture_results": opt_results,
    }


def save_extract_outputs(result: dict, out_dir: Path) -> None:
    """Write the standard output files to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "trace_fit.npy", result["trace_fit"])
    np.save(out_dir / "extract_2D.npy", result["extract_2D"])
    np.save(out_dir / "clean_2D.npy", result["clean_2D"])
    np.save(out_dir / "oot_mask.npy", result["oot_mask"])


# ----------------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------------

def plot_trace(
    data_all: np.ndarray,
    trace_fit: np.ndarray,
    half_width: int,
    outdir: str | Path,
    *,
    detector: str | None = None,
) -> Path:
    """Trace overlay + aperture dashes + spatial profile.

    Read the plot
    -------------
    - Left: median frame with red trace line and dashed ±half_width aperture.
      Watch for trace sitting high or low relative to the PSF — that means
      trace_half_width is too narrow or trace_poly_order is off.
    - Right: collapsed spatial profile. The PSF should be centered within the
      aperture and decay to < 10 % of peak at the edges. If the profile is
      asymmetric or clipped at an edge, widen `trace_half_width`.
    """
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    median = np.nanmedian(data_all, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(f"extract/trace — {detector or ''}   "
                 f"row range {np.nanmin(trace_fit):.1f}–{np.nanmax(trace_fit):.1f},   "
                 f"half_width={half_width}")

    vmin = float(np.nanpercentile(median, 5))
    vmax = float(np.nanpercentile(median, 98))
    x = np.arange(len(trace_fit))
    ax = axes[0]
    ax.imshow(median, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.plot(x, trace_fit, "r-", lw=1, label="trace center")
    ax.plot(x, trace_fit - half_width, "r--", lw=0.5)
    ax.plot(x, trace_fit + half_width, "r--", lw=0.5)
    ax.set_xlabel("column"); ax.set_ylabel("row"); ax.legend(loc="upper right")

    # Collapsed spatial profile around the trace
    ap = 2 * half_width
    n_rows = median.shape[0]
    profile = np.full(ap, np.nan)
    for j in range(median.shape[1]):
        lo = int(np.round(trace_fit[j] - half_width))
        hi = lo + ap
        src_lo, src_hi = max(0, lo), min(n_rows, hi)
        dst_lo = src_lo - lo
        col = median[src_lo:src_hi, j]
        if col.size:
            np.nansum([profile[dst_lo:dst_lo + col.size], col], axis=0, out=profile[dst_lo:dst_lo + col.size])
    profile = profile / np.nanmax(profile)
    ax = axes[1]
    ax.plot(profile, np.arange(ap), "b-")
    ax.set_xlabel("norm flux")
    ax.set_ylabel("row within ±half_width")
    ax.axhline(half_width, color="r", ls="--", lw=0.5)
    ax.set_title("Spatial profile")

    plt.tight_layout()
    png = outdir / "trace.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png


def plot_aperture_scan(
    extract_2D: np.ndarray,
    all_results: list,
    best_up: int,
    best_down: int,
    oot_mask: np.ndarray,
    ingress_idx: int,
    time_hr: np.ndarray,
    wavelength_left: int,
    wavelength_right: int,
    outdir: str | Path,
    *,
    detector: str | None = None,
) -> Path:
    """Aperture scan + spatial profile + white-light curve with OOT baseline.

    Read the plot
    -------------
    - Left: OOT scatter vs aperture width (every tried (up, down) pair). The
      selected aperture is the minimum. A broad flat bowl is healthy; a sharp
      V-shape at the edge means the optimizer couldn't find a plateau — likely
      an extraction contamination problem.
    - Middle: collapsed spatial PSF with the selected aperture shaded. The
      aperture should cover the bright core and trail into the wings symmetrically.
    - Right: white-light curve (normalized) using the best aperture, with the
      detected OOT baseline in red and the ingress frame marked. If baseline
      looks wrong, `build_pretransit_oot_mask` heuristic misfired — check time
      ordering and segment stitching.
    """
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    widths = np.array([r[1] - r[0] for r in all_results])
    scats = np.array([r[2] for r in all_results])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"extract/aperture — {detector or ''}   "
                 f"best rows [{best_up}:{best_down}], width {best_down - best_up}")

    axes[0].scatter(widths, scats, s=5, alpha=0.4)
    axes[0].axvline(best_down - best_up, color="r", ls="--", label=f"best width={best_down - best_up}")
    axes[0].set_xlabel("aperture width (rows)")
    axes[0].set_ylabel("OOT scatter (criterion metric)")
    axes[0].set_title("Aperture scan")
    axes[0].legend()

    median_trace = np.nanmedian(extract_2D, axis=0)
    spatial = np.nansum(median_trace, axis=1)
    spatial = spatial / np.nanmax(spatial)
    axes[1].plot(spatial, "b-")
    axes[1].axvspan(best_up, best_down, color="red", alpha=0.2, label=f"aperture [{best_up}:{best_down}]")
    axes[1].set_xlabel("row within extracted cube")
    axes[1].set_ylabel("norm flux")
    axes[1].set_title("Spatial profile + selected aperture")
    axes[1].legend()

    cl = np.nansum(extract_2D[:, best_up:best_down, wavelength_left:wavelength_right], axis=(1, 2))
    baseline = np.nanmedian(cl[oot_mask]) if oot_mask.any() else np.nanmedian(cl)
    wl_nor = cl / baseline
    axes[2].scatter(time_hr[~oot_mask], wl_nor[~oot_mask], s=2, color="0.6", alpha=0.6, label="transit/post")
    axes[2].scatter(time_hr[oot_mask], wl_nor[oot_mask], s=4, color="red", alpha=0.9, label=f"OOT baseline ({int(oot_mask.sum())} frames)")
    if 0 < ingress_idx < len(time_hr):
        axes[2].axvline(time_hr[ingress_idx], color="blue", ls="--", lw=1, label=f"ingress (frame {ingress_idx})")
    axes[2].set_xlabel("time (hours)")
    axes[2].set_ylabel("norm flux")
    axes[2].set_title("White-light curve (best aperture)")
    axes[2].legend(markerscale=3, fontsize=9)

    plt.tight_layout()
    png = outdir / "aperture.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png


def plot_clean(
    cl_2D_before: np.ndarray,
    cl_2D_after: np.ndarray,
    time_hr: np.ndarray,
    wavelength_left: int,
    wavelength_right: int,
    outdir: str | Path,
    *,
    detector: str | None = None,
) -> Path:
    """White-light curve before and after per-channel rolling-median clean.

    Read the plot
    -------------
    - Both curves should overlap closely. If `after` looks significantly smoother
      than `before` (not just individual spikes gone), your outlier_threshold is
      too aggressive and you're smoothing real transit data. Raise `threshold`
      or widen `window` so the rolling median is more permissive.
    """
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    before = np.nansum(cl_2D_before[:, wavelength_left:wavelength_right], axis=1)
    after = np.nansum(cl_2D_after[:, wavelength_left:wavelength_right], axis=1)
    med = np.nanmedian(before)
    before = before / med
    after = after / med

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_hr, before, ".", ms=2, alpha=0.4, color="0.6", label="before clean")
    ax.plot(time_hr, after, ".", ms=2, alpha=0.7, color="tab:red", label="after clean")
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("normalized white-light flux")
    ax.set_title(f"extract/clean — {detector or ''}   if after << before = outlier_threshold too tight")
    ax.legend()
    plt.tight_layout()
    png = outdir / "clean.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png
