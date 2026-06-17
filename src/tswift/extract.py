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
    centering: str | None = None,
    centroid_half: int = 10,
    fit_window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Per-column trace row + robust polynomial smoothing.

    Two centering algorithms — pick by the PSF shape:

    - `"argmax_parab"` — argmax of a vertically smoothed column + parabolic
      sub-pixel refinement.  Right for narrow, single-peaked PSFs (NIRSpec
      G395H, PRISM).  ~5× cheaper than centroid mode.

    - `"centroid"` — argmax seed + flux-weighted centroid in a ±centroid_half-
      row window around the seed.  Right for wide or multi-peaked PSFs
      where argmax snaps to one of several lobes (NIRISS SOSS GR700XD has
      a double-peaked spatial profile, MIRI LRS has detector-defect
      sub-peaks).  Argmax can sit 5+ rows off the PSF center of mass on
      SOSS — the centroid stays anchored to the bright core regardless.

    `centering=None` picks `"centroid"` for SOSS / MIRI_LRS and
    `"argmax_parab"` everywhere else.

    Parameters
    ----------
    median_frame : (n_rows, n_cols)
        Time-median of the 2D cube after bad-pixel fixing.
    mode : {"SOSS", "G395H", "PRISM", "MIRI_LRS"}
    detector : "nis" | "nrs1" | "nrs2" | "mirimage" | None
        Currently unused by the trace logic — reserved for future per-detector
        handling (e.g. MIRI detector-defect masking). Kept so callers can pass
        it without a signature change later.
    poly_order : int
        Polynomial order for the final smoothing fit. SOSS needs ≥5;
        NIRSpec 3 is fine.
    outlier_clip : float
        Flag per-column centroids further than this many rows from a
        rough polynomial before the final fit.
    centering : {"argmax_parab", "centroid", None}
        Per-column centering algorithm.  None → mode-aware default.
    centroid_half : int
        Radius (in rows) of the centroid window around the argmax seed.
        Empirically `10` worked best on WASP-94 A b SOSS — wider pulls in
        background, narrower under-resolves the double peak.  Only used
        when `centering="centroid"`.
    fit_window : (int, int) or None
        Half-open column range ``[lo, hi)`` of *illuminated* columns the
        polynomial is allowed to use.  Columns outside it (e.g. the dark
        left ~650 columns of G395H NRS1, or the 8 reference columns at the
        right edge) carry no real trace — their per-column centroids are
        argmax-of-noise and, because they can span a large contiguous block,
        the Pass-3 outlier clip cannot reject them, so they drag the global
        low-order fit.  When given, only in-window columns constrain the
        provisional + final fits; outside the window the returned trace is
        clamped to the nearest in-window edge value (no wild extrapolation).
        ``None`` (default) uses every finite column — prior behaviour.

    Returns
    -------
    np.ndarray, shape (n_cols,)
        Polynomial trace center for every column (float).
    """
    if centering is None:
        centering = "centroid" if mode in ("SOSS", "MIRI_LRS") else "argmax_parab"
    if centering not in ("argmax_parab", "centroid"):
        raise ValueError(f"unknown centering={centering!r}")

    median = fill_nan_with_nanmedian(median_frame, kernel_size=10)
    n_rows, n_cols = median.shape

    # Light vertical smoothing stabilizes argmax against single-pixel noise peaks.
    smoothed = uniform_filter1d(
        np.where(np.isfinite(median), median, 0.0), size=3, axis=0, mode="nearest"
    )

    # Pass 1: coarse argmax per column (used as seed for both centering modes).
    trace_argmax = np.full(n_cols, np.nan)
    for i in range(n_cols):
        col = smoothed[:, i]
        if np.all(col <= 0) or not np.any(np.isfinite(median[:, i])):
            continue
        trace_argmax[i] = float(np.argmax(col))

    trace_y = trace_argmax.copy()
    if centering == "argmax_parab":
        # Pass 2a: parabolic sub-pixel refinement around each argmax.
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
    else:
        # Pass 2b: flux-weighted centroid in a ±centroid_half-row window
        # around the argmax seed.  Pulls toward the PSF center of mass even
        # when the brightest pixel is on one side of an asymmetric profile.
        for i in range(n_cols):
            k = trace_argmax[i]
            if not np.isfinite(k):
                continue
            ki = int(k)
            lo = max(0, ki - centroid_half)
            hi = min(n_rows, ki + centroid_half + 1)
            col = np.maximum(median[lo:hi, i], 0)   # clip negatives → no neg-mass
            tot = np.nansum(col)
            if tot > 0 and np.isfinite(tot):
                rows = np.arange(lo, hi, dtype=float)
                trace_y[i] = float(np.nansum(rows * col) / tot)
            else:
                trace_y[i] = np.nan

    # Pass 3: robust outlier rejection — clip against a provisional polynomial
    # fit at the SAME order as the final fit.  Clipping against the global
    # median fails for curved traces (e.g. G395H NRS2 spans ~18 rows; cols
    # near the ends deviate by >10 from the global median, get rejected as
    # "outliers", and the surviving cols then extrapolate a wrong shape).
    trace_clean = trace_y.copy()
    x = np.arange(n_cols)

    # Restrict the fit to the illuminated columns.  Dark columns produce
    # argmax-of-noise centroids; when they form a large contiguous block
    # (e.g. cols 0-650 on G395H NRS1) the Pass-3 clip below can't reject
    # them and they bias the global low-order polynomial.
    if fit_window is not None:
        lo_w, hi_w = int(fit_window[0]), int(fit_window[1])
        lo_w = max(0, lo_w)
        hi_w = n_cols if hi_w is None else min(n_cols, hi_w)
        trace_clean[:lo_w] = np.nan
        trace_clean[hi_w:] = np.nan
    else:
        lo_w, hi_w = 0, n_cols

    valid0 = np.isfinite(trace_clean)
    if valid0.sum() > poly_order + 1:
        # `min(poly_order, 5)` keeps the provisional fit numerically stable
        # for very high orders (matching the previous SOSS code path).
        prov_order = min(poly_order, 5)
        c0 = np.polyfit(x[valid0], trace_clean[valid0], prov_order)
        smooth = np.polyval(c0, x)
        trace_clean[np.abs(trace_clean - smooth) > outlier_clip] = np.nan

    valid = np.isfinite(trace_clean)
    if valid.sum() < poly_order + 1:
        logger.warning("Not enough valid trace centroids; using median row.")
        finite_rows = trace_clean[valid]
        if finite_rows.size:
            med = float(np.median(finite_rows))
        else:
            # Blank/faint detector: no valid centroids anywhere. Fall back to the
            # geometric center row so downstream mask builders get a finite,
            # sane constant trace instead of NaN (which crashes int() casts).
            med = (n_rows - 1) / 2.0
        return np.full(n_cols, med)

    coeffs = np.polyfit(x[valid], trace_clean[valid], poly_order)
    trace_fit = np.polyval(coeffs, x)

    # Outside the illuminated window the polynomial is unconstrained and can
    # extrapolate wildly; clamp to the nearest in-window edge value so the
    # (science-excluded) dark columns get a sane constant row instead.
    if fit_window is not None:
        trace_fit[:lo_w] = trace_fit[lo_w]
        trace_fit[hi_w:] = trace_fit[hi_w - 1]
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
    restrict_trace_fit_to_window: bool = False,
    outlier_window: int = 20,
    outlier_threshold: float = 4.0,
) -> dict:
    """Run the full extraction pipeline from a bad-pixel-fixed 3D cube.

    Parameters
    ----------
    data_all : (n_frames, n_rows, n_cols)
        Input cube. Usually the output of `bad_pixel.mad_clip`.
    mode : {"SOSS", "G395H"}
        SOSS and G395H are supported here.  PRISM (row-slicing) and MIRI LRS
        (row-dispersed, ``bg_axis='row'``) use a different extraction geometry
        and are not yet ported through ``run_extract`` — they raise
        ``NotImplementedError``.  ``run_calibrate`` does support MIRI LRS.
    detector : "nis" | "nrs1" | "nrs2" | None
    restrict_trace_fit_to_window : bool
        If True, the trace polynomial is fit only over ``[wavelength_left,
        wavelength_right)`` (dark columns outside are excluded and the trace is
        clamped to the edge value).  Use for detectors with a large dark block
        that biases the global fit (G395H NRS1 cols 0-650).  Default False
        reproduces the historical trace exactly — this knob CHANGES the trace,
        so existing reductions are unaffected unless they opt in.

    Returns
    -------
    dict with keys:
        trace_fit, extract_2D, clean_2D, aperture=(up, down), oot_mask, ingress_idx,
        all_aperture_results
    """
    if mode == "PRISM":
        raise NotImplementedError(
            "PRISM mode uses row slicing, not polynomial trace finding. "
            "Will be ported separately."
        )
    if mode in ("MIRI", "MIRILRS", "MIRI_LRS"):
        raise NotImplementedError(
            "MIRI LRS extraction is not yet ported through run_extract: it is "
            "row-dispersed (wavelength along rows, bg_axis='row'), so the "
            "column-oriented trace + aperture logic here does not apply. "
            "run_calibrate() supports MIRI LRS; extract it separately for now."
        )
    if mode not in ("SOSS", "G395H"):
        raise ValueError(f"unknown mode {mode!r}")

    median = np.nanmedian(data_all, axis=0)
    n_cols = median.shape[1]
    wl_right_eff = n_cols if wavelength_right is None else min(n_cols, wavelength_right)
    # Opt-in only: restrict the trace polyfit to the illuminated window.  This
    # is a genuine improvement where a large contiguous dark block (e.g. G395H
    # NRS1 cols 0-650) biases the global low-order fit, but it CHANGES the trace
    # vs prior reductions, so it must be requested explicitly — `wavelength_left/
    # right` historically governed only the LC window, not the trace polyfit.
    fit_window = (max(0, wavelength_left), wl_right_eff) if restrict_trace_fit_to_window else None
    logger.info(
        f"Finding trace (mode={mode}, detector={detector}, poly_order={trace_poly_order}"
        + (f", fit restricted to cols [{fit_window[0]}, {fit_window[1]}))" if fit_window else ")")
    )
    trace_fit = find_trace(
        median,
        mode=mode,
        detector=detector,
        poly_order=trace_poly_order,
        outlier_clip=trace_outlier_clip,
        fit_window=fit_window,
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


def measure_trace_position(
    extract_2D: np.ndarray,
    clean_2D: np.ndarray,
    oot_mask: np.ndarray,
    aperture: tuple[int, int],
    wavelength_left: int,
    wavelength_right: int,
) -> np.ndarray:
    """Per-integration trace-position drift, for pointing-systematics decorrelation.

    Returns an (n_frames, 2) array of OOT-referenced drift vectors:

      * column 0 — **y-drift** (cross-dispersion / spatial), the aperture-
        weighted spatial centroid of the rectified ``extract_2D`` cube over the
        science columns, minus its OOT median.  Units: detector rows.
      * column 1 — **x-shift** (dispersion), a linearized least-squares shift of
        each integration's 1D spectrum against the OOT-median spectrum
        (``dx = Σ Δs·s' / Σ s'²``), minus its OOT median.  Units: columns.

    On JWST BOTS these sub-pixel drifts modulate the captured flux at the
    aperture edge and are the usual driver of white-light red noise; feed the
    output to ``fit_wl_mcmc_joint(detrend_vectors=...)`` / ``fit_spec_curves``.
    """
    up, down = int(aperture[0]), int(aperture[1])
    wl_left = max(0, wavelength_left)
    wl_right = clean_2D.shape[1] if wavelength_right is None else wavelength_right
    oot_mask = np.asarray(oot_mask, dtype=bool)

    # y-drift: spatial centroid of the rectified cube over science columns.
    sub = extract_2D[:, up:down, wl_left:wl_right]            # (n, ap_h, ncol)
    prof = np.nansum(sub, axis=2)                            # (n, ap_h)
    prof = np.where(prof > 0, prof, 0.0)
    rows = np.arange(up, down, dtype=float)
    tot = np.nansum(prof, axis=1)
    ycen = np.nansum(prof * rows[None, :], axis=1) / np.where(tot > 0, tot, np.nan)
    y_drift = ycen - np.nanmedian(ycen[oot_mask])

    # x-shift: linearized LSQ of each spectrum vs the OOT-median spectrum.
    spec = clean_2D[:, wl_left:wl_right].astype(float)       # (n, ncol)
    spec_med = np.nanmedian(spec[oot_mask], axis=0)
    sprime = np.gradient(spec_med)
    good = np.isfinite(spec_med) & np.isfinite(sprime)
    denom = np.nansum(sprime[good] ** 2)
    if denom > 0:
        dx = np.array([
            np.nansum((spec[i] - spec_med)[good] * sprime[good]) / denom
            for i in range(spec.shape[0])
        ])
    else:
        dx = np.zeros(spec.shape[0])
    x_shift = dx - np.nanmedian(dx[oot_mask])

    out = np.column_stack([y_drift, x_shift])
    out[~np.isfinite(out)] = 0.0      # neutral for a linear regressor
    return out


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
