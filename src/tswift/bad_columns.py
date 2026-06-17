"""Repair outlier columns in the per-channel transit spectrum.

After ``fit_spec_curves`` produces ``spec_fit.npy``, the per-channel
``Rp/Rs`` spectrum will sometimes have a small handful of channels
whose depths sit several × the local MAD off the spectral baseline.
The standard MAD bad-pixel clip in ``mad_clip`` only catches
*time-varying* outliers — a stable hot pixel that contributes a
constant ~5-10 kADU/sec on top of the trace is invisible to the time
clip but biases the per-channel depth (CLAUDE.md pitfall #6b).

This module:

1. **Finds outlier channels** by rolling-median + MAD across the
   spectrum (excluding the user's ``protected_wvl_um`` list — at
   minimum the He I 1083 nm column for SOSS).
2. **Diagnoses bad pixels** at each outlier column by comparing the
   median 2D frame at that column to neighbour columns; flags any
   row in the aperture where the value is ``n_sigma`` × MAD over the
   neighbour-row median.
3. **Repairs** by re-extracting the column light curve with those
   pixels masked, then re-fitting the transit depth with the same
   geometry as ``fit_spec_curves`` (a, inc, t0_offset fixed; per-bin
   slope, rp, constant, LD2 free).  LD1 is fixed at the per-channel
   exotic_ld value.
4. **Falls back to column masking** when repair doesn't bring the
   depth within ``post_repair_n_sigma`` of the local median (e.g. a
   snowball event that affects multiple rows time-dependently).

The returned `RepairResult` includes a per-column report so you can
audit what was changed and why.

Use as a stage AFTER ``spec``:

    repair = repair_outlier_columns(
        clean_2D, data_fixed, trace_fit, oot_mask,
        wvl, time_hr, spec_fit, spec_err, u1_per_wvl, u2_per_wvl,
        geom={"a": ..., "inc": ..., "t0_offset": ...},
        period_days=..., aperture=(up, down), trace_half_width=22,
        protected_wvl_um=[1.0833],     # always protect He
    )
    np.save("product/clean_2D.npy",   repair.clean_2D_repaired)
    np.save("product/spec_fit.npy",   repair.spec_fit_repaired)
    np.save("product/spec_fit_err.npy", repair.spec_err_repaired)
    plot_repair_diagnostics(repair, outdir="product/figure",
                            planet_name=...)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """Result of a `repair_outlier_columns` run."""
    # Per-column reports.  One entry per OUTLIER column (non-outliers are
    # left untouched in the returned arrays).
    reports: list[dict] = field(default_factory=list)
    # Final arrays — caller saves these to product/.
    clean_2D_repaired: np.ndarray = field(repr=False, default=None)
    spec_fit_repaired: np.ndarray = field(repr=False, default=None)
    spec_err_repaired: np.ndarray = field(repr=False, default=None)
    # Bookkeeping for subsequent `combine_spectrum` masking.
    bad_columns_masked:   list[int] = field(default_factory=list)
    bad_columns_repaired: list[int] = field(default_factory=list)
    # Original (pre-repair) versions — useful for diagnostics.
    spec_fit_original:   np.ndarray = field(repr=False, default=None)
    spec_err_original:   np.ndarray = field(repr=False, default=None)
    # Spectrum-wide rolling-median used for outlier detection
    rolling_median_rp: np.ndarray = field(repr=False, default=None)


# ---------------------------------------------------------------------------
# 1. Find outlier channels
# ---------------------------------------------------------------------------

def find_outlier_channels(
    rp: np.ndarray,
    rp_err: np.ndarray,
    wvl: np.ndarray,
    *,
    window: int = 41,
    n_sigma: float = 5.0,
    min_periods: int = 20,
    protected_wvl_um: Optional[Sequence[float]] = None,
    protect_half_nm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag channels far from the local rolling median.

    Returns
    -------
    outlier_cols : (k,) int  — column indices of flagged channels
    rolling_med  : (n_cols,) float
    rolling_mad  : (n_cols,) float (1.4826 × MAD; gaussian-σ-equivalent)
    """
    valid = np.isfinite(rp) & (rp > 0) & np.isfinite(rp_err) & (rp_err > 0)
    rp_s = pd.Series(np.where(valid, rp, np.nan))
    rolling_med = (
        rp_s.rolling(window, center=True, min_periods=min_periods)
            .median().to_numpy()
    )
    rolling_mad = (
        (rp_s - rolling_med).abs()
        .rolling(window, center=True, min_periods=min_periods)
        .median().to_numpy() * 1.4826
    )
    sig = (rp - rolling_med) / np.maximum(rolling_mad, 1e-5)
    sig = np.where(valid, sig, 0.0)

    # Protect known atmospheric features (He I 1083 nm at minimum).
    if protected_wvl_um:
        for wp_um in protected_wvl_um:
            wp_low = wp_um - protect_half_nm / 1000.0
            wp_high = wp_um + protect_half_nm / 1000.0
            protect = (wvl > wp_low) & (wvl < wp_high)
            sig[protect] = 0.0

    outlier_cols = np.where(np.abs(sig) > n_sigma)[0]
    return outlier_cols, rolling_med, rolling_mad


# ---------------------------------------------------------------------------
# 2. Diagnose bad pixels in the aperture of one column
# ---------------------------------------------------------------------------

def find_aperture_bad_pixels(
    data_fixed: np.ndarray,
    col: int,
    ap_abs_lo: int,
    ap_abs_hi: int,
    *,
    neighbour_cols: Sequence[int] = (-3, -2, -1, +1, +2, +3),
    n_sigma_value: float = 5.0,
    n_sigma_var: float = 3.0,
) -> list[int]:
    """Find rows in the aperture of `col` whose median value is far above
    the same row in adjacent columns OR whose time variability is much
    larger than neighbours.

    Returns
    -------
    bad_rows : list[int]  (absolute detector rows)
    """
    n_frames, n_rows, n_cols = data_fixed.shape
    bad: list[int] = []
    for r in range(ap_abs_lo, ap_abs_hi):
        if r < 0 or r >= n_rows:
            continue
        target_ts = data_fixed[:, r, col]
        if not np.any(np.isfinite(target_ts)):
            continue
        target_med = float(np.nanmedian(target_ts))
        target_std = float(np.nanstd(target_ts))

        # Neighbour values at the same row (excluding col).
        nbr_vals = []
        nbr_stds = []
        for dc in neighbour_cols:
            c = col + dc
            if 0 <= c < n_cols:
                ts = data_fixed[:, r, c]
                if np.any(np.isfinite(ts)):
                    nbr_vals.append(float(np.nanmedian(ts)))
                    nbr_stds.append(float(np.nanstd(ts)))
        if len(nbr_vals) < 2:
            continue
        nbr_med = float(np.nanmedian(nbr_vals))
        nbr_mad = float(np.nanmedian(np.abs(np.asarray(nbr_vals) - nbr_med))) * 1.4826
        nbr_std_med = float(np.nanmedian(nbr_stds))

        value_excess = (target_med - nbr_med) / max(nbr_mad, 1.0)
        var_excess = target_std / max(nbr_std_med, 1.0)

        if value_excess > n_sigma_value or var_excess > n_sigma_var:
            bad.append(r)

    return bad


# ---------------------------------------------------------------------------
# 3. Re-extract a column LC with bad rows masked
# ---------------------------------------------------------------------------

def reextract_column_lc(
    data_fixed: np.ndarray,
    col: int,
    ap_abs_lo: int,
    ap_abs_hi: int,
    *,
    mask_rows: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Sum data_fixed[:, ap_abs_lo:ap_abs_hi, col] across rows, NaN-ing
    any rows in `mask_rows` first."""
    cube = data_fixed[:, ap_abs_lo:ap_abs_hi, col].astype(float).copy()
    if mask_rows:
        for r in mask_rows:
            r_local = r - ap_abs_lo
            if 0 <= r_local < cube.shape[1]:
                cube[:, r_local] = np.nan
    return np.nansum(cube, axis=1)


# ---------------------------------------------------------------------------
# 4. Refit the per-channel transit depth
# ---------------------------------------------------------------------------

def _fit_one_column(
    lc: np.ndarray,
    time_hr: np.ndarray,
    oot_mask: np.ndarray,
    *,
    geom: dict,
    period_hr: float,
    ecc: float,
    omega: float,
    u1: float,
    u2: float,
    rp_init: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Single-column curve_fit with `[slope, rp, constant, LD2]` free.

    Column order matches ``fit_spec_curves`` exactly so the repaired params can
    be written straight back into spec_fit.npy and read by red_noise.

    Returns
    -------
    popt : (4,)  best-fit params [slope, rp, constant, LD2]
    perr : (4,)  1σ from cov
    rms  : float  residual RMS
    """
    import batman
    from scipy.optimize import curve_fit

    valid = np.isfinite(lc)
    med = float(np.nanmedian(lc[oot_mask & valid]))
    if not np.isfinite(med) or med == 0:
        return np.array([np.nan]*4), np.array([np.nan]*4), np.nan
    lc_n = lc / med

    oot_idx = np.where(oot_mask)[0]
    time_phase = time_hr / period_hr

    # Build geometry-only TransitModel once.
    p0 = batman.TransitParams()
    p0.t0 = geom["t0_offset"] / period_hr
    p0.per = 1.0; p0.rp = 0.1
    p0.a = geom["a"]; p0.inc = geom["inc"]
    p0.ecc = ecc; p0.w = omega
    p0.u = [u1, u2]; p0.limb_dark = "quadratic"
    m_geom = batman.TransitModel(p0, time_phase)

    t_ref = float(np.median(time_hr))

    def model_lc(_t, slope, rp_, const_, LD2_):
        p = batman.TransitParams()
        p.t0 = geom["t0_offset"] / period_hr; p.per = 1.0
        p.rp = rp_; p.a = geom["a"]; p.inc = geom["inc"]
        p.ecc = ecc; p.w = omega; p.u = [u1, LD2_]
        p.limb_dark = "quadratic"
        flux = m_geom.light_curve(p)
        norm = float(np.nanmedian(flux[oot_idx]))
        return (flux / norm) * const_ + slope * (time_hr - t_ref)

    err = np.full(int(valid.sum()), float(np.nanstd(lc_n[oot_mask & valid])))
    p0_init = [0.0, rp_init, 1.0, u2]
    try:
        popt, pcov = curve_fit(
            lambda t, *theta: model_lc(t, *theta)[valid],
            time_hr[valid], lc_n[valid],
            p0=p0_init, sigma=err, absolute_sigma=True, maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        res = lc_n[valid] - model_lc(time_hr[valid], *popt)[valid]
        rms = float(np.std(res))
    except Exception as exc:
        logger.warning("col fit failed: %s", exc)
        return np.array([np.nan]*4), np.array([np.nan]*4), np.nan

    return popt, perr, rms


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def repair_outlier_columns(
    clean_2D: np.ndarray,
    data_fixed: np.ndarray,
    trace_fit: np.ndarray,
    oot_mask: np.ndarray,
    wvl: np.ndarray,
    time_hr: np.ndarray,
    spec_fit: np.ndarray,
    spec_err: np.ndarray,
    u1_per_wvl: np.ndarray,
    u2_per_wvl: np.ndarray,
    *,
    geom: dict,                     # {"a", "inc", "t0_offset"}
    period_days: float,
    aperture: tuple[int, int],      # cutout-row aperture (lo, hi)
    trace_half_width: int,
    ecc: float = 0.0,
    omega: float = 90.0,
    detection_window: int = 41,
    detection_n_sigma: float = 5.0,
    pixel_n_sigma_value: float = 5.0,
    pixel_n_sigma_var: float = 3.0,
    post_repair_n_sigma: float = 4.0,
    protected_wvl_um: Optional[Sequence[float]] = None,
    protect_half_nm: float = 5.0,
) -> RepairResult:
    """Find + repair (or mask) outlier channels in the per-channel spectrum.

    Parameters
    ----------
    clean_2D : (n_frames, n_cols)
        Per-channel cleaned light curves (output of `run_extract`).
    data_fixed : (n_frames, n_rows, n_cols)
        Bad-pixel-cleaned data cube (output of `mad_clip`).
    trace_fit : (n_cols,)
        Trace polynomial center per column.
    oot_mask : (n_frames,) bool
    wvl : (n_cols,) µm
    time_hr : (n_frames,) hours
    spec_fit, spec_err : (n_cols, 4)
        Per-channel curve_fit results from `fit_spec_curves`.  Columns:
        slope, rp, constant, LD2.
    u1_per_wvl, u2_per_wvl : (n_cols,) limb-darkening per column
    geom : dict  fixed orbital geometry
    aperture : (lo, hi) cutout-row range used by `run_extract`
    trace_half_width : int — half-width of the cutout in `run_extract`
    detection_window, detection_n_sigma : outlier-channel detection
    pixel_n_sigma_value : per-pixel value threshold (vs neighbour columns)
    pixel_n_sigma_var : per-pixel time-variability threshold
    post_repair_n_sigma : if repair doesn't bring the channel within this
        many σ of the local median, mask the entire column instead
    protected_wvl_um : list of µm to never flag as outlier (e.g. He I)

    Returns
    -------
    RepairResult
    """
    rp = np.asarray(spec_fit[:, 1])
    rp_err = np.asarray(spec_err[:, 1])
    period_hr = period_days * 24.0
    n_frames, n_cols = clean_2D.shape
    ap_lo, ap_hi = aperture

    out_cols, rolling_med, rolling_mad = find_outlier_channels(
        rp, rp_err, wvl,
        window=detection_window, n_sigma=detection_n_sigma,
        protected_wvl_um=protected_wvl_um,
        protect_half_nm=protect_half_nm,
    )
    logger.info(
        "outlier_columns: %d channels at >%.1fσ (window=%d, %d protected wvl)",
        len(out_cols), detection_n_sigma, detection_window,
        len(protected_wvl_um) if protected_wvl_um else 0,
    )

    clean_2D_out = clean_2D.copy()
    spec_fit_out = spec_fit.copy()
    spec_err_out = spec_err.copy()
    reports: list[dict] = []
    repaired: list[int] = []
    masked: list[int] = []

    for col in out_cols:
        col = int(col)
        rp_before = float(rp[col])
        depth_before = rp_before ** 2 * 1e6 if rp_before > 0 else np.nan
        local_rp = float(rolling_med[col])
        local_depth = local_rp ** 2 * 1e6 if np.isfinite(local_rp) else np.nan
        sig_before = (rp_before - local_rp) / max(rolling_mad[col], 1e-5)

        # Aperture in absolute detector rows.
        tc = float(trace_fit[col])
        abs_lo = int(round(tc)) - trace_half_width + ap_lo
        abs_hi = int(round(tc)) - trace_half_width + ap_hi
        bad_rows = find_aperture_bad_pixels(
            data_fixed, col, abs_lo, abs_hi,
            n_sigma_value=pixel_n_sigma_value,
            n_sigma_var=pixel_n_sigma_var,
        )

        report = {
            "col": col,
            "wvl_nm": float(wvl[col] * 1000),
            "rp_before": rp_before,
            "depth_before_ppm": float(depth_before),
            "local_rp": local_rp,
            "local_depth_ppm": float(local_depth),
            "sigma_before": float(sig_before),
            "trace_center_row": tc,
            "aperture_abs_lo": abs_lo,
            "aperture_abs_hi": abs_hi,
            "bad_rows": list(bad_rows),
        }

        if not bad_rows:
            # No diagnosable bad pixel — mask the column.
            spec_fit_out[col, :] = np.nan
            spec_err_out[col, :] = np.nan
            clean_2D_out[:, col] = np.nan
            report["action"] = "masked_column"
            report["reason"] = "no aperture-row diagnosable as bad pixel"
            masked.append(col)
            logger.warning(
                "  col %4d (%.2f nm, %.1fσ) — no bad pixel found → masked",
                col, wvl[col] * 1000, sig_before,
            )
            reports.append(report)
            continue

        # Re-extract + refit.
        lc_new = reextract_column_lc(
            data_fixed, col, abs_lo, abs_hi, mask_rows=bad_rows,
        )
        u1, u2 = float(u1_per_wvl[col]), float(u2_per_wvl[col])
        if not (np.isfinite(u1) and np.isfinite(u2)):
            spec_fit_out[col, :] = np.nan
            spec_err_out[col, :] = np.nan
            clean_2D_out[:, col] = np.nan
            report["action"] = "masked_column"
            report["reason"] = "non-finite LD coefficient"
            masked.append(col)
            reports.append(report)
            continue

        popt, perr, rms = _fit_one_column(
            lc_new, time_hr, oot_mask,
            geom=geom, period_hr=period_hr, ecc=ecc, omega=omega,
            u1=u1, u2=u2, rp_init=local_rp,
        )
        rp_after = float(popt[1])
        depth_after = rp_after ** 2 * 1e6 if rp_after > 0 else np.nan
        sig_after = (rp_after - local_rp) / max(rolling_mad[col], 1e-5)

        if (
            np.isfinite(rp_after)
            and abs(sig_after) <= post_repair_n_sigma
            and rp_after > 0
        ):
            # Accept repair.
            clean_2D_out[:, col] = lc_new
            spec_fit_out[col, :] = popt
            spec_err_out[col, :] = perr
            report.update({
                "action": "repaired",
                "rp_after": rp_after,
                "depth_after_ppm": float(depth_after),
                "rms_after_ppm": float(rms * 1e6) if np.isfinite(rms) else None,
                "sigma_after": float(sig_after),
                "depth_change_ppm": float(depth_after - depth_before),
            })
            repaired.append(col)
            logger.info(
                "  col %4d (%.2f nm) %.1fσ → %.1fσ  "
                "(%.0f → %.0f ppm)  bad_rows=%s",
                col, wvl[col] * 1000, sig_before, sig_after,
                depth_before, depth_after, bad_rows,
            )
        else:
            # Repair didn't help — mask the column.
            spec_fit_out[col, :] = np.nan
            spec_err_out[col, :] = np.nan
            clean_2D_out[:, col] = np.nan
            report.update({
                "action": "masked_column",
                "reason": (
                    f"post-repair sigma {sig_after:.1f} > "
                    f"{post_repair_n_sigma:.1f}"
                ),
                "rp_after_attempted": rp_after,
                "depth_after_attempted_ppm": float(depth_after),
            })
            masked.append(col)
            logger.warning(
                "  col %4d (%.2f nm) %.1fσ → %.1fσ AFTER repair "
                "(%.0f → %.0f ppm) — masked",
                col, wvl[col] * 1000, sig_before, sig_after,
                depth_before, depth_after,
            )
        reports.append(report)

    return RepairResult(
        reports=reports,
        clean_2D_repaired=clean_2D_out,
        spec_fit_repaired=spec_fit_out,
        spec_err_repaired=spec_err_out,
        bad_columns_masked=masked,
        bad_columns_repaired=repaired,
        spec_fit_original=spec_fit,
        spec_err_original=spec_err,
        rolling_median_rp=rolling_med,
    )


# ---------------------------------------------------------------------------
# Plot diagnostics
# ---------------------------------------------------------------------------

def plot_repair_diagnostics(
    repair: RepairResult,
    wvl: np.ndarray,
    outdir: str | Path,
    *,
    planet_name: str = "",
    detector: str = "",
) -> dict:
    """Write before/after spectrum plot + per-column LC summary.

    Returns ``dict[name → Path]``.
    """
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    rp_before = repair.spec_fit_original[:, 1]
    rp_err_before = repair.spec_err_original[:, 1]
    rp_after = repair.spec_fit_repaired[:, 1]
    rp_err_after = repair.spec_err_repaired[:, 1]
    depth_before = rp_before ** 2 * 1e6
    depth_after = rp_after ** 2 * 1e6
    err_before = 2 * np.abs(rp_before) * rp_err_before * 1e6
    err_after = 2 * np.abs(rp_after) * rp_err_after * 1e6
    rolling_med_depth = repair.rolling_median_rp ** 2 * 1e6

    title_tag = planet_name + (f" — {detector.upper()}" if detector else "")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].errorbar(wvl * 1000, depth_before, yerr=err_before, fmt=".",
                     ms=2, alpha=0.4, color="tab:gray", label="before")
    axes[0].plot(wvl * 1000, rolling_med_depth,
                 color="black", lw=1.0, label="local median")
    for col in repair.bad_columns_repaired:
        axes[0].errorbar(wvl[col] * 1000, depth_before[col],
                         yerr=err_before[col], fmt="o", ms=8,
                         color="tab:orange", alpha=0.9)
    for col in repair.bad_columns_masked:
        axes[0].errorbar(wvl[col] * 1000, depth_before[col],
                         yerr=err_before[col], fmt="x", ms=10,
                         color="tab:red", alpha=0.9)
    axes[0].scatter([], [], s=80, color="tab:orange",
                    label=f"repaired ({len(repair.bad_columns_repaired)})")
    axes[0].scatter([], [], s=80, color="tab:red", marker="x",
                    label=f"masked ({len(repair.bad_columns_masked)})")
    axes[0].set_ylabel("Depth before (ppm)")
    axes[0].set_title(f"{title_tag} — bad-column repair")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)
    if depth_before[np.isfinite(depth_before)].size:
        ymed = float(np.nanmedian(rolling_med_depth))
        axes[0].set_ylim(ymed - 3000, ymed + 3000)

    axes[1].errorbar(wvl * 1000, depth_after, yerr=err_after, fmt=".",
                     ms=2, alpha=0.4, color="tab:blue", label="after")
    axes[1].plot(wvl * 1000, rolling_med_depth,
                 color="black", lw=1.0, label="local median")
    for col in repair.bad_columns_repaired:
        axes[1].errorbar(wvl[col] * 1000, depth_after[col],
                         yerr=err_after[col], fmt="o", ms=8,
                         color="tab:orange", alpha=0.9)
    axes[1].set_ylabel("Depth after (ppm)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)
    if depth_after[np.isfinite(depth_after)].size:
        ymed = float(np.nanmedian(rolling_med_depth))
        axes[1].set_ylim(ymed - 3000, ymed + 3000)
    plt.tight_layout()
    p = outdir / "bad_column_repair.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths["repair_spectrum"] = p.resolve()

    # Per-column report (machine-readable).
    p_json = outdir / "bad_column_repair.json"
    with open(p_json, "w") as f:
        json.dump({
            "n_repaired": len(repair.bad_columns_repaired),
            "n_masked":   len(repair.bad_columns_masked),
            "repaired_cols": list(repair.bad_columns_repaired),
            "masked_cols":   list(repair.bad_columns_masked),
            "reports": repair.reports,
        }, f, indent=2)
    paths["report"] = p_json.resolve()

    return paths
