"""Per-wavelength transit curve fitting with scipy.optimize.curve_fit.

For each wavelength column, fits a transit model with free [slope, rp, constant, LD2]
while holding orbital geometry (a, inc, t0) fixed at the white-light best-fit. This
assumes the transit shape is dominated by orbital geometry (shared across wavelengths)
and only the depth + limb-darkening vary with color — standard assumption for
spectroscopic transit fits.

Limb-darkening coefficients come from exotic_ld per wavelength. At grid edges exotic_ld
raises; we fill those with nearest-neighbor values from successfully computed
wavelengths rather than a global fallback. The old fallback of (u1=0.5, u2=0.1) biased
rp by ~300 ppm at the blue end of NRS1 — see comment in legacy step 7.

`fix_ld2_stagger`: if True, drops LD2 from the fit (uses exotic_ld's u2 per wavelength).
Eliminates the rp–LD2 degeneracy that can produce inverted features when LD2 hits the
±0.5 bound; recommended for SOSS. Default False for backward-compat.
"""
from __future__ import annotations

import logging
from typing import Optional

import batman
import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def compute_ld_per_wavelength(
    wvl: np.ndarray,
    wavelength_left: int,
    wavelength_right: int,
    *,
    stellar_teff: float,
    stellar_logg: float,
    stellar_mh: float,
    ld_model: str,
    ld_mode: str,
    ld_data_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute quadratic LD coeffs (u1, u2) per wavelength with NN fill at grid edges.

    Returns two arrays of length `len(wvl)`. Entries outside [left, right) are untouched
    (still NaN); entries where exotic_ld raised are filled with the nearest successful
    neighbor's values.
    """
    # Local import keeps import-time cost low for users who don't fit spectra.
    from exotic_ld import StellarLimbDarkening

    sld = StellarLimbDarkening(
        stellar_mh, stellar_teff, stellar_logg,
        ld_model, ld_data_path, verbose=False,
    )

    half = (wvl[10] - wvl[9]) / 2 if len(wvl) > 10 else (wvl[1] - wvl[0]) / 2
    u1 = np.full(len(wvl), np.nan)
    u2 = np.full(len(wvl), np.nan)
    for i in range(wavelength_left, wavelength_right):
        wvl_range_aa = np.array([wvl[i] - half, wvl[i] + half]) * 10_000.0
        try:
            u1[i], u2[i] = sld.compute_quadratic_ld_coeffs(wvl_range_aa, ld_mode)
        except Exception:
            pass

    valid_ld = np.isfinite(u1) & np.isfinite(u2)
    n_missing = int((~valid_ld[wavelength_left:wavelength_right]).sum())
    if n_missing:
        valid_idx = np.where(valid_ld)[0]
        if len(valid_idx) == 0:
            logger.error("No valid LD coefficients — using global fallback (0.5, 0.1)")
            u1[:], u2[:] = 0.5, 0.1
        else:
            bad = np.where(~valid_ld)[0]
            for i in bad:
                j = valid_idx[np.argmin(np.abs(valid_idx - i))]
                u1[i], u2[i] = u1[j], u2[j]
            logger.warning(
                f"Limb-darkening grid missed {n_missing}/{wavelength_right - wavelength_left} "
                "wavelengths; filled with nearest-neighbor values."
            )
    return u1, u2


def fit_spec_curves(
    clean_2D: np.ndarray,
    time_hr: np.ndarray,
    wvl: np.ndarray,
    *,
    wavelength_left: int,
    wavelength_right: int,
    # Orbital geometry fixed from WL fit
    period_days: float,
    a_over_rs: float,
    inclination_deg: float,
    t0_offset_hr: float,
    ecc: float = 0.0,
    omega: float = 90.0,
    # Per-wavelength LD (pre-computed by compute_ld_per_wavelength)
    u1_arr: np.ndarray,
    u2_arr: np.ndarray,
    # Fit options
    bounds_lower: tuple = (-0.005, 0.04, 0.995, -0.5),
    bounds_upper: tuple = (0.005, 0.15, 1.005, 0.5),
    fix_ld2: bool = False,
    mask_indices: Optional[np.ndarray] = None,
) -> dict:
    """Fit a transit depth at every wavelength column.

    Parameters follow the legacy step 7 convention. The returned dict mirrors the
    legacy output files:

    - `fit`        : (n_wvl, 4) → [slope, rp, constant, LD2]
    - `fit_err`    : (n_wvl, 4) → 1-σ uncertainties (0 for fixed LD2)
    - `residuals_rms`: (n_wvl,) std of (data - model)
    - `bandpass`   : slice(wavelength_left, wavelength_right)

    Every `wvl[i]` for i outside [left, right) gets all-NaN entries so downstream
    combine code can keep the indexing simple.
    """
    if clean_2D.ndim != 2:
        raise ValueError(f"expected clean_2D (n_frames, n_wvl), got {clean_2D.shape}")
    n_frames, n_wvl = clean_2D.shape
    right = min(wavelength_right, n_wvl)
    if wavelength_left >= right:
        raise ValueError(f"bad wavelength window [{wavelength_left}, {right})")

    # Per-channel OOT normalization.
    n_edge = max(10, int(n_frames * 0.15))
    oot_mask = np.zeros(n_frames, bool)
    oot_mask[:n_edge] = True
    oot_mask[-n_edge:] = True
    oot_median = np.nanmedian(clean_2D[oot_mask, :], axis=0)[None, :]
    cl_nor = clean_2D / oot_median

    # Mask bad integrations (applied before fit)
    if mask_indices is not None and len(mask_indices) > 0:
        cl_nor[np.asarray(mask_indices, dtype=int), :] = np.nan

    # Reference a column in the middle of the bandpass for the valid-frame mask.
    ref_col = (wavelength_left + right) // 2
    valid_frames = ~np.isnan(cl_nor[:, ref_col])
    x = time_hr[valid_frames]
    cl_fit = cl_nor[valid_frames]
    logger.info(
        f"Fitting {right - wavelength_left} wavelengths × {valid_frames.sum()} frames "
        f"(fix_ld2={fix_ld2})"
    )

    # Transit model closure. a/inc/t0 are closed over.
    t0_days = t0_offset_hr / 24.0
    def _model(x_hr, slope, rp, constant, ld2, u1_val):
        t = x_hr / 24.0
        p = batman.TransitParams()
        p.t0, p.per, p.rp = t0_days, period_days, rp
        p.a, p.inc = a_over_rs, inclination_deg
        p.ecc, p.w = ecc, omega
        p.u, p.limb_dark = [u1_val, ld2], "quadratic"
        m = batman.TransitModel(p, t)
        return m.light_curve(p) * constant + slope * (x_hr - x_hr[0])

    if fix_ld2:
        lower = bounds_lower[:3]
        upper = bounds_upper[:3]
    else:
        lower = bounds_lower
        upper = bounds_upper

    fit_arr = np.full((n_wvl, 4), np.nan)
    err_arr = np.full((n_wvl, 4), np.nan)
    rms_arr = np.full(n_wvl, np.nan)

    for i in range(wavelength_left, right):
        lc = cl_fit[:, i]
        u1_i, u2_i = u1_arr[i], u2_arr[i]
        if not np.isfinite(u1_i):
            continue
        try:
            if fix_ld2:
                popt, pcov = curve_fit(
                    lambda xx, s, r, c: _model(xx, s, r, c, u2_i, u1_i),
                    x, lc, bounds=(lower, upper),
                )
                perr = np.sqrt(np.diag(pcov))
                fit_arr[i] = [popt[0], popt[1], popt[2], u2_i]
                err_arr[i] = [perr[0], perr[1], perr[2], 0.0]
                model = _model(x, *popt, u2_i, u1_i)
            else:
                popt, pcov = curve_fit(
                    lambda xx, s, r, c, l: _model(xx, s, r, c, l, u1_i),
                    x, lc, bounds=(lower, upper),
                )
                perr = np.sqrt(np.diag(pcov))
                fit_arr[i] = popt
                err_arr[i] = perr
                model = _model(x, *popt, u1_i)
            rms_arr[i] = float(np.nanstd(lc - model))
        except Exception as e:
            # Leave NaNs — downstream combine code drops NaN rows.
            logger.debug(f"wvl {i} failed: {e}")

    n_good = int(np.sum(np.isfinite(fit_arr[:, 1])))
    logger.info(f"Successfully fit {n_good}/{right - wavelength_left} wavelengths")

    return {
        "fit": fit_arr,
        "fit_err": err_arr,
        "residuals_rms": rms_arr,
        "wavelength_left": wavelength_left,
        "wavelength_right": right,
    }
