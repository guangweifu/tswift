"""Optional: morning vs evening limb spectra via per-bin catwoman MCMC.

Tomographic / asymmetric limb retrieval on top of the standard symmetric
batman pipeline.  Most JWST transit programs need only the symmetric
spectrum from `spec_fit` + `combine`; this module is **opt-in** for
targets where you specifically want to constrain limb-to-limb differences
(hot-Jupiter day-night asymmetry, oblique terminators, asymmetric clouds).

Approach (mirrors the TOI-6894 b reference reduction)
-----------------------------------------------------
- **Catwoman model** with `rp1`, `rp2` for the two limbs (`phi=90` →
  `rp1` = evening / leading limb, `rp2` = morning / trailing limb).
- **Reparameterization**: sample `(rp_mean, drp)` with
  `rp1 = rp_mean − drp/2`, `rp2 = rp_mean + drp/2`.  rp_mean is the
  symmetric depth (well-constrained) and drp is the half-difference
  (small).  Sampling the natural axes of the posterior avoids the strong
  rp1↔rp2 anti-correlation that wastes MCMC steps.
- **Geometry FIXED** to the joint white-light catwoman/batman fit:
  `a, inc, t0_offset, period, ecc, omega` are all locked to the values
  passed in `geom` so each bin's MCMC fits only `slope, rp_mean, drp,
  constant, [LD1,] LD2`.
- **LD1 fixed by default** (CLAUDE.md pitfall #21) — fitting both LD1
  and LD2 opens the LD1↔LD2 ridge that masquerades as drp.

Convention
----------
- `rp1` = evening (leading) limb depth: rp1² is `evening_depth_ppm`.
- `rp2` = morning (trailing) limb depth: rp2² is `morning_depth_ppm`.
- `Δdepth = rp2² − rp1²` is "depth(morning) − depth(evening)" (positive
  → morning limb deeper, negative → evening deeper).
- `drp = rp2 − rp1 = R_morning − R_evening` in stellar-radius units.

Output: per-bin medians + 1-sigma errors saved as `limb_spectra.txt`,
plus four diagnostic figures (morning_evening, delta_depth, drp,
symmetric_depth).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Module-level state used by emcee workers; set per-process via
# `_init_la_globals`.  Catwoman's `TransitModel` is not picklable so
# this follows the same pattern as `transit_model._G`.
_LA: dict = {}


# ----------------------------------------------------------------------------
# Bin edge utilities
# ----------------------------------------------------------------------------

def make_uniform_bin_edges(
    wvl_um_lo: float, wvl_um_hi: float, bin_width_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform bin edges covering [wvl_um_lo, wvl_um_hi].

    Returns ``(bin_edges_lo_um, bin_edges_hi_um)`` of equal length.
    """
    width_um = bin_width_nm / 1000.0
    edges = np.arange(wvl_um_lo, wvl_um_hi + width_um, width_um)
    return edges[:-1], edges[1:]


# ----------------------------------------------------------------------------
# Free-parameter bookkeeping
# ----------------------------------------------------------------------------

def _free_param_order(fit_ld1: bool) -> list[str]:
    if fit_ld1:
        return ["slope", "rp_mean", "drp", "constant", "LD1", "LD2"]
    return ["slope", "rp_mean", "drp", "constant", "LD2"]


# ----------------------------------------------------------------------------
# Worker globals + log-probability
# ----------------------------------------------------------------------------

def _init_la_globals(
    geom: dict,
    fit_ld1: bool,
    ecc: float,
    omega: float,
    period_hr: float,
    oot_indices: np.ndarray,
    time_data_ref: float,
    priors: dict,
    catwoman_fac: float,
) -> None:
    _LA["geom"] = geom                # {"a", "inc", "t0_offset"}
    _LA["fit_ld1"] = fit_ld1
    _LA["ecc"] = ecc
    _LA["omega"] = omega
    _LA["period_hr"] = period_hr
    _LA["oot_indices"] = np.asarray(oot_indices, dtype=int)
    _LA["time_data_ref"] = float(time_data_ref)
    _LA["catwoman_fac"] = float(catwoman_fac)
    _LA["param_order"] = _free_param_order(fit_ld1)
    bounds = [priors[p] for p in _LA["param_order"]]

    def log_prior(theta):
        for v, (lo, hi) in zip(theta, bounds):
            if not (lo < v < hi):
                return -np.inf
        return 0.0

    _LA["log_prior_fn"] = log_prior


def _eval_catwoman(
    theta: np.ndarray, time_data_hr: np.ndarray, u1_fixed: float,
) -> np.ndarray:
    """Evaluate catwoman transit + linear baseline + multiplicative constant."""
    import catwoman

    fit_ld1 = _LA["fit_ld1"]
    if fit_ld1:
        slope, rp_mean, drp, constant, LD1, LD2 = theta
    else:
        slope, rp_mean, drp, constant, LD2 = theta
        LD1 = u1_fixed

    rp1 = rp_mean - drp / 2.0
    rp2 = rp_mean + drp / 2.0

    geom = _LA["geom"]
    period_hr = _LA["period_hr"]
    fac = _LA["catwoman_fac"]

    p = catwoman.TransitParams()
    p.t0 = geom["t0_offset"] / period_hr
    p.per = 1.0
    p.rp = float(rp1)
    p.rp2 = float(rp2)
    p.a = float(geom["a"])
    p.inc = float(geom["inc"])
    p.ecc = float(_LA["ecc"])
    p.w = float(_LA["omega"])
    p.phi = 90.0       # phi=90: rp1=evening (leading), rp2=morning (trailing)
    p.u = [float(LD1), float(LD2)]
    p.limb_dark = "quadratic"

    time_phase = time_data_hr / period_hr
    m = catwoman.TransitModel(p, time_phase, fac=fac)
    flux = m.light_curve(p)

    oot = _LA["oot_indices"]
    valid = oot[oot < len(flux)]
    norm = float(np.median(flux[valid])) if valid.size else 1.0
    flux_norm = flux / norm
    return flux_norm * constant + slope * (time_data_hr - _LA["time_data_ref"])


def _log_probability_la(
    theta: np.ndarray, time_data_hr: np.ndarray, flux_data: np.ndarray,
    flux_err: np.ndarray, u1_fixed: float,
) -> float:
    lp = _LA["log_prior_fn"](theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        model = _eval_catwoman(theta, time_data_hr, u1_fixed)
    except Exception:
        return -np.inf
    sigma2 = flux_err ** 2
    ll = float(-0.5 * np.sum((flux_data - model) ** 2 / sigma2
                             + np.log(2 * np.pi * sigma2)))
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ----------------------------------------------------------------------------
# Per-bin MCMC driver
# ----------------------------------------------------------------------------

def fit_limb_asymmetry(
    clean_2D: np.ndarray,
    time_hr: np.ndarray,
    wvl: np.ndarray,
    bin_edges_lo_um: np.ndarray,
    bin_edges_hi_um: np.ndarray,
    *,
    geom: dict,
    period_days: float,
    u1_per_wvl: np.ndarray,
    u2_per_wvl: np.ndarray,
    rp_mean_init: float,
    oot_mask: Optional[np.ndarray] = None,
    ecc: float = 0.0,
    omega: float = 90.0,
    fit_ld1: bool = False,
    drp_range: tuple[float, float] = (-0.05, 0.05),
    rp_mean_range: Optional[tuple[float, float]] = None,
    LD2_range: tuple[float, float] = (-0.5, 0.5),
    LD1_range: tuple[float, float] = (0.0, 1.0),
    constant_range: tuple[float, float] = (0.98, 1.02),
    slope_range: tuple[float, float] = (-0.01, 0.01),
    nwalkers: int = 32,
    nsteps: int = 2000,
    nburn: int = 600,
    catwoman_fac: float = 0.001,
    nprocesses: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> dict:
    """Per-bin catwoman MCMC; returns morning + evening limb depths.

    Parameters
    ----------
    clean_2D : (n_frames, n_cols)
        Per-channel clean light curves from the extract stage.
    time_hr : (n_frames,)
        Time stamps in hours from visit start.
    wvl : (n_cols,)
        Wavelength per column in micrometers.
    bin_edges_lo_um, bin_edges_hi_um : (n_bins,)
        Bin edges; e.g. from `make_uniform_bin_edges`.
    geom : dict
        FIXED orbital geometry from the joint WL fit:
        ``{"a", "inc", "t0_offset"}``.
    period_days : float
    u1_per_wvl, u2_per_wvl : (n_cols,)
        Limb-darkening coefficients per column from
        `compute_ld_per_wavelength`.  Per-bin median is used.
    rp_mean_init : float
        Starting value for rp_mean (≈ Rp/Rs from the WL fit).
    oot_mask : (n_frames,) bool, optional
        OOT mask from the extract stage.  Defaults to first 25 % of
        frames if not provided (rough — pass the real mask).
    fit_ld1 : bool
        If False (default; CLAUDE.md pitfall #21), fix LD1 at the
        per-bin exotic_ld median.  If True, fit LD1 freely.
    rp_mean_range : (lo, hi), optional
        Prior on rp_mean.  Defaults to ``(0.5*rp_mean_init, 1.5*rp_mean_init)``.
    drp_range : (lo, hi)
        Prior on drp = rp2 − rp1.  Default ``(-0.05, 0.05)`` covers
        ±5 % of rp_mean for a hot Jupiter — plenty for known asymmetry
        signals which are < 1 %.
    nwalkers, nsteps, nburn : int
        Per-bin emcee config.  Defaults sufficient for converged fits
        at ~5σ asymmetry significance with 1000+ integrations.
    catwoman_fac : float
        Catwoman accuracy parameter (0.001 → ~50 ppm precision; 0.0001
        → ~5 ppm but ~10× slower).  Default OK for transit-depth fits.

    Returns
    -------
    dict with arrays of length `n_bins`:
        wvl_lo_um, wvl_hi_um, wvl_center_um
        rp_mean, rp_mean_err_lo, rp_mean_err_hi
        drp, drp_err_lo, drp_err_hi
        rp1_sq_ppm, rp1_sq_err_lo_ppm, rp1_sq_err_hi_ppm   (evening)
        rp2_sq_ppm, rp2_sq_err_lo_ppm, rp2_sq_err_hi_ppm   (morning)
        delta_depth_ppm, delta_err_ppm
        slope, slope_err_lo, slope_err_hi
        constant, constant_err_lo, constant_err_hi
        LD2, LD2_err_lo, LD2_err_hi
        LD1, LD1_err_lo, LD1_err_hi   (NaN if fit_ld1=False)
        u1_fixed, u2_init
        rms_residual_ppm
    """
    import emcee

    # --- bin → column-slice mapping ---
    n_bins = len(bin_edges_lo_um)
    if len(bin_edges_hi_um) != n_bins:
        raise ValueError(
            f"bin edge arrays must match: {n_bins} vs {len(bin_edges_hi_um)}"
        )
    # SOSS wavelengths decrease with column index (NIRSpec increases) — pick
    # the matching column for each edge then ensure (col_lo < col_hi)
    # regardless of the wvl direction so `clean_2D[:, col_lo:col_hi]` is
    # always a non-empty slice over the bin's spectral channels.
    raw_a = np.array([int(np.argmin(np.abs(wvl - lo))) for lo in bin_edges_lo_um])
    raw_b = np.array([int(np.argmin(np.abs(wvl - hi))) for hi in bin_edges_hi_um])
    cols_lo = np.minimum(raw_a, raw_b)
    cols_hi = np.maximum(raw_a, raw_b)

    period_hr = period_days * 24.0
    n_frames = clean_2D.shape[0]

    # Per-bin fluxes + intersection of valid-frame masks → fixed time grid.
    bin_flux = np.full((n_bins, n_frames), np.nan)
    keep = np.ones(n_frames, dtype=bool)
    for i in range(n_bins):
        col_lo, col_hi = int(cols_lo[i]), int(cols_hi[i])
        if col_hi <= col_lo:
            keep[:] = keep   # bin invalid; don't touch keep
            continue
        chunk = clean_2D[:, col_lo:col_hi]
        # nansum across cols; mark frames where ALL cols NaN as NaN.
        f = np.nansum(chunk, axis=1)
        all_nan = np.all(np.isnan(chunk), axis=1)
        f[all_nan] = np.nan
        bin_flux[i] = f
        keep &= np.isfinite(f)

    if oot_mask is None:
        oot_mask = np.zeros(n_frames, dtype=bool)
        oot_mask[: n_frames // 4] = True
        oot_mask[3 * n_frames // 4:] = True
        logger.warning(
            "limb_asymmetry: no oot_mask given — defaulting to first/last 25%%"
        )
    oot_mask = np.asarray(oot_mask, dtype=bool)
    t_kept = time_hr[keep]
    oot_kept_mask = oot_mask[keep]
    oot_indices = np.where(oot_kept_mask)[0]
    time_data_ref = float(np.median(t_kept))
    logger.info(
        "limb_asymmetry: %d bins, %d/%d frames kept, %d OOT after intersection",
        n_bins, int(keep.sum()), n_frames, int(oot_kept_mask.sum()),
    )

    # Per-bin LD1/LD2 medians (LD1 fixed; LD2 used as init).
    u1_bins = np.full(n_bins, np.nan)
    u2_bins = np.full(n_bins, np.nan)
    for i in range(n_bins):
        col_lo, col_hi = int(cols_lo[i]), int(cols_hi[i])
        if col_hi > col_lo:
            u1_bins[i] = float(np.nanmedian(u1_per_wvl[col_lo:col_hi]))
            u2_bins[i] = float(np.nanmedian(u2_per_wvl[col_lo:col_hi]))

    fit_keys = _free_param_order(fit_ld1)
    n_free = len(fit_keys)

    if rp_mean_range is None:
        rp_mean_range = (0.5 * rp_mean_init, 1.5 * rp_mean_init)
    priors_template = {
        "slope":    slope_range,
        "rp_mean":  rp_mean_range,
        "drp":      drp_range,
        "constant": constant_range,
        "LD2":      LD2_range,
    }
    if fit_ld1:
        priors_template["LD1"] = LD1_range

    # Output containers.
    out: dict = {
        "wvl_lo_um":     np.asarray(bin_edges_lo_um, dtype=float),
        "wvl_hi_um":     np.asarray(bin_edges_hi_um, dtype=float),
        "wvl_center_um": 0.5 * (np.asarray(bin_edges_lo_um, dtype=float)
                                + np.asarray(bin_edges_hi_um, dtype=float)),
        "u1_fixed":      u1_bins,
        "u2_init":       u2_bins,
    }
    for k in ("slope", "rp_mean", "drp", "constant", "LD2", "LD1"):
        out[k]              = np.full(n_bins, np.nan)
        out[f"{k}_err_lo"]  = np.full(n_bins, np.nan)
        out[f"{k}_err_hi"]  = np.full(n_bins, np.nan)
    for k in ("rp1_sq_ppm", "rp2_sq_ppm"):
        out[k]              = np.full(n_bins, np.nan)
        out[f"{k.replace('_ppm','')}_err_lo_ppm"] = np.full(n_bins, np.nan)
        out[f"{k.replace('_ppm','')}_err_hi_ppm"] = np.full(n_bins, np.nan)
    out["delta_depth_ppm"]   = np.full(n_bins, np.nan)
    out["delta_err_ppm"]     = np.full(n_bins, np.nan)
    out["rms_residual_ppm"]  = np.full(n_bins, np.nan)

    ncpu = nprocesses if nprocesses is not None else mp.cpu_count()
    rng = np.random.default_rng(rng_seed)

    init_args = (geom, fit_ld1, ecc, omega, period_hr,
                 oot_indices, time_data_ref, priors_template, catwoman_fac)
    # Initialize parent process for the post-fit best-curve render.
    _init_la_globals(*init_args)

    logger.info(
        "limb_asymmetry: starting MCMC, %d walkers × %d steps (burn=%d), "
        "fit_ld1=%s, %d workers",
        nwalkers, nsteps, nburn, fit_ld1, ncpu,
    )

    with mp.Pool(ncpu, initializer=_init_la_globals, initargs=init_args) as pool:
        for i in range(n_bins):
            wlo, whi = float(bin_edges_lo_um[i]), float(bin_edges_hi_um[i])
            col_lo, col_hi = int(cols_lo[i]), int(cols_hi[i])
            if col_hi <= col_lo or not np.isfinite(u1_bins[i]):
                logger.warning(
                    "[bin %3d/%d %.4f-%.4f um] skipped (no cols / no LD)",
                    i + 1, n_bins, wlo, whi,
                )
                continue

            f_bin = bin_flux[i][keep]
            if not np.isfinite(f_bin).all():
                logger.warning(
                    "[bin %3d/%d %.4f-%.4f um] residual NaNs after keep — skip",
                    i + 1, n_bins, wlo, whi,
                )
                continue
            wl_med_oot = float(np.nanmedian(f_bin[oot_indices]))
            f_nor = f_bin / wl_med_oot
            oot_scatter = float(np.nanstd(f_nor[oot_indices]))
            if not np.isfinite(oot_scatter) or oot_scatter == 0:
                logger.warning(
                    "[bin %3d/%d] zero/nan OOT scatter — skip",
                    i + 1, n_bins,
                )
                continue
            f_err = np.full(t_kept.size, oot_scatter)

            initial = {
                "slope": 0.0, "rp_mean": rp_mean_init, "drp": 0.0,
                "constant": 1.0, "LD2": float(u2_bins[i]),
            }
            if fit_ld1:
                initial["LD1"] = float(u1_bins[i])
            initial_arr = np.array([initial[k] for k in fit_keys])
            pos = initial_arr + 1e-4 * rng.standard_normal((nwalkers, n_free))

            sampler = emcee.EnsembleSampler(
                nwalkers, n_free, _log_probability_la,
                args=(t_kept, f_nor, f_err, float(u1_bins[i])),
                pool=pool,
            )
            sampler.run_mcmc(pos, nsteps, progress=False)
            flat = sampler.get_chain(discard=nburn, flat=True)
            log_prob_flat = sampler.get_log_prob(discard=nburn, flat=True)

            median = np.percentile(flat, 50, axis=0)
            lo16   = np.percentile(flat, 16, axis=0)
            hi84   = np.percentile(flat, 84, axis=0)

            # Per-param fill.
            for j, k in enumerate(fit_keys):
                out[k][i]              = float(median[j])
                out[f"{k}_err_lo"][i]  = float(median[j] - lo16[j])
                out[f"{k}_err_hi"][i]  = float(hi84[j] - median[j])
            if not fit_ld1:
                out["LD1"][i] = float(u1_bins[i])  # report the fixed value

            # Derived rp1², rp2², Δdepth from the joint chain.
            j_rm = fit_keys.index("rp_mean")
            j_drp = fit_keys.index("drp")
            rp1_s = flat[:, j_rm] - flat[:, j_drp] / 2.0
            rp2_s = flat[:, j_rm] + flat[:, j_drp] / 2.0
            rp1_sq_s = rp1_s ** 2
            rp2_sq_s = rp2_s ** 2
            delta_s = (rp2_sq_s - rp1_sq_s) * 1e6

            out["rp1_sq_ppm"][i]        = float(np.median(rp1_sq_s) * 1e6)
            out["rp1_sq_err_lo_ppm"][i] = float(
                (np.median(rp1_sq_s) - np.percentile(rp1_sq_s, 16)) * 1e6)
            out["rp1_sq_err_hi_ppm"][i] = float(
                (np.percentile(rp1_sq_s, 84) - np.median(rp1_sq_s)) * 1e6)
            out["rp2_sq_ppm"][i]        = float(np.median(rp2_sq_s) * 1e6)
            out["rp2_sq_err_lo_ppm"][i] = float(
                (np.median(rp2_sq_s) - np.percentile(rp2_sq_s, 16)) * 1e6)
            out["rp2_sq_err_hi_ppm"][i] = float(
                (np.percentile(rp2_sq_s, 84) - np.median(rp2_sq_s)) * 1e6)
            out["delta_depth_ppm"][i]   = float(np.median(delta_s))
            out["delta_err_ppm"][i]     = float(
                0.5 * (np.percentile(delta_s, 84) - np.percentile(delta_s, 16)))

            # RMS at MAP.
            map_idx = int(np.nanargmax(log_prob_flat))
            best = flat[map_idx]
            best_curve = _eval_catwoman(best, t_kept, float(u1_bins[i]))
            out["rms_residual_ppm"][i] = float(np.std(f_nor - best_curve) * 1e6)

            logger.info(
                "[bin %3d/%d %.4f-%.4f um] rp_mean=%.5f drp=%+.5f "
                "rp1²=%.0f rp2²=%.0f Δ=%+.0f±%.0f RMS=%.0f ppm",
                i + 1, n_bins, wlo, whi,
                out["rp_mean"][i], out["drp"][i],
                out["rp1_sq_ppm"][i], out["rp2_sq_ppm"][i],
                out["delta_depth_ppm"][i], out["delta_err_ppm"][i],
                out["rms_residual_ppm"][i],
            )

    return out


# ----------------------------------------------------------------------------
# Plotting + machine-readable output
# ----------------------------------------------------------------------------

def plot_limb_asymmetry(
    result: dict,
    outdir: str | Path,
    *,
    planet_name: str = "",
    detector: str = "",
    bad_wavelengths_um: Optional[Sequence[float]] = None,
    sanity_max_delta_ppm: float = 10000.0,
    sanity_min_depth_ppm: float = 0.0,
) -> dict:
    """Write 4 limb-asymmetry diagnostic PNGs + a machine-readable txt.

    Plots
    -----
    - ``morning_evening_spectra.png`` — rp1² (evening, blue) and rp2²
      (morning, red) overplotted vs wavelength.
    - ``delta_depth_vs_wavelength.png`` — Δdepth = rp2² − rp1² with
      inverse-variance-weighted mean shaded band.
    - ``drp_vs_wavelength.png`` — drp = rp2 − rp1 vs wavelength.
    - ``symmetric_depth_vs_wavelength.png`` — rp_mean² (catwoman
      symmetric depth) for cross-check against the standard pipeline's
      `combine_spectrum` output.

    Sanity filters drop bins flagged for: |Δdepth| > sanity_max_delta_ppm,
    rp_mean² < sanity_min_depth_ppm, or wavelength matching any entry in
    `bad_wavelengths_um` (within the bin's [lo, hi] range).
    """
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wvl_c = np.asarray(result["wvl_center_um"])
    wvl_lo = np.asarray(result["wvl_lo_um"])
    wvl_hi = np.asarray(result["wvl_hi_um"])
    rp_mean = np.asarray(result["rp_mean"])
    rp_mean_lo = np.asarray(result["rp_mean_err_lo"])
    rp_mean_hi = np.asarray(result["rp_mean_err_hi"])
    drp = np.asarray(result["drp"])
    drp_lo = np.asarray(result["drp_err_lo"])
    drp_hi = np.asarray(result["drp_err_hi"])
    rp1_sq = np.asarray(result["rp1_sq_ppm"])
    rp1_lo = np.asarray(result["rp1_sq_err_lo_ppm"])
    rp1_hi = np.asarray(result["rp1_sq_err_hi_ppm"])
    rp2_sq = np.asarray(result["rp2_sq_ppm"])
    rp2_lo = np.asarray(result["rp2_sq_err_lo_ppm"])
    rp2_hi = np.asarray(result["rp2_sq_err_hi_ppm"])
    delta = np.asarray(result["delta_depth_ppm"])
    delta_err = np.asarray(result["delta_err_ppm"])
    rms = np.asarray(result["rms_residual_ppm"])

    # Sanity / bad-wvl masking.
    skipped = ~np.isfinite(rp_mean)
    sanity_bad = ((np.abs(delta) > sanity_max_delta_ppm) |
                  ((rp_mean ** 2) * 1e6 < sanity_min_depth_ppm))
    sanity_bad &= ~skipped
    bad_mask = skipped | sanity_bad
    if bad_wavelengths_um:
        for bw in bad_wavelengths_um:
            for i in range(len(wvl_c)):
                if wvl_lo[i] <= bw < wvl_hi[i]:
                    bad_mask[i] = True
    keep = ~bad_mask
    n_dropped = int(bad_mask.sum())

    title_tag = planet_name
    if detector:
        title_tag = f"{title_tag} — {detector.upper()}".strip(" —")

    paths: dict[str, Path] = {}

    # 1. morning vs evening
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(wvl_c[keep], rp1_sq[keep], yerr=[rp1_lo[keep], rp1_hi[keep]],
                fmt="o", ms=5, color="C0", alpha=0.85,
                label="Evening limb (rp$_1^2$)")
    ax.errorbar(wvl_c[keep], rp2_sq[keep], yerr=[rp2_lo[keep], rp2_hi[keep]],
                fmt="o", ms=5, color="C3", alpha=0.85,
                label="Morning limb (rp$_2^2$)")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Transit depth (ppm)")
    ax.set_title(f"{title_tag}  morning vs evening limb spectra")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = outdir / "morning_evening_spectra.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["morning_evening"] = p.resolve()

    # 2. Δdepth vs wavelength
    fig, ax = plt.subplots(figsize=(11, 5))
    avg_str = ""
    if keep.any():
        w = 1.0 / np.maximum(delta_err[keep], 1.0) ** 2
        avg = float(np.sum(w * delta[keep]) / np.sum(w))
        avg_err = float(1.0 / np.sqrt(np.sum(w)))
        ax.axhspan(avg - avg_err, avg + avg_err, color="grey", alpha=0.2,
                   label=f"Wavelength-averaged Δ = {avg:.0f} ± {avg_err:.0f} ppm")
        avg_str = f"avg Δ = {avg:.0f} ± {avg_err:.0f} ppm"
    ax.errorbar(wvl_c[keep], delta[keep], yerr=delta_err[keep],
                fmt="o", ms=5, color="C2", alpha=0.85,
                label="Δdepth = depth(morning) − depth(evening)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Δ Transit depth (ppm)")
    ax.set_title(f"{title_tag}  limb asymmetry  ({n_dropped} bins masked)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = outdir / "delta_depth_vs_wavelength.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["delta_depth"] = p.resolve()

    # 3. drp vs wavelength
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(wvl_c[keep], drp[keep], yerr=[drp_lo[keep], drp_hi[keep]],
                fmt="o", ms=5, color="C4", alpha=0.85,
                label="drp = R$_{morning}$ − R$_{evening}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("drp")
    ax.set_title(f"{title_tag}  radius difference (morning − evening)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = outdir / "drp_vs_wavelength.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["drp"] = p.resolve()

    # 4. rp_mean² (symmetric depth)
    rp_mean_sq = rp_mean ** 2 * 1e6
    rp_mean_lo_ppm = (2 * np.abs(rp_mean) * rp_mean_lo) * 1e6
    rp_mean_hi_ppm = (2 * np.abs(rp_mean) * rp_mean_hi) * 1e6
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(wvl_c[keep], rp_mean_sq[keep],
                yerr=[rp_mean_lo_ppm[keep], rp_mean_hi_ppm[keep]],
                fmt="o", ms=5, color="C1",
                label="catwoman rp$_{mean}^2$")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Transit depth (ppm)")
    ax.set_title(f"{title_tag}  symmetric depth (catwoman rp_mean$^2$)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = outdir / "symmetric_depth_vs_wavelength.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["symmetric"] = p.resolve()

    # Machine-readable text.
    p_txt = outdir / "limb_spectra.txt"
    with open(p_txt, "w") as f:
        f.write(f"# {title_tag}  limb-limb spectra\n")
        f.write("# Convention: rp1 = evening (leading), rp2 = morning (trailing); catwoman phi=90.\n")
        f.write("# Depths in ppm. err columns are (lower 1-sigma, upper 1-sigma).\n")
        f.write("# masked = 1 if bin was dropped (sanity / bad_wavelengths / NaN).\n")
        f.write(
            "wvl_low_um wvl_high_um wvl_center_um "
            "rp_mean rp_mean_lo rp_mean_hi "
            "drp drp_lo drp_hi "
            "rp1_sq_ppm rp1_sq_lo_ppm rp1_sq_hi_ppm "
            "rp2_sq_ppm rp2_sq_lo_ppm rp2_sq_hi_ppm "
            "delta_depth_ppm delta_err_ppm "
            "rms_residual_ppm masked\n"
        )
        for i in range(len(wvl_c)):
            f.write(
                f"{wvl_lo[i]:.4f} {wvl_hi[i]:.4f} {wvl_c[i]:.4f} "
                f"{rp_mean[i]:.6f} {rp_mean_lo[i]:.6f} {rp_mean_hi[i]:.6f} "
                f"{drp[i]:.6f} {drp_lo[i]:.6f} {drp_hi[i]:.6f} "
                f"{rp1_sq[i]:.2f} {rp1_lo[i]:.2f} {rp1_hi[i]:.2f} "
                f"{rp2_sq[i]:.2f} {rp2_lo[i]:.2f} {rp2_hi[i]:.2f} "
                f"{delta[i]:.2f} {delta_err[i]:.2f} "
                f"{rms[i]:.2f} {int(bad_mask[i])}\n"
            )
    paths["txt"] = p_txt.resolve()

    if avg_str:
        logger.info("limb_asymmetry: %s over %d kept of %d bins (%d masked)",
                    avg_str, int(keep.sum()), len(wvl_c), n_dropped)

    return paths


def save_limb_asymmetry(
    result: dict,
    outdir: str | Path,
) -> Path:
    """Persist the full per-bin result dict as a single .npz."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "limb_asymmetry.npz"
    np.savez(p, **{k: np.asarray(v) for k, v in result.items()})
    return p.resolve()
