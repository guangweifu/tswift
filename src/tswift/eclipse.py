"""Secondary-eclipse (emission) fitting — the occultation analog of the transit path.

For a secondary eclipse the planet passes BEHIND the star, so you measure the
planet's dayside flux (an emission/reflection spectrum), not transmission. The
differences from the transit path are fundamental:

- **No limb darkening.** You occult the (≈uniform) planet dayside disk, so the
  model uses ``batman`` ``transittype="secondary"`` with ``limb_dark="uniform"``.
- **Depth means flux ratio.** The eclipse depth is ``Fp/Fs`` (≈ ``fp``), not
  ``(Rp/Rs)²``. The per-channel depths ARE the emission spectrum — no squaring.
- **Timing is at superior conjunction.** We fit ``t_secondary`` (the eclipse
  mid-time), not the transit ``t0``.

Orbital geometry (``a``, ``inc``, ``rp``) is fixed from the known transit
parameters — the eclipse mainly constrains depth + timing. The white-light fit
solves ``[slope, fp, constant, t_sec_offset]``; each spectroscopic channel then
fixes geometry + timing and fits ``[slope, fp, constant]``.

batman convention (verified): with ``fp`` the planet/star flux ratio,
``light_curve`` returns ``1+fp`` out of eclipse and ``1`` in eclipse; the dip is
``fp/(1+fp)``. We normalize by ``1+fp`` so the model baseline is 1.
"""
from __future__ import annotations

import logging
from typing import Optional

import batman
import numpy as np
from scipy.optimize import curve_fit

from tswift.combine import bin_inverse_variance

logger = logging.getLogger(__name__)

ECLIPSE_PARAM_ORDER = ["slope", "fp", "constant", "t_sec_offset"]


def _make_model(time_hr, period_hr, geom, rp, ecc, omega, t_sec_off_hr):
    """Build a reusable batman secondary-eclipse TransitModel + params.

    ``geom`` = {"a", "inc"}; ``t_sec_off_hr`` is the eclipse mid-time in the same
    'hours from start' frame as ``time_hr``. Returns (model, params, time_phase).
    """
    time_phase = time_hr / period_hr
    p = batman.TransitParams()
    p.per = 1.0
    p.rp = rp
    p.a = geom["a"]
    p.inc = geom["inc"]
    p.ecc = ecc
    p.w = omega
    p.limb_dark = "uniform"
    p.u = []
    p.fp = 1.0
    p.t_secondary = t_sec_off_hr / period_hr
    p.t0 = p.t_secondary - 0.5          # transit time (circular); eclipse uses t_secondary
    m = batman.TransitModel(p, time_phase, transittype="secondary")
    return m, p, time_phase


def _eval(m, p, time_hr, slope, fp, constant, t_sec_off_hr, period_hr, t_ref):
    """OOT-normalized eclipse model (baseline 1, dip ≈ fp) + slope baseline."""
    p.fp = fp
    p.t_secondary = t_sec_off_hr / period_hr
    p.t0 = p.t_secondary - 0.5
    lc = m.light_curve(p)               # 1+fp OOT, 1 in eclipse
    lc_norm = lc / (1.0 + fp)           # baseline 1, dip fp/(1+fp)
    return constant * lc_norm + slope * (time_hr - t_ref)


# ---------------------------------------------------------------------------
# White-light eclipse fit (emcee)
# ---------------------------------------------------------------------------

def fit_eclipse_wl(
    time_hr: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    period_days: float,
    geom: dict,
    rp: float,
    ecc: float = 0.0,
    omega: float = 90.0,
    initial: dict,
    priors: dict,
    nwalkers: int = 32,
    nsteps: int = 6000,
    nburn: int = 2000,
    rng_seed: Optional[int] = None,
) -> dict:
    """White-light secondary-eclipse MCMC fit.

    Fits ``[slope, fp, constant, t_sec_offset]`` with orbital geometry
    (``a``, ``inc``, ``rp``) fixed. Saves the MAP (argmax log_prob), per
    pitfall #25.

    ``initial`` / ``priors`` are keyed by ``ECLIPSE_PARAM_ORDER``; ``priors``
    values are ``(lo, hi)`` uniform bounds. Returns a dict with ``best_params``
    (MAP), ``median_params``, ``best_errors``, ``samples``, ``log_prob``,
    ``best_fit_curve``, ``residuals``, ``rms_residual``, ``eclipse_depth_ppm``.
    """
    import emcee

    period_hr = period_days * 24.0
    t_ref = float(np.median(time_hr))
    order = ECLIPSE_PARAM_ORDER
    p0 = np.array([initial[k] for k in order], dtype=float)
    bounds = [priors[k] for k in order]
    m, p, _ = _make_model(time_hr, period_hr, geom, rp, ecc, omega, initial["t_sec_offset"])

    def log_prob(theta):
        for v, (lo, hi) in zip(theta, bounds):
            if not (lo < v < hi):
                return -np.inf
        slope, fp, constant, tsec = theta
        model = _eval(m, p, time_hr, slope, fp, constant, tsec, period_hr, t_ref)
        return float(-0.5 * np.sum((flux - model) ** 2 / flux_err ** 2))

    rng = np.random.default_rng(rng_seed)
    pos = p0 + 1e-6 * rng.standard_normal((nwalkers, len(order))) \
        + np.array([1e-5, 1e-5, 1e-5, 1e-3]) * rng.standard_normal((nwalkers, len(order)))
    sampler = emcee.EnsembleSampler(nwalkers, len(order), log_prob)
    sampler.run_mcmc(pos, nsteps, progress=False)

    flat = sampler.get_chain(discard=nburn, flat=True)
    logp = sampler.get_log_prob(discard=nburn, flat=True)
    best = flat[int(np.nanargmax(logp))].copy()      # MAP (pitfall #25)
    median = np.percentile(flat, 50, axis=0)
    lo16, hi84 = np.percentile(flat, [16, 84], axis=0)
    errors = np.stack([best - lo16, hi84 - best], axis=1)

    s, fp, c, tsec = best
    curve = _eval(m, p, time_hr, s, fp, c, tsec, period_hr, t_ref)
    resid = flux - curve
    fp_v = float(fp)
    return {
        "best_params": best, "median_params": median, "best_errors": errors,
        "param_order": order, "samples": flat, "log_prob": logp,
        "best_fit_curve": curve, "residuals": resid,
        "rms_residual": float(np.std(resid)),
        "eclipse_depth_ppm": fp_v / (1.0 + fp_v) * 1e6,
        "t_sec_offset": float(tsec),
    }


# ---------------------------------------------------------------------------
# Per-channel eclipse depth (emission spectrum)
# ---------------------------------------------------------------------------

def fit_eclipse_curves(
    clean_2D: np.ndarray,
    time_hr: np.ndarray,
    wvl: np.ndarray,
    *,
    wavelength_left: int,
    wavelength_right: int,
    period_days: float,
    geom: dict,
    rp: float,
    t_sec_offset: float,
    ecc: float = 0.0,
    omega: float = 90.0,
    oot_mask: Optional[np.ndarray] = None,
    fp_init: float = 0.002,
    bounds_lower: tuple = (-0.005, 0.0, 0.995),
    bounds_upper: tuple = (0.005, 0.02, 1.005),
) -> dict:
    """Fit the eclipse depth at every wavelength column (geometry + timing fixed).

    Returns dict with ``fit`` (n_wvl, 3) = [slope, fp, constant], ``fit_err``,
    ``depth_ppm`` / ``depth_err_ppm`` (eclipse depth Fp/Fs per channel — the
    emission spectrum), and the bandpass slice.
    """
    if clean_2D.ndim != 2:
        raise ValueError(f"expected clean_2D (n_frames, n_wvl), got {clean_2D.shape}")
    n_frames, n_wvl = clean_2D.shape
    right = min(wavelength_right, n_wvl)
    period_hr = period_days * 24.0
    t_ref = float(np.median(time_hr))
    m, p, _ = _make_model(time_hr, period_hr, geom, rp, ecc, omega, t_sec_offset)

    if oot_mask is not None:
        norm_mask = np.asarray(oot_mask, dtype=bool)
    else:
        n_edge = max(10, int(n_frames * 0.15))
        norm_mask = np.zeros(n_frames, bool)
        norm_mask[:n_edge] = True
        norm_mask[-n_edge:] = True
    oot_median = np.nanmedian(clean_2D[norm_mask, :], axis=0)[None, :]
    cl_nor = clean_2D / oot_median

    valid_frames = np.isfinite(cl_nor[:, (wavelength_left + right) // 2])
    x = time_hr[valid_frames]
    cl_fit = cl_nor[valid_frames]

    def model_lc(_t, slope, fp, constant):
        return _eval(m, p, x, slope, fp, constant, t_sec_offset, period_hr, t_ref)

    fit_arr = np.full((n_wvl, 3), np.nan)
    err_arr = np.full((n_wvl, 3), np.nan)
    for i in range(wavelength_left, right):
        lc = cl_fit[:, i]
        if not np.all(np.isfinite(lc)):
            lc = np.where(np.isfinite(lc), lc, np.nanmedian(lc))
        try:
            popt, pcov = curve_fit(model_lc, x, lc, p0=[0.0, fp_init, 1.0],
                                   bounds=(bounds_lower, bounds_upper), maxfev=10000)
            fit_arr[i] = popt
            err_arr[i] = np.sqrt(np.diag(pcov))
        except Exception as e:
            logger.debug(f"wvl {i} eclipse fit failed: {e}")

    fp = fit_arr[:, 1]
    fp_err = err_arr[:, 1]
    depth_ppm = fp / (1.0 + fp) * 1e6
    depth_err_ppm = fp_err / (1.0 + fp) ** 2 * 1e6
    n_good = int(np.sum(np.isfinite(fp)))
    logger.info(f"Eclipse: fit {n_good}/{right - wavelength_left} channels")
    return {
        "fit": fit_arr, "fit_err": err_arr,
        "depth_ppm": depth_ppm, "depth_err_ppm": depth_err_ppm,
        "wavelength_left": wavelength_left, "wavelength_right": right,
    }


# ---------------------------------------------------------------------------
# Combine -> emission spectrum
# ---------------------------------------------------------------------------

def combine_emission(
    wvl: np.ndarray,
    depth_ppm: np.ndarray,
    depth_err_ppm: np.ndarray,
    *,
    bin_widths_nm=(10, 20, 50),
    bad_wavelengths_um: Optional[list] = None,
    wavelength_range_um: Optional[tuple] = None,
) -> dict:
    """Bin the per-channel eclipse depths into an emission spectrum.

    Unlike ``combine_spectrum`` (transmission) there is NO ``(Rp/Rs)²``
    conversion — the eclipse depth IS the measured quantity (Fp/Fs in ppm).
    """
    wvl = np.asarray(wvl, float)
    d = np.asarray(depth_ppm, float)
    e = np.asarray(depth_err_ppm, float)
    good = np.isfinite(d) & np.isfinite(e) & (e > 0)
    if bad_wavelengths_um:
        bad = np.asarray(bad_wavelengths_um, float)
        good &= ~np.any(np.abs(wvl[:, None] - bad[None, :]) < 0.001, axis=1)
    wg, dg, eg = wvl[good], d[good], e[good]
    if wavelength_range_um is not None:
        lo, hi = wavelength_range_um
    else:
        lo = float(np.nanmin(wg)) if wg.size else 0.0
        hi = float(np.nanmax(wg)) if wg.size else 1.0

    binned = {}
    for width_nm in bin_widths_nm:
        w_um = width_nm / 1000.0
        n_bins = int(np.ceil((hi - lo) / w_um))
        if n_bins < 1:
            continue
        edges = lo + w_um * np.arange(n_bins + 1)
        if edges[-1] <= hi:
            edges[-1] = np.nextafter(hi, hi + 1.0)
        wc, dep, err = bin_inverse_variance(wg, dg, eg, edges)
        binned[f"{int(width_nm)}nm"] = {"wvl_um": wc, "depth_ppm": dep, "depth_err_ppm": err}
        logger.info(f"Emission bin {width_nm} nm: {len(wc)} bins "
                    f"(mean depth {np.mean(dep):.0f} ± {np.mean(err):.0f} ppm)")
    return {
        "native": {"wvl_um": wvl, "depth_ppm": d, "depth_err_ppm": e, "mask": good},
        "binned": binned, "kind": "emission",
    }


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_eclipse_wl(result: dict, time_hr, flux, outdir, *, detector: str = "") -> "Path":
    import matplotlib.pyplot as plt
    from pathlib import Path
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    curve, resid = result["best_fit_curve"], result["residuals"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6),
                           gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax[0].plot(time_hr, flux, ".", ms=2, color="0.6", alpha=0.5, label="data")
    bw = max(1, len(time_hr) // 120); nb = len(time_hr) // bw
    tb = time_hr[:nb * bw].reshape(-1, bw).mean(1); fb = flux[:nb * bw].reshape(-1, bw).mean(1)
    ax[0].plot(tb, fb, "o", ms=4, color="tab:blue", label=f"binned ×{bw}")
    ax[0].plot(time_hr, curve, "-", color="tab:red", lw=1.5, label="eclipse model")
    ax[0].set_ylabel("normalized flux")
    ax[0].set_title(f"White-light eclipse {detector}   depth={result['eclipse_depth_ppm']:.0f} ppm   "
                    f"RMS={result['rms_residual']*1e6:.0f} ppm")
    ax[0].legend(fontsize=8)
    ax[1].plot(time_hr, resid * 1e6, ".", ms=2, color="0.6", alpha=0.5)
    rb = resid[:nb * bw].reshape(-1, bw).mean(1)
    ax[1].plot(tb, rb * 1e6, "o", ms=4, color="tab:blue")
    ax[1].axhline(0, color="tab:red", ls="--"); ax[1].set_ylabel("resid (ppm)")
    ax[1].set_xlabel("time (hours)")
    plt.tight_layout(); p = outdir / "eclipse_wl.png"; fig.savefig(p, dpi=120); plt.close(fig)
    return p


def plot_emission_spectrum(combined: dict, outdir, *, planet_name: str = "", detector: str = "") -> "Path":
    import matplotlib.pyplot as plt
    from pathlib import Path
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    nat = combined["native"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    m = nat["mask"]
    ax.plot(nat["wvl_um"][m], nat["depth_ppm"][m], ".", ms=2, color="0.7", alpha=0.4, label="native")
    for w, col in zip(["10nm", "20nm", "50nm"], ["tab:green", "tab:orange", "tab:red"]):
        if w in combined["binned"]:
            b = combined["binned"][w]
            ax.errorbar(b["wvl_um"], b["depth_ppm"], yerr=b["depth_err_ppm"],
                        fmt="o", ms=3, capsize=1, label=w, color=col, alpha=0.8)
    ax.set_xlabel("wavelength (µm)"); ax.set_ylabel("eclipse depth Fp/Fs (ppm)")
    ax.set_title(f"{planet_name} emission spectrum {detector}".strip())
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    # Robust y-limits from the binned points (+ error bars), so a single noisy
    # native channel can't stretch the axis and hide the spectrum.
    lo_hi = []
    for b in combined["binned"].values():
        d, e = np.asarray(b["depth_ppm"], float), np.asarray(b["depth_err_ppm"], float)
        if d.size:
            lo_hi += [float(np.nanmin(d - e)), float(np.nanmax(d + e))]
    if not lo_hi:
        dn = nat["depth_ppm"][m]
        lo_hi = [float(np.nanpercentile(dn, 1)), float(np.nanpercentile(dn, 99))]
    lo, hi = min(lo_hi), max(lo_hi)
    pad = 0.15 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - pad, hi + pad)
    plt.tight_layout(); p = outdir / "emission_spectrum.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def save_emission_spectrum(combined: dict, outdir, *, planet_name: Optional[str] = None) -> dict:
    """Write native + binned emission spectra to txt + a summary JSON + figure."""
    import json
    from pathlib import Path
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    nat = combined["native"]; m = nat["mask"]
    out = {}
    np.savetxt(outdir / "emission_native.txt",
               np.column_stack([nat["wvl_um"][m], nat["depth_ppm"][m], nat["depth_err_ppm"][m]]),
               header="wvl_um  eclipse_depth_ppm  depth_err_ppm")
    out["native"] = outdir / "emission_native.txt"
    for w, b in combined["binned"].items():
        f = outdir / f"emission_{w}.txt"
        np.savetxt(f, np.column_stack([b["wvl_um"], b["depth_ppm"], b["depth_err_ppm"]]),
                   header="wvl_um  eclipse_depth_ppm  depth_err_ppm")
        out[w] = f
    json.dump({"planet": planet_name, "kind": "emission",
               "n_native": int(m.sum()),
               "bins": {w: len(b["wvl_um"]) for w, b in combined["binned"].items()}},
              open(outdir / "emission_summary.json", "w"), indent=2)
    plot_emission_spectrum(combined, outdir, planet_name=planet_name or "")
    return out
