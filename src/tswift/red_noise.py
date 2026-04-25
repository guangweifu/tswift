"""Red-noise diagnostics for white-light and spectroscopic fit residuals.

Standard transit-fit pipelines treat residual scatter as white noise.  In
practice, JWST detector systematics (NIRISS 1/f, NIRSpec readout patterns,
imperfect ramp settling) and astrophysical contaminants (LD mismatch,
unmodelled stellar variability) all show up as **correlated** residuals
that scale slower than 1/√N when binned.  At 30–60 min binning — exactly
the cadence atmospheric retrievals care about — the difference between
white and red can be a factor of 2–3 in error bars.

This module computes the canonical Pont, Zucker & Queloz (2006) battery
of diagnostics:

- **σ-vs-bin-size** (Allan-style) curve with the white-noise expectation
  overlaid.
- **β factor** = σ_obs(bin) / [σ(1)/√N] at fiducial bin sizes.
- **Auto-correlation function** of residuals with the 95 % white-noise
  band marked.
- **Lomb-Scargle PSD** with a fitted log-log slope α.
- **Q-Q plot** vs Gaussian (catches kurtosis-driven outliers).
- **Time-domain residuals + sliding RMS** (catches localised excess).

Two flavours
============

`compute_wl_red_noise` + `plot_wl_red_noise`
    Single light curve.  Use on the WL fit residuals (one panel
    per detector).

`compute_spec_red_noise` + `plot_spec_red_noise`
    Per-channel light curves.  Re-builds residuals from the saved
    `spec_fit.npy`/`spec_fit_err.npy` + saved `clean_2D.npy`, computes
    the same six metrics per channel, and plots them as a function of
    wavelength.  Important because red noise is wavelength-dependent
    (NIRISS 1/f is brightness-driven; NIRSpec NRS2 is photon-noise
    limited at red wavelengths).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default fiducial bin sizes (minutes) for β.  30 min ≈ ingress duration
# for typical hot Jupiters and is the conventional headline number.
DEFAULT_BETA_BINS_MIN = (5.0, 15.0, 30.0, 60.0)


# ---------------------------------------------------------------------------
# Core stats kernels (no plotting)
# ---------------------------------------------------------------------------

def allan_curve(
    residuals: np.ndarray, *, max_bin: Optional[int] = None,
    n_points: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin residuals at a log-spaced grid of N and return σ(N).

    Returns ``(bin_sizes, sigma_obs, sigma_white_expectation)`` where
    ``sigma_white_expectation = sigma_obs[0] / sqrt(N)``.
    """
    n = len(residuals)
    if max_bin is None:
        max_bin = max(2, n // 4)
    bins = np.unique(np.geomspace(1, max_bin, num=n_points).astype(int))
    bins = bins[bins >= 1]
    sigma_obs = np.zeros(len(bins))
    for i, b in enumerate(bins):
        nuse = (n // b) * b
        if nuse < b:
            sigma_obs[i] = np.nan
            continue
        binned = residuals[:nuse].reshape(-1, b).mean(axis=1)
        sigma_obs[i] = float(np.nanstd(binned))
    sigma_1 = sigma_obs[0]
    sigma_white = sigma_1 / np.sqrt(bins)
    return bins, sigma_obs, sigma_white


def beta_factor(
    residuals: np.ndarray, dt_min: float, bin_minutes: float,
) -> float:
    """Pont/Zucker/Queloz 2006 β = σ_obs(bin) / (σ_1 / √N).

    `dt_min` is the integration cadence in minutes.
    """
    n = len(residuals)
    bin_n = max(1, int(round(bin_minutes / dt_min)))
    nuse = (n // bin_n) * bin_n
    if nuse < bin_n:
        return float("nan")
    binned = residuals[:nuse].reshape(-1, bin_n).mean(axis=1)
    sigma_obs = float(np.nanstd(binned))
    sigma_white = float(np.nanstd(residuals)) / np.sqrt(bin_n)
    if sigma_white == 0:
        return float("nan")
    return sigma_obs / sigma_white


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized ACF up to ``max_lag`` (lag 0 = 1.0)."""
    x = x - np.nanmean(x)
    n = len(x)
    out = np.zeros(max_lag + 1)
    var = np.nanvar(x)
    if var == 0:
        out[0] = 1.0
        return out
    for k in range(max_lag + 1):
        if k == 0:
            out[k] = 1.0
        else:
            out[k] = float(np.nanmean(x[:-k] * x[k:])) / var
    return out


def lombscargle_psd(
    time_hr: np.ndarray, residuals: np.ndarray, n_freq: int = 500,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lomb-Scargle periodogram + linear log-log slope of the spectrum.

    Returns ``(freqs_per_hour, psd, slope_alpha)``.
    """
    from scipy.signal import lombscargle as _ls

    t_sec = np.asarray(time_hr) * 3600.0
    f_min = 1.0 / max(1e-9, t_sec[-1] - t_sec[0])
    f_nyq = 1.0 / max(1e-9, 2 * float(np.median(np.diff(t_sec))))
    freqs = np.geomspace(f_min, f_nyq, n_freq)
    angular = 2 * np.pi * freqs
    psd = _ls(t_sec, residuals - np.nanmean(residuals), angular, normalize=False)
    log_f = np.log10(freqs)
    log_p = np.log10(np.maximum(psd, 1e-30))
    slope = float(np.polyfit(log_f, log_p, 1)[0])
    return freqs * 3600.0, psd, slope     # freqs in 1/hour


# ---------------------------------------------------------------------------
# WL red-noise diagnostic
# ---------------------------------------------------------------------------

@dataclass
class WLRedNoiseResult:
    """All metrics + raw arrays needed to remake the diagnostic panel."""
    detector: str
    n_points: int
    cadence_min: float
    duration_hr: float
    rms_ppm: float
    oot_scatter_ppm: Optional[float]
    bins_min: np.ndarray = field(repr=False)
    sigma_obs_ppm: np.ndarray = field(repr=False)
    sigma_white_ppm: np.ndarray = field(repr=False)
    beta_factors: dict = field(default_factory=dict)
    acf_lags: np.ndarray = field(repr=False, default=None)
    acf_values: np.ndarray = field(repr=False, default=None)
    acf_ci95: float = 0.0
    psd_freqs_per_hr: np.ndarray = field(repr=False, default=None)
    psd_values: np.ndarray = field(repr=False, default=None)
    psd_slope_alpha: float = 0.0
    skew: float = 0.0
    excess_kurtosis: float = 0.0
    sliding_rms_ppm: np.ndarray = field(repr=False, default=None)
    sliding_window: int = 0
    time_hr: np.ndarray = field(repr=False, default=None)
    residuals_ppm: np.ndarray = field(repr=False, default=None)


def compute_wl_red_noise(
    time_hr: np.ndarray,
    residuals: np.ndarray,
    *,
    detector: str = "",
    oot_scatter_ppm: Optional[float] = None,
    beta_bins_min: tuple[float, ...] = DEFAULT_BETA_BINS_MIN,
) -> WLRedNoiseResult:
    """Compute the full diagnostic suite on a single residual time series.

    Parameters
    ----------
    time_hr : (N,)
        Time stamps in hours (need not be uniform; the LS PSD handles it).
    residuals : (N,)
        Fit residuals as a fraction (NOT ppm).  Will be converted to ppm
        internally for human-readable plots.
    detector : str
        Label for the panel title.
    oot_scatter_ppm : float, optional
        Out-of-transit scatter from the WL fit, plotted as reference.
    beta_bins_min : tuple of float
        Bin sizes (in minutes) at which to report β.

    Returns
    -------
    WLRedNoiseResult
    """
    from scipy.stats import skew as _skew, kurtosis as _kurt

    res_ppm = np.asarray(residuals) * 1e6
    n = len(res_ppm)
    dt_min = float(np.nanmedian(np.diff(time_hr))) * 60.0

    bins, sigma_obs, sigma_white = allan_curve(res_ppm)
    bins_min = bins * dt_min

    betas = {f"{int(b)}min": float(beta_factor(res_ppm, dt_min, b))
             for b in beta_bins_min}

    max_lag = max(20, n // 40)
    acf = autocorrelation(res_ppm, max_lag)
    ci95 = 1.96 / float(np.sqrt(n))

    freqs_per_hr, psd, slope = lombscargle_psd(time_hr, res_ppm)

    win = max(20, n // 50)
    sliding_rms = np.array([
        float(np.nanstd(res_ppm[max(0, i - win // 2):
                                i + win // 2 + 1]))
        for i in range(n)
    ])

    return WLRedNoiseResult(
        detector=detector,
        n_points=n,
        cadence_min=dt_min,
        duration_hr=float(time_hr[-1] - time_hr[0]),
        rms_ppm=float(np.nanstd(res_ppm)),
        oot_scatter_ppm=oot_scatter_ppm,
        bins_min=bins_min,
        sigma_obs_ppm=sigma_obs,
        sigma_white_ppm=sigma_white,
        beta_factors=betas,
        acf_lags=np.arange(max_lag + 1),
        acf_values=acf,
        acf_ci95=ci95,
        psd_freqs_per_hr=freqs_per_hr,
        psd_values=psd,
        psd_slope_alpha=slope,
        skew=float(_skew(res_ppm)),
        excess_kurtosis=float(_kurt(res_ppm, fisher=True)),
        sliding_rms_ppm=sliding_rms,
        sliding_window=win,
        time_hr=np.asarray(time_hr),
        residuals_ppm=res_ppm,
    )


def plot_wl_red_noise(
    result: WLRedNoiseResult,
    outdir: str | Path,
    *,
    planet_name: str = "",
) -> dict:
    """Write the standard 6-panel red_noise.png + summary JSON.

    Returns ``dict[name → Path]``.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import probplot

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    title_tag = planet_name + (f" — {result.detector.upper()}"
                               if result.detector else "")

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle(
        f"{title_tag}  red-noise diagnostics  "
        f"(N={result.n_points}, cadence={result.cadence_min:.2f} min, "
        f"RMS={result.rms_ppm:.0f} ppm)",
        y=1.00,
    )

    # (1) σ vs bin size.
    ax = axes[0, 0]
    ax.loglog(result.bins_min, result.sigma_obs_ppm, "o-", color="C0",
              label="observed σ(bin)")
    ax.loglog(result.bins_min, result.sigma_white_ppm, "--", color="black",
              label="white noise (1/√N)")
    ax.set_xlabel("Bin size (min)")
    ax.set_ylabel("Residual σ (ppm)")
    ax.set_title("σ vs bin size — red noise = above the dashed line")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # (2) Stats panel.
    ax = axes[0, 1]
    ax.axis("off")
    text = (
        "Pont 2006 β factor\n  σ_obs(bin) / [σ(1)/√N]\n  β > 1 ⇒ red noise\n\n"
    )
    for label, val in result.beta_factors.items():
        text += f"  β @ {label:>6s}:   {val:5.2f}\n"
    text += f"\nResidual statistics:\n"
    text += f"  median:        {float(np.median(result.residuals_ppm)):+6.1f} ppm\n"
    text += f"  std:           {result.rms_ppm:6.1f} ppm\n"
    if result.oot_scatter_ppm is not None:
        text += f"  oot_scatter:   {result.oot_scatter_ppm:6.1f} ppm\n"
    text += f"  skew:          {result.skew:+5.2f}\n"
    text += f"  excess kurt.:  {result.excess_kurtosis:+5.2f}\n"
    text += f"\nPSD slope (log-log):\n"
    text += f"  α = {result.psd_slope_alpha:+.2f}  (white = 0; red = -1)\n"
    text += f"\nACF lag 1:\n"
    text += (f"  r₁ = {result.acf_values[1]:+.3f}  "
             f"(95 % CI ±{result.acf_ci95:.3f})\n")
    ax.text(0.02, 0.98, text, family="monospace", fontsize=10,
            va="top", transform=ax.transAxes)

    # (3) ACF.
    ax = axes[1, 0]
    ax.stem(result.acf_lags, result.acf_values, basefmt=" ",
            linefmt="C0-", markerfmt="C0o")
    ax.axhspan(-result.acf_ci95, result.acf_ci95, color="gray", alpha=0.2,
               label="95 % CI for white noise")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Lag (integrations)")
    ax.set_ylabel("ACF")
    ax.set_title("Auto-correlation — significant lags = correlated noise")
    ax.legend()
    ax.grid(alpha=0.3)

    # (4) PSD.
    ax = axes[1, 1]
    ax.loglog(result.psd_freqs_per_hr, result.psd_values,
              color="C0", lw=0.8, alpha=0.7)
    log_f = np.log10(result.psd_freqs_per_hr / 3600.0)
    intercept = float(np.polyfit(log_f, np.log10(np.maximum(result.psd_values, 1e-30)), 1)[1])
    fit_p = 10 ** (intercept + result.psd_slope_alpha * log_f)
    ax.loglog(result.psd_freqs_per_hr, fit_p, "--", color="tab:red",
              label=f"slope α = {result.psd_slope_alpha:+.2f}")
    ax.axhline(np.nanmedian(result.psd_values), color="black", ls=":", alpha=0.5,
               label="median (white-noise reference)")
    ax.set_xlabel("Frequency (1/hour)")
    ax.set_ylabel("Lomb-Scargle PSD")
    ax.set_title("PSD — flat = white, sloped = 1/f red")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # (5) Q-Q.
    ax = axes[2, 0]
    probplot(result.residuals_ppm, dist="norm", plot=ax)
    ax.set_title("Q-Q vs Gaussian — straight line = Gaussian noise")
    ax.grid(alpha=0.3)

    # (6) Time-domain + sliding RMS.
    ax = axes[2, 1]
    ax.scatter(result.time_hr, result.residuals_ppm, s=3, alpha=0.3, color="gray")
    ax2 = ax.twinx()
    ax2.plot(result.time_hr, result.sliding_rms_ppm, color="tab:red", lw=1.2,
             label=f"sliding RMS (window {result.sliding_window})")
    ax2.set_ylabel("Sliding RMS (ppm)", color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Residual (ppm)")
    ax.set_title("Time-domain residuals + sliding RMS")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p_png = outdir / "red_noise.png"
    fig.savefig(p_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["red_noise_png"] = p_png.resolve()

    # Compact summary JSON.
    summary = {
        "detector": result.detector,
        "n_points": result.n_points,
        "cadence_min": result.cadence_min,
        "duration_hr": result.duration_hr,
        "rms_ppm": result.rms_ppm,
        "oot_scatter_ppm": result.oot_scatter_ppm,
        "beta_factors": result.beta_factors,
        "psd_slope_alpha": result.psd_slope_alpha,
        "acf_lag1": float(result.acf_values[1]),
        "skew": result.skew,
        "excess_kurtosis": result.excess_kurtosis,
    }
    p_json = outdir / "red_noise_summary.json"
    with open(p_json, "w") as f:
        json.dump(summary, f, indent=2)
    paths["red_noise_summary"] = p_json.resolve()

    return paths


# ---------------------------------------------------------------------------
# Spectroscopic red-noise diagnostic
# ---------------------------------------------------------------------------

@dataclass
class SpecRedNoiseResult:
    """Per-channel red-noise metrics across the full wavelength range."""
    detector: str
    wvl_um: np.ndarray = field(repr=False)
    n_channels: int = 0
    n_points: int = 0
    cadence_min: float = 0.0
    rms_ppm: np.ndarray = field(repr=False, default=None)
    beta_factors: dict[str, np.ndarray] = field(default_factory=dict)
    acf_lag1: np.ndarray = field(repr=False, default=None)
    psd_slope_alpha: np.ndarray = field(repr=False, default=None)
    skew: np.ndarray = field(repr=False, default=None)
    excess_kurtosis: np.ndarray = field(repr=False, default=None)
    valid_mask: np.ndarray = field(repr=False, default=None)


def compute_spec_red_noise(
    clean_2D: np.ndarray,
    time_hr: np.ndarray,
    wvl: np.ndarray,
    spec_fit: np.ndarray,
    *,
    geom: dict,
    period_days: float,
    u1_per_wvl: np.ndarray,
    u2_per_wvl: np.ndarray,
    oot_mask: np.ndarray,
    ecc: float = 0.0,
    omega: float = 90.0,
    fit_ld2: bool = True,
    detector: str = "",
    beta_bins_min: tuple[float, ...] = DEFAULT_BETA_BINS_MIN,
    channel_subsample: int = 1,
) -> SpecRedNoiseResult:
    """Per-channel red-noise diagnostics.

    Reconstructs each channel's residual time series from
    ``clean_2D[:, col]`` and the saved per-channel fit
    ``(slope, rp, [LD2,] constant)``, then runs the same metrics as
    `compute_wl_red_noise` per channel.

    Parameters
    ----------
    clean_2D : (n_frames, n_cols)
    time_hr  : (n_frames,)
    wvl      : (n_cols,)  µm
    spec_fit : (n_cols, 4)  per-channel fit results from
        `fit_spec_curves`.  Column order is **[slope, rp, constant, LD2]**
        (matches the legacy file convention; LD2 is in column 3 even
        when `fix_ld2=True` — in which case it equals the per-channel
        u2 from exotic_ld and the column is read as a fixed value).
    geom : dict — fixed orbital geometry {"a", "inc", "t0_offset"}
    period_days : float
    u1_per_wvl, u2_per_wvl : (n_cols,)  limb-darkening per channel
    oot_mask : (n_frames,) bool — used only for normalization
    fit_ld2 : bool — must match what `fit_spec_curves` used
    channel_subsample : int — compute every Nth channel (default 1 = all)

    Returns
    -------
    SpecRedNoiseResult
    """
    import batman
    from scipy.stats import skew as _skew, kurtosis as _kurt

    # Match `fit_spec_curves` exactly: model time in DAYS, normalize each
    # channel by edge-OOT median (first + last 15 % of frames), and use
    # `slope * (time - time[0])` baseline anchored at the visit start.
    n_frames, n_cols = clean_2D.shape
    time_days = np.asarray(time_hr) / 24.0
    t0_days = geom["t0_offset"] / 24.0
    dt_min = float(np.nanmedian(np.diff(time_hr))) * 60.0
    cols_iter = list(range(0, n_cols, channel_subsample))
    n_channels = len(cols_iter)

    # Per-channel OOT median for normalization — replicate
    # `fit_spec_curves`' edge-only OOT mask (first 15 % + last 15 %)
    # because that's what the saved spec_fit was fitted against.
    n_edge = max(10, int(n_frames * 0.15))
    fit_oot_mask = np.zeros(n_frames, bool)
    fit_oot_mask[:n_edge] = True
    fit_oot_mask[-n_edge:] = True
    with np.errstate(invalid="ignore"):
        oot_median_per_chan = np.nanmedian(clean_2D[fit_oot_mask, :], axis=0)
    # Channels with zero / NaN OOT median are off-trace edges; their
    # `cl_nor` will be NaN/inf and the loop skips them via the
    # ``valid.sum() < 100`` guard below.
    with np.errstate(divide="ignore", invalid="ignore"):
        cl_nor = clean_2D / oot_median_per_chan[None, :]

    # Build a single geometry-anchored TransitModel reused per channel.
    p0 = batman.TransitParams()
    p0.t0 = t0_days; p0.per = period_days; p0.rp = 0.1
    p0.a = geom["a"]; p0.inc = geom["inc"]
    p0.ecc = ecc; p0.w = omega
    p0.u = [0.3, 0.3]; p0.limb_dark = "quadratic"
    m_geom = batman.TransitModel(p0, time_days)

    rms_ppm = np.full(n_cols, np.nan)
    beta_arrs = {f"{int(b)}min": np.full(n_cols, np.nan)
                 for b in beta_bins_min}
    acf1_arr = np.full(n_cols, np.nan)
    psd_alpha_arr = np.full(n_cols, np.nan)
    skew_arr = np.full(n_cols, np.nan)
    kurt_arr = np.full(n_cols, np.nan)
    valid_mask = np.zeros(n_cols, dtype=bool)

    logger.info(
        "spec_red_noise: %d channels, every-%d sampling, cadence=%.2f min",
        n_channels, channel_subsample, dt_min,
    )

    for col in cols_iter:
        rp = float(spec_fit[col, 1])
        if not np.isfinite(rp) or rp <= 0:
            continue
        u1 = float(u1_per_wvl[col]) if np.isfinite(u1_per_wvl[col]) else 0.3
        # Column order from fit_spec_curves: [slope, rp, constant, LD2].
        slope = float(spec_fit[col, 0])
        const = float(spec_fit[col, 2])
        ld2 = float(spec_fit[col, 3])
        if not np.isfinite(ld2) and not fit_ld2:
            ld2 = float(u2_per_wvl[col]) if np.isfinite(u2_per_wvl[col]) else 0.1

        # Forward model — IDENTICAL to `fit_spec_curves._model`:
        #   m.light_curve(p) * constant + slope * (time_hr - time_hr[0])
        # Critical: NO OOT-median renormalization of the batman flux
        # (it's already ~1.0 at OOT by construction).  Use slope baseline
        # anchored at time_hr[0], matching the fit.
        p = batman.TransitParams()
        p.t0 = t0_days; p.per = period_days
        p.rp = rp; p.a = geom["a"]; p.inc = geom["inc"]
        p.ecc = ecc; p.w = omega; p.u = [u1, ld2]; p.limb_dark = "quadratic"
        flux = m_geom.light_curve(p)
        model = flux * const + slope * (np.asarray(time_hr) - time_hr[0])

        # Data: edge-OOT-normalized cl_nor — matches the fit input.
        channel_norm = cl_nor[:, col]
        residual = channel_norm - model
        valid = np.isfinite(residual)
        if int(valid.sum()) < 100:
            continue
        residual = residual[valid]
        t_use = np.asarray(time_hr)[valid]

        res_ppm = residual * 1e6
        rms_ppm[col] = float(np.nanstd(res_ppm))
        for b in beta_bins_min:
            key = f"{int(b)}min"
            beta_arrs[key][col] = beta_factor(res_ppm, dt_min, b)
        acf = autocorrelation(res_ppm, max_lag=2)
        acf1_arr[col] = float(acf[1])
        try:
            _, _, slope_psd = lombscargle_psd(t_use, res_ppm, n_freq=200)
            psd_alpha_arr[col] = slope_psd
        except Exception:
            pass
        skew_arr[col] = float(_skew(res_ppm))
        kurt_arr[col] = float(_kurt(res_ppm, fisher=True))
        valid_mask[col] = True

    return SpecRedNoiseResult(
        detector=detector,
        wvl_um=np.asarray(wvl),
        n_channels=int(valid_mask.sum()),
        n_points=n_frames,
        cadence_min=dt_min,
        rms_ppm=rms_ppm,
        beta_factors=beta_arrs,
        acf_lag1=acf1_arr,
        psd_slope_alpha=psd_alpha_arr,
        skew=skew_arr,
        excess_kurtosis=kurt_arr,
        valid_mask=valid_mask,
    )


def plot_spec_red_noise(
    result: SpecRedNoiseResult,
    outdir: str | Path,
    *,
    planet_name: str = "",
    rolling_smooth: int = 21,
) -> dict:
    """Wavelength-dependent red-noise panels.

    Five rows, all sharing the same wavelength axis:

      1. RMS(λ)         — per-channel residual scatter (smoothed median)
      2. β @ 30 min(λ)  — primary red-noise indicator
      3. ACF lag 1(λ)
      4. PSD slope α(λ)
      5. excess kurtosis(λ)

    A fitted constant for each wavelength-dependent metric and the
    "channels worse than k×median" outlier marks are overlaid so a
    glance tells you whether a specific wavelength range is
    contaminated.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    wvl_nm = result.wvl_um * 1000.0
    valid = result.valid_mask
    title_tag = planet_name + (f" — {result.detector.upper()}"
                               if result.detector else "")

    def smooth(arr: np.ndarray) -> np.ndarray:
        s = pd.Series(arr).rolling(rolling_smooth, center=True,
                                   min_periods=max(3, rolling_smooth // 4))
        return s.median().to_numpy()

    rms_smooth = smooth(np.where(valid, result.rms_ppm, np.nan))
    beta30 = result.beta_factors.get("30min", np.full_like(wvl_nm, np.nan))
    beta30_smooth = smooth(np.where(valid, beta30, np.nan))
    acf1_smooth = smooth(np.where(valid, result.acf_lag1, np.nan))
    psd_smooth = smooth(np.where(valid, result.psd_slope_alpha, np.nan))
    kurt_smooth = smooth(np.where(valid, result.excess_kurtosis, np.nan))

    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    fig.suptitle(
        f"{title_tag}  spectroscopic red-noise diagnostics  "
        f"(N_ch={result.n_channels}, cadence={result.cadence_min:.2f} min)",
        y=1.00,
    )

    # 1. RMS
    ax = axes[0]
    ax.scatter(wvl_nm[valid], result.rms_ppm[valid], s=3, alpha=0.35, color="gray")
    ax.plot(wvl_nm, rms_smooth, color="C0", lw=1.5,
            label=f"rolling median ({rolling_smooth})")
    ax.set_ylabel("Residual RMS (ppm)")
    ax.set_title("Per-channel RMS — photon-noise floor + systematics")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # 2. β @ 30 min
    ax = axes[1]
    ax.scatter(wvl_nm[valid], beta30[valid], s=3, alpha=0.35, color="gray")
    ax.plot(wvl_nm, beta30_smooth, color="C3", lw=1.5,
            label=f"rolling median ({rolling_smooth})")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="white noise (β=1)")
    ax.axhline(2.0, color="tab:red", ls=":", lw=0.6, alpha=0.6,
               label="2× red-noise threshold")
    ax.set_ylabel("β @ 30 min")
    ax.set_title("β factor (Pont 2006) — values > 1 ⇒ red noise")
    ax.set_ylim(0, max(3.5, np.nanmax(beta30) * 1.05) if np.any(valid) else 3.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # 3. ACF lag 1
    ax = axes[2]
    ax.scatter(wvl_nm[valid], result.acf_lag1[valid], s=3, alpha=0.35, color="gray")
    ax.plot(wvl_nm, acf1_smooth, color="C2", lw=1.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-1.96 / np.sqrt(result.n_points),
               +1.96 / np.sqrt(result.n_points),
               color="gray", alpha=0.2, label="95 % CI for white noise")
    ax.set_ylabel("ACF lag 1")
    ax.set_title("ACF lag-1 — short-timescale correlation per channel")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # 4. PSD slope
    ax = axes[3]
    ax.scatter(wvl_nm[valid], result.psd_slope_alpha[valid], s=3,
               alpha=0.35, color="gray")
    ax.plot(wvl_nm, psd_smooth, color="C4", lw=1.5)
    ax.axhline(0, color="k", lw=0.8, ls="--", label="white-noise (α=0)")
    ax.set_ylabel("PSD slope α")
    ax.set_title("PSD slope — α=0 is white, α<0 is 1/|f|^|α| red")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # 5. Kurtosis.
    ax = axes[4]
    ax.scatter(wvl_nm[valid], result.excess_kurtosis[valid], s=3,
               alpha=0.35, color="gray")
    ax.plot(wvl_nm, kurt_smooth, color="C5", lw=1.5)
    ax.axhline(0, color="k", lw=0.8, ls="--", label="Gaussian (kurt=0)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Excess kurtosis")
    ax.set_title("Excess kurtosis — > 0 ⇒ heavy-tailed (outliers)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    p_png = outdir / "spec_red_noise.png"
    fig.savefig(p_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["spec_red_noise_png"] = p_png.resolve()

    # Save the per-channel arrays as npz for downstream use.
    p_npz = outdir / "spec_red_noise.npz"
    np.savez(
        p_npz,
        wvl_um=result.wvl_um, valid_mask=result.valid_mask,
        rms_ppm=result.rms_ppm,
        beta_5min=result.beta_factors.get("5min"),
        beta_15min=result.beta_factors.get("15min"),
        beta_30min=result.beta_factors.get("30min"),
        beta_60min=result.beta_factors.get("60min"),
        acf_lag1=result.acf_lag1,
        psd_slope_alpha=result.psd_slope_alpha,
        skew=result.skew, excess_kurtosis=result.excess_kurtosis,
    )
    paths["spec_red_noise_npz"] = p_npz.resolve()

    # Compact summary JSON — ONE-NUMBER aggregates for the wavelength-
    # averaged behaviour.
    def _summary(arr):
        v = arr[result.valid_mask]
        return {
            "median": float(np.nanmedian(v)),
            "mad": float(np.nanmedian(np.abs(v - np.nanmedian(v))) * 1.4826),
            "frac_above_1": float(np.mean(v > 1.0)) if v.size else 0.0,
        }
    summary = {
        "detector": result.detector,
        "n_channels_valid": int(result.n_channels),
        "n_points": result.n_points,
        "cadence_min": result.cadence_min,
        "rms_ppm_median": float(np.nanmedian(result.rms_ppm[result.valid_mask])),
        "beta_30min_summary": _summary(beta30),
        "acf_lag1_summary": _summary(result.acf_lag1),
        "psd_slope_summary": _summary(result.psd_slope_alpha),
    }
    p_json = outdir / "spec_red_noise_summary.json"
    with open(p_json, "w") as f:
        json.dump(summary, f, indent=2)
    paths["spec_red_noise_summary"] = p_json.resolve()

    logger.info(
        "spec_red_noise: median β @ 30 min = %.2f, %.0f%% of channels > 1.0; "
        "median PSD slope α = %.2f",
        summary["beta_30min_summary"]["median"],
        summary["beta_30min_summary"]["frac_above_1"] * 100,
        summary["psd_slope_summary"]["median"],
    )

    return paths


# ---------------------------------------------------------------------------
# Convenience driver — used by analyze.py
# ---------------------------------------------------------------------------

def reconstruct_wl_residuals_from_curve(
    clean_2D: np.ndarray, time_hr: np.ndarray,
    oot_mask: np.ndarray, best_fit_curve: np.ndarray,
    *, wl_left: int, wl_right: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper: rebuild WL residuals from saved best-fit curve."""
    wl_flux = np.nansum(clean_2D[:, wl_left:wl_right], axis=1)
    wl_med = np.nanmedian(wl_flux[oot_mask])
    wl_nor = wl_flux / wl_med
    keep = np.isfinite(wl_nor)
    return time_hr[keep], (wl_nor[keep] - best_fit_curve)
