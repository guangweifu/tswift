"""He I 1083 nm pixel-level check for SOSS transits.

(In principle this works for any mode whose bandpass covers 1.083 µm, but only
SOSS is exercised in the pipeline — the ±5 nm WCS-offset search below is tuned
to SOSS GR700XD.)

The metastable He I triplet at 1083.3 nm (vacuum) is the standard
near-IR escaping-atmosphere tracer.  When the planet has a hydrodynamic
wind, the He absorption can extend beyond the optical transit (a
"comet-tail" of escaping gas), with three observational consequences:

1. **The He pixel is deeper than the surrounding continuum.**  Just
   look at the per-channel transit-depth spectrum — a ~few hundred ppm
   excess at 1.083 µm is typical for hot Jupiters with winds.

2. **The He light curve has extended ingress / egress shoulders.**
   Material outside the planet's optical limb is still absorbing in
   He, so the transit looks longer in this band.

3. **Naive pipelines silently bias the He depth LOW.**  Standard
   spec-fit pipelines pick "OOT frames" by `wl_flux > 0.998 × median`
   on the white-light curve.  But pre/post-transit He absorption from
   a tail makes those "OOT" frames also slightly absorbed at 1.083 µm.
   The per-channel OOT median pulls down with the tail, which makes
   the in-transit dip look smaller — bias is the worst exactly when
   the science is most interesting.

This module produces the standard sanity-check artefacts that catch
all three issues at a glance:

- ``helium_pixel_spectrum.png`` — pixel-level depth zoomed to ±25 nm
  around 1083 nm with the local continuum band marked and any
  >Nσ-excess pixels highlighted.
- ``helium_lightcurves.png`` — He pixel LC overplotted on the WL LC,
  with a residual panel `(channel − WL)` so any pre / post-transit
  asymmetry is visible.
- ``helium_summary.json`` — machine-readable: He pixel column, depth,
  excess significance, pre/post/in-transit residuals under both
  "local OOT" (pipeline default) and "pre+post balanced" normalizations.

**Wavelength caveats**
SOSS GR700XD has a ~1-nm WCS offset at 1.08 µm — the max-excess pixel
typically lands ~1 nm blueward of the catalog wavelength.  This module
SEARCHES within a configurable window (default ±5 nm) for the
maximum-excess pixel rather than trusting the WCS to better than a
fraction of a pixel.

Use as a normal pipeline step:

    from tswift import check_helium_1083, plot_helium

    result = check_helium_1083(
        clean_2D, time_hr, wvl,
        rp=spec_fit[:, 1], rp_err=spec_err[:, 1], oot_mask=oot_mask,
        wl_left=..., wl_right=..., t0_hr=wl_best["t0_offset"],
    )
    plot_helium(result, outdir="product/helium",
                planet_name="Planet X b")

When `result["max_excess_significance_sigma"] > 3`, you have a
detection and the LC plot is your first sanity check that the depth
isn't biased by a tail.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
HE_I_1083_VAC_NM = 1083.30   # Triplet centroid (vacuum); air ~ 1083.0 nm
DEFAULT_SEARCH_HALF_NM = 5.0
DEFAULT_ZOOM_HALF_NM = 25.0


@dataclass
class HeliumResult:
    """Per-pixel He I 1083 nm measurements.

    All wavelengths in nanometers, all depths in ppm.  The class is a
    plain data container — no MCMC or fits run inside.
    """
    he_vac_nm: float
    max_excess_col: int
    max_excess_wvl_nm: float
    max_excess_depth_ppm: float
    max_excess_depth_err_ppm: float
    local_continuum_ppm: float
    local_continuum_mad_ppm: float
    excess_above_continuum_ppm: float
    excess_significance_sigma: float
    n_excess_pixels: int
    cols_zoom: np.ndarray = field(repr=False)
    wvl_zoom_nm: np.ndarray = field(repr=False)
    depth_zoom_ppm: np.ndarray = field(repr=False)
    depth_err_zoom_ppm: np.ndarray = field(repr=False)
    excess_pix_mask: np.ndarray = field(repr=False)
    # Light-curve products keyed on the max-excess column.
    time_hr: np.ndarray = field(repr=False)
    wl_lc: np.ndarray = field(repr=False)
    he_lc_local: np.ndarray = field(repr=False)
    he_lc_balanced: np.ndarray = field(repr=False)
    pre_mask: np.ndarray = field(repr=False)
    post_mask: np.ndarray = field(repr=False)
    in_transit_mask: np.ndarray = field(repr=False)
    pre_excess_ppm_local: float
    post_excess_ppm_local: float
    in_transit_excess_ppm_local: float
    pre_excess_ppm_balanced: float
    post_excess_ppm_balanced: float
    in_transit_excess_ppm_balanced: float
    post_minus_pre_local_ppm: float
    observed_t1_hr: Optional[float]
    observed_t4_hr: Optional[float]


def check_helium_1083(
    clean_2D: np.ndarray,
    time_hr: np.ndarray,
    wvl: np.ndarray,
    rp: np.ndarray,
    rp_err: np.ndarray,
    oot_mask: np.ndarray,
    *,
    wl_left: int,
    wl_right: int,
    he_vac_nm: float = HE_I_1083_VAC_NM,
    search_half_nm: float = DEFAULT_SEARCH_HALF_NM,
    zoom_half_nm: float = DEFAULT_ZOOM_HALF_NM,
    excess_sigma: float = 3.0,
    oot_flux_threshold: float = 0.998,
    t0_hr: Optional[float] = None,
) -> HeliumResult:
    """Find the He I 1083 nm pixel and quantify excess + tail signatures.

    Parameters
    ----------
    clean_2D : (n_frames, n_cols)
        Per-channel cleaned light curves.
    time_hr : (n_frames,)
    wvl : (n_cols,)
        Wavelength solution (µm).
    rp, rp_err : (n_cols,)
        Per-channel Rp/Rs fit results from ``fit_spec_curves`` —
        these are the inputs to depth = rp² · 1e6.
    oot_mask : (n_frames,) bool
        Pipeline OOT mask (used only for WL normalization; pre/post
        masks are recomputed from `wl_flux` directly so this works
        even when oot_mask is pre-transit-only).
    wl_left, wl_right : int
        Column slice used for the WL light curve sum.
    he_vac_nm : float
        He I triplet centroid; default 1083.3 nm.
    search_half_nm : float
        Half-width of the "look for max-excess pixel" window around
        ``he_vac_nm``.  SOSS WCS is offset by ~1 nm at this
        wavelength, so the default 5 nm allows the search to find the
        true line even when the WCS is slightly off.
    zoom_half_nm : float
        Half-width of the diagnostic-plot zoom window.
    excess_sigma : float
        Significance threshold for flagging excess pixels.
    oot_flux_threshold : float
        WL-flux floor that defines OOT vs in-transit (default 0.998
        of the WL OOT median; conservative, drops borderline
        ingress/egress frames out of OOT).
    t0_hr : float, optional
        Transit center time (hours from visit start), used to split
        OOT into pre/post halves.  If None, midpoint of in-transit
        frames is used.

    Returns
    -------
    HeliumResult
    """
    n_frames, n_cols = clean_2D.shape
    rp = np.asarray(rp, dtype=float)
    rp_err = np.asarray(rp_err, dtype=float)
    depth = np.sign(rp) * rp ** 2 * 1e6
    depth_err = 2 * np.abs(rp) * rp_err * 1e6
    wvl_nm = np.asarray(wvl) * 1000.0

    # WL light curve.
    wl_flux = np.nansum(clean_2D[:, wl_left:wl_right], axis=1)
    wl_med = np.nanmedian(wl_flux[oot_mask])
    wl_lc = wl_flux / wl_med

    # Robust pre/post OOT split — recompute from WL flux directly so we
    # get both halves even if the saved oot_mask is pre-only.
    in_oot = wl_lc > oot_flux_threshold
    in_idx = np.where(~in_oot)[0]
    if len(in_idx):
        observed_t1_hr = float(time_hr[in_idx[0]])
        observed_t4_hr = float(time_hr[in_idx[-1]])
        if t0_hr is None:
            t0_hr = 0.5 * (observed_t1_hr + observed_t4_hr)
    else:
        observed_t1_hr = observed_t4_hr = None
        if t0_hr is None:
            t0_hr = float(np.median(time_hr))

    pre_mask = in_oot & (time_hr < t0_hr)
    post_mask = in_oot & (time_hr > t0_hr)
    in_transit_mask = ~in_oot

    # Search for the max-excess pixel.  Zoom is for plotting; search is
    # narrower.
    he_um = he_vac_nm / 1000.0
    search_half_um = search_half_nm / 1000.0
    zoom_half_um = zoom_half_nm / 1000.0
    search_mask = (np.asarray(wvl) > he_um - search_half_um) & \
                  (np.asarray(wvl) < he_um + search_half_um)
    zoom_mask = (np.asarray(wvl) > he_um - zoom_half_um) & \
                (np.asarray(wvl) < he_um + zoom_half_um)
    cols_search = np.where(search_mask)[0]
    cols_zoom = np.where(zoom_mask)[0]
    if cols_search.size == 0 or cols_zoom.size == 0:
        raise ValueError(
            f"He window 1083 nm not in the wavelength range "
            f"({wvl_nm.min():.1f}-{wvl_nm.max():.1f} nm)"
        )

    # Continuum from the zoom window EXCLUDING ±4 cols around the
    # nominal He center (so the line itself doesn't pollute the
    # continuum estimate).
    col_he_center = int(np.argmin(np.abs(np.asarray(wvl) - he_um)))
    exclude = set(range(col_he_center - 4, col_he_center + 5))
    cont_idx = np.array([c for c in cols_zoom if c not in exclude], dtype=int)
    cont_med = float(np.nanmedian(depth[cont_idx]))
    cont_mad = float(np.nanmedian(np.abs(depth[cont_idx] - cont_med))) * 1.4826
    if not np.isfinite(cont_med):
        cont_med = float(np.nanmedian(depth[cols_zoom]))
        # Keep the robust MAD scaling consistent with the primary path; only
        # fall back to a plain std if the MAD itself is non-finite.
        cont_mad = float(np.nanmedian(np.abs(depth[cols_zoom] - cont_med))) * 1.4826
        if not np.isfinite(cont_mad):
            cont_mad = float(np.nanstd(depth[cols_zoom]))

    # Excess significance per pixel in the search window.
    sig_search = (depth[cols_search] - cont_med) / np.maximum(
        depth_err[cols_search], 1.0
    )
    excess_pix_search = sig_search > excess_sigma
    n_excess = int(np.nansum(excess_pix_search))

    if excess_pix_search.any():
        # Pick the max-significance pixel.
        idx_in_search = int(np.nanargmax(sig_search * excess_pix_search))
        col_max = int(cols_search[idx_in_search])
    else:
        # No excess found — return the closest-to-line pixel as a placeholder.
        col_max = int(np.argmin(np.abs(np.asarray(wvl) - he_um)))

    excess_above = float(depth[col_max] - cont_med)
    excess_sig = float(excess_above / depth_err[col_max]) if depth_err[col_max] > 0 \
                 else float("nan")

    # Per-pixel LCs at col_max — local norm + balanced norm.
    f_he = clean_2D[:, col_max]
    pre_med = float(np.nanmedian(f_he[pre_mask])) if pre_mask.any() else np.nan
    post_med = float(np.nanmedian(f_he[post_mask])) if post_mask.any() else np.nan
    local_med = float(np.nanmedian(f_he[oot_mask])) if oot_mask.any() else np.nan
    if np.isfinite(pre_med) and np.isfinite(post_med):
        balanced_med = 0.5 * (pre_med + post_med)
    elif np.isfinite(pre_med):
        balanced_med = pre_med
    else:
        balanced_med = local_med if np.isfinite(local_med) else 1.0
    he_lc_local = f_he / local_med if np.isfinite(local_med) else f_he
    he_lc_balanced = f_he / balanced_med

    # Tail residuals (channel − WL).
    diff_local = he_lc_local - wl_lc
    diff_balanced = he_lc_balanced - wl_lc

    def _med_ppm(arr, mask):
        if not mask.any():
            return float("nan")
        return float(np.nanmedian(arr[mask]) * 1e6)

    pre_l  = _med_ppm(diff_local, pre_mask)
    post_l = _med_ppm(diff_local, post_mask)
    in_l   = _med_ppm(diff_local, in_transit_mask)
    pre_b  = _med_ppm(diff_balanced, pre_mask)
    post_b = _med_ppm(diff_balanced, post_mask)
    in_b   = _med_ppm(diff_balanced, in_transit_mask)
    post_minus_pre = post_l - pre_l if (np.isfinite(pre_l) and np.isfinite(post_l)) \
                     else float("nan")

    # Excess-pixel mask aligned with cols_zoom (for plotting).
    excess_zoom_mask = np.zeros(cols_zoom.size, dtype=bool)
    for k, c in enumerate(cols_zoom):
        if c in cols_search:
            j = int(np.where(cols_search == c)[0][0])
            excess_zoom_mask[k] = bool(excess_pix_search[j])

    logger.info(
        "He I 1083: max-excess col %d (wvl %.2f nm), depth=%.0f ± %.0f ppm, "
        "continuum=%.0f ± %.0f ppm, excess=%+.0f ppm (%.1fσ), n_excess=%d",
        col_max, wvl_nm[col_max], depth[col_max], depth_err[col_max],
        cont_med, cont_mad, excess_above, excess_sig, n_excess,
    )
    if np.isfinite(post_minus_pre):
        logger.info(
            "He I 1083 tail check: pre %+.0f ppm  /  post %+.0f ppm  "
            "(local norm) → post−pre = %+.0f ppm  %s",
            pre_l, post_l, post_minus_pre,
            "[possible tail bias]" if abs(post_minus_pre) > 100 else "",
        )

    return HeliumResult(
        he_vac_nm=he_vac_nm,
        max_excess_col=col_max,
        max_excess_wvl_nm=float(wvl_nm[col_max]),
        max_excess_depth_ppm=float(depth[col_max]),
        max_excess_depth_err_ppm=float(depth_err[col_max]),
        local_continuum_ppm=cont_med,
        local_continuum_mad_ppm=cont_mad,
        excess_above_continuum_ppm=excess_above,
        excess_significance_sigma=excess_sig,
        n_excess_pixels=n_excess,
        cols_zoom=cols_zoom,
        wvl_zoom_nm=wvl_nm[cols_zoom],
        depth_zoom_ppm=depth[cols_zoom],
        depth_err_zoom_ppm=depth_err[cols_zoom],
        excess_pix_mask=excess_zoom_mask,
        time_hr=np.asarray(time_hr),
        wl_lc=wl_lc,
        he_lc_local=he_lc_local,
        he_lc_balanced=he_lc_balanced,
        pre_mask=pre_mask,
        post_mask=post_mask,
        in_transit_mask=in_transit_mask,
        pre_excess_ppm_local=pre_l,
        post_excess_ppm_local=post_l,
        in_transit_excess_ppm_local=in_l,
        pre_excess_ppm_balanced=pre_b,
        post_excess_ppm_balanced=post_b,
        in_transit_excess_ppm_balanced=in_b,
        post_minus_pre_local_ppm=post_minus_pre,
        observed_t1_hr=observed_t1_hr,
        observed_t4_hr=observed_t4_hr,
    )


def plot_helium(
    result: HeliumResult,
    outdir: str | Path,
    *,
    planet_name: str = "",
    excess_sigma: float = 3.0,
    bin_width: int = 8,
) -> dict:
    """Write the two diagnostic PNGs + JSON summary.

    Returns ``dict[name → Path]`` for chaining.
    """
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    title_tag = planet_name + ("  " if planet_name else "")

    # ------------------------------------------------------------------
    # 1) Pixel-level depth zoom.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        result.wvl_zoom_nm, result.depth_zoom_ppm,
        yerr=result.depth_err_zoom_ppm, fmt="o", ms=4,
        color="tab:gray", alpha=0.7, label="per-pixel depth",
    )
    ax.axhspan(
        result.local_continuum_ppm - result.local_continuum_mad_ppm,
        result.local_continuum_ppm + result.local_continuum_mad_ppm,
        color="tab:blue", alpha=0.15,
        label=f"continuum {result.local_continuum_ppm:.0f} ± "
              f"{result.local_continuum_mad_ppm:.0f} ppm",
    )
    ax.axhline(result.local_continuum_ppm, color="tab:blue", lw=1.0, ls="--")
    ax.axvline(result.he_vac_nm, color="tab:red", lw=1.0, ls=":", alpha=0.6,
               label=f"He I {result.he_vac_nm:.1f} nm (vac)")
    if result.excess_pix_mask.any():
        idx = np.where(result.excess_pix_mask)[0]
        ax.errorbar(
            result.wvl_zoom_nm[idx], result.depth_zoom_ppm[idx],
            yerr=result.depth_err_zoom_ppm[idx], fmt="o", ms=8,
            color="tab:red", zorder=10,
        )
        ax.scatter([], [], s=80, color="tab:red",
                   label=f">{excess_sigma:.0f}σ excess pixel")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transit depth (ppm)")
    ax.set_title(
        f"{title_tag}He I 1083 nm pixel-level zoom  "
        f"(max excess = +{result.excess_above_continuum_ppm:.0f} ppm "
        f"@ {result.max_excess_wvl_nm:.2f} nm, "
        f"{result.excess_significance_sigma:.1f}σ)"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = outdir / "helium_pixel_spectrum.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["pixel_spectrum"] = p.resolve()

    # ------------------------------------------------------------------
    # 2) He pixel LC vs WL with residual.
    # ------------------------------------------------------------------
    def bin_down(arr, w):
        n = (len(arr) // w) * w
        return arr[:n].reshape(-1, w).mean(axis=1)

    t = result.time_hr
    tb = bin_down(t, bin_width)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9),
                             gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
    axes[0].scatter(t, result.wl_lc, s=2, alpha=0.25, color="lightgray")
    axes[0].plot(bin_down(t, 30), bin_down(result.wl_lc, 30),
                 color="black", lw=1.8, label="WL (binned ×30)")
    axes[0].plot(tb, bin_down(result.he_lc_local, bin_width),
                 color="tab:red", lw=1.5,
                 label=f"He pixel col {result.max_excess_col} "
                       f"({result.max_excess_wvl_nm:.2f} nm)")
    axes[0].set_ylabel("Normalized flux")
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].set_title(
        f"{title_tag}He I 1083 — pixel LC vs WL\n"
        f"He pixel depth = {result.max_excess_depth_ppm:.0f} ppm  "
        f"(continuum {result.local_continuum_ppm:.0f} ppm, excess "
        f"+{result.excess_above_continuum_ppm:.0f} ppm "
        f"= {result.excess_significance_sigma:.1f}σ)"
    )
    axes[0].grid(alpha=0.3)

    if result.observed_t1_hr is not None and result.observed_t4_hr is not None:
        for ax in axes:
            ax.axvspan(result.observed_t1_hr, result.observed_t4_hr,
                       color="tab:red", alpha=0.06, label="in-transit")

    diff_local = result.he_lc_local - result.wl_lc
    diff_balanced = result.he_lc_balanced - result.wl_lc
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].plot(tb, bin_down(diff_local, bin_width) * 1e6,
                 color="tab:red", lw=1.0, alpha=0.85,
                 label="He pixel − WL  (local OOT norm)")
    axes[1].plot(tb, bin_down(diff_balanced, bin_width) * 1e6,
                 color="tab:blue", lw=1.0, alpha=0.85,
                 label="He pixel − WL  (pre+post balanced norm)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Excess absorption (ppm)\nbinned (channel − WL)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    p = outdir / "helium_lightcurves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["lightcurves"] = p.resolve()

    # ------------------------------------------------------------------
    # JSON summary (only floats / ints — no arrays).
    # ------------------------------------------------------------------
    summary = {
        "he_vac_nm": result.he_vac_nm,
        "max_excess_col": result.max_excess_col,
        "max_excess_wvl_nm": result.max_excess_wvl_nm,
        "max_excess_depth_ppm": result.max_excess_depth_ppm,
        "max_excess_depth_err_ppm": result.max_excess_depth_err_ppm,
        "local_continuum_ppm": result.local_continuum_ppm,
        "local_continuum_mad_ppm": result.local_continuum_mad_ppm,
        "excess_above_continuum_ppm": result.excess_above_continuum_ppm,
        "excess_significance_sigma": result.excess_significance_sigma,
        "n_excess_pixels": result.n_excess_pixels,
        "observed_t1_hr": result.observed_t1_hr,
        "observed_t4_hr": result.observed_t4_hr,
        "pre_excess_ppm_local":  result.pre_excess_ppm_local,
        "post_excess_ppm_local": result.post_excess_ppm_local,
        "in_transit_excess_ppm_local": result.in_transit_excess_ppm_local,
        "pre_excess_ppm_balanced":  result.pre_excess_ppm_balanced,
        "post_excess_ppm_balanced": result.post_excess_ppm_balanced,
        "in_transit_excess_ppm_balanced": result.in_transit_excess_ppm_balanced,
        "post_minus_pre_local_ppm": result.post_minus_pre_local_ppm,
    }
    p = outdir / "helium_summary.json"
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    paths["summary"] = p.resolve()

    return paths
