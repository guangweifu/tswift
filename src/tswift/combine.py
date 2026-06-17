"""Turn per-wavelength Rp/Rs fits into a transmission spectrum at one or more
binning resolutions.

For SOSS (single detector) this is just depth conversion + rebinning. For NIRSpec
G395H you have two detectors (nrs1, nrs2) to stitch first — `combine_detectors()`
handles that case.

Error propagation: Rp/Rs → depth uses `depth = sign(rp) · rp² · 1e6` and
`sigma_depth = 2·|rp|·sigma_rp · 1e6`. The sign is preserved so unphysical negative
depths (fit failures near the grid edge) stay flagged as negative rather than
silently becoming positive on squaring.

Rebinning uses inverse-variance weighting within each bin: `v_bin = Σ(v·w) / Σ(w)`
with `w = 1/σ²` and `σ_bin = 1 / √Σw`. Bins with zero valid samples are dropped.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def rp_to_depth_ppm(rp: np.ndarray, rp_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Rp/Rs → transit depth (ppm) with sign preservation + error propagation."""
    depth = np.sign(rp) * rp ** 2 * 1e6
    depth_err = 2.0 * np.abs(rp) * rp_err * 1e6
    return depth, depth_err


def bin_inverse_variance(
    wvl: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin (values, errors) to the grid defined by `bin_edges` using 1/σ² weights.

    Returns (bin_centers, binned_values, binned_errors), each of length
    `len(bin_edges) - 1` minus any empty bins (which are dropped).
    """
    centers, vals, errs = [], [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        m = (wvl >= lo) & (wvl < hi) & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
        if not m.any():
            continue
        w = 1.0 / errors[m] ** 2
        centers.append(0.5 * (lo + hi))
        vals.append(float(np.sum(values[m] * w) / np.sum(w)))
        errs.append(float(1.0 / np.sqrt(np.sum(w))))
    return np.asarray(centers), np.asarray(vals), np.asarray(errs)


def combine_spectrum(
    wvl: np.ndarray,
    rp: np.ndarray,
    rp_err: np.ndarray,
    *,
    bin_widths_nm: list[float] = (10, 20, 50),
    bad_wavelengths_um: Optional[list[float]] = None,
    wavelength_range_um: Optional[tuple[float, float]] = None,
) -> dict:
    """Single-detector spectrum combine.

    Parameters
    ----------
    wvl : (n_wvl,) float
        Wavelength in micrometers.
    rp, rp_err : (n_wvl,) float
        Rp/Rs and 1-sigma uncertainty per wavelength.
    bin_widths_nm : iterable of float
        Bin widths in nanometers to produce in addition to the native spectrum.
    bad_wavelengths_um : list of float, optional
        Micron values to mask out before binning. Matched with ±1 nm tolerance.
    wavelength_range_um : (lo, hi), optional
        Restrict bin edges to this range. Default uses the (finite) data extent.

    Returns
    -------
    dict with:
        native     : dict(wvl, depth_ppm, depth_err_ppm, rp, rp_err)
        binned     : {"10nm": {...}, ...} each with (wvl, depth_ppm, depth_err_ppm)
    """
    wvl = np.asarray(wvl, dtype=float)
    rp = np.asarray(rp, dtype=float)
    rp_err = np.asarray(rp_err, dtype=float)

    depth, depth_err = rp_to_depth_ppm(rp, rp_err)

    good = np.isfinite(rp) & np.isfinite(rp_err) & (rp_err > 0)

    if bad_wavelengths_um:
        bad = np.asarray(bad_wavelengths_um, dtype=float)
        wvl_near_bad = np.any(
            np.abs(wvl[:, None] - bad[None, :]) < 0.001,  # 1 nm tolerance
            axis=1,
        )
        good &= ~wvl_near_bad
        logger.info(f"Masked {int(wvl_near_bad.sum())} wavelengths near bad list")

    w_good = wvl[good]
    d_good = depth[good]
    e_good = depth_err[good]

    if wavelength_range_um is not None:
        lo, hi = wavelength_range_um
    else:
        lo = float(np.nanmin(w_good)) if w_good.size else 0.0
        hi = float(np.nanmax(w_good)) if w_good.size else 1.0

    binned = {}
    for width_nm in bin_widths_nm:
        width_um = width_nm / 1000.0
        # Deterministic edge count (np.arange's float stop can silently drop the
        # reddest partial bin). ceil guarantees the last edge reaches >= hi.
        n_bins = int(np.ceil((hi - lo) / width_um))
        if n_bins < 1:
            continue
        edges = lo + width_um * np.arange(n_bins + 1)
        # bin_inverse_variance uses a half-open [lo, hi) test, so nudge the final
        # edge just past hi to keep the reddest sample when (hi-lo) is an exact
        # multiple of the bin width.
        if edges[-1] <= hi:
            edges[-1] = np.nextafter(hi, hi + 1.0)
        wc, dep, err = bin_inverse_variance(w_good, d_good, e_good, edges)
        binned[f"{int(width_nm)}nm"] = {
            "wvl_um": wc,
            "depth_ppm": dep,
            "depth_err_ppm": err,
        }
        logger.info(
            f"Bin {width_nm} nm: {len(wc)} bins "
            f"(mean depth {np.mean(dep):.0f} ± {np.mean(err):.0f} ppm)"
        )

    return {
        "native": {
            "wvl_um": wvl,
            "depth_ppm": depth,
            "depth_err_ppm": depth_err,
            "rp": rp,
            "rp_err": rp_err,
            "mask": good,
        },
        "binned": binned,
    }


def combine_detectors(
    detectors: dict[str, dict],
    *,
    bin_widths_nm: list[float] = (10, 20, 50),
    bad_wavelengths_um: Optional[list[float]] = None,
) -> dict:
    """Stitch multiple detectors and return a combined spectrum.

    Parameters
    ----------
    detectors : dict[name, {"wvl_um":..., "rp":..., "rp_err":...}]
        One entry per detector.

    Used for NIRSpec G395H (nrs1+nrs2), whose wavelength ranges do not overlap.
    If detectors did overlap, native-resolution channels are concatenated and
    sorted but kept as duplicates (NOT merged at native resolution); only the
    binned outputs combine overlapping channels, via inverse-variance weighting
    within each bin.
    """
    all_w, all_rp, all_err = [], [], []
    for name, d in detectors.items():
        logger.info(f"Detector {name}: {len(d['wvl_um'])} wavelengths "
                    f"{d['wvl_um'][0]:.3f}-{d['wvl_um'][-1]:.3f} µm")
        all_w.append(d["wvl_um"])
        all_rp.append(d["rp"])
        all_err.append(d["rp_err"])

    wvl = np.concatenate(all_w)
    rp = np.concatenate(all_rp)
    rp_err = np.concatenate(all_err)

    order = np.argsort(wvl)
    return combine_spectrum(
        wvl[order], rp[order], rp_err[order],
        bin_widths_nm=bin_widths_nm, bad_wavelengths_um=bad_wavelengths_um,
    )
