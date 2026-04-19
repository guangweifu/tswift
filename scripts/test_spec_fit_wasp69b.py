#!/usr/bin/env python3
"""Smoke test: v2 spec_fit on a subset of WASP-69 b wavelengths (fast).

Uses legacy WL best-fit for (a, inc, t0) so the spectral fit has no dependency on
the v2 WL MCMC still being noisy from short chains. Fits 50 wavelengths in the
middle of the bandpass and compares depths against legacy v1_LD2.txt.

Invocation:
    /opt/anaconda3/envs/jwst_latest/bin/python tswift/scripts/test_spec_fit_wasp69b.py
"""
from __future__ import annotations

import logging
import time as _time
from pathlib import Path

import numpy as np
import yaml

from tswift import compute_ld_per_wavelength, fit_spec_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spec-fit-test")

PROJ = Path("/Users/guangweifu/Documents/JWST_Tswift/WASP-69b")
V1 = PROJ / "product"
V2 = PROJ / "product_v2"


def main():
    cfg = yaml.safe_load((PROJ / "config.yaml").read_text())
    extraction = cfg["extraction"]
    stellar = cfg["stellar"]
    ld_cfg = cfg["limb_darkening"]
    orbital = cfg["orbital"]
    mcmc_cfg = cfg["mcmc"]
    cf_cfg = cfg["curve_fit"]

    # Use legacy WL best-fit so this test is independent of WL MCMC convergence.
    v1_best = np.load(V1 / "mcmc_best_params.npy")
    _, rp_wl, _, _, a_wl, inc_wl, t0_wl = v1_best
    log.info(f"Using legacy WL best-fit: a={a_wl:.3f}, inc={inc_wl:.3f}, t0_offset={t0_wl:.4f} hr")

    cl = np.load(V2 / "clean_2D.npy")  # identical to v1
    time_all = np.load(V1 / "time_all.npy")
    wvl = np.load(V1 / "SOSS_wvl.npy")
    time_hr = (time_all - time_all[0]) * 24.0

    # Narrow window for the smoke test: 50 channels in the middle of the bandpass.
    center = (extraction["wavelength_left"] + extraction["wavelength_right"]) // 2
    w_left = max(extraction["wavelength_left"], center - 25)
    w_right = min(extraction["wavelength_right"], center + 25)
    log.info(f"Narrow test window: wvl pixels [{w_left}, {w_right}) "
             f"= {wvl[w_left]:.3f}-{wvl[w_right-1]:.3f} µm")

    t0 = _time.time()
    u1, u2 = compute_ld_per_wavelength(
        wvl, w_left, w_right,
        stellar_teff=stellar["Teff"],
        stellar_logg=stellar["logg"],
        stellar_mh=stellar["M_H"],
        ld_model=ld_cfg["model"],
        ld_mode=ld_cfg["mode"],
        ld_data_path=cfg["paths"]["ld_data"],
    )
    log.info(f"LD coefficients computed in {_time.time() - t0:.1f}s — "
             f"u1 range [{np.nanmin(u1[w_left:w_right]):.3f}, "
             f"{np.nanmax(u1[w_left:w_right]):.3f}], "
             f"u2 range [{np.nanmin(u2[w_left:w_right]):.3f}, "
             f"{np.nanmax(u2[w_left:w_right]):.3f}]")

    t1 = _time.time()
    result = fit_spec_curves(
        cl, time_hr, wvl,
        wavelength_left=w_left, wavelength_right=w_right,
        period_days=orbital["period"],
        a_over_rs=a_wl, inclination_deg=inc_wl,
        t0_offset_hr=t0_wl,
        ecc=orbital.get("ecc", 0.0),
        omega=orbital.get("omega", 90.0),
        u1_arr=u1, u2_arr=u2,
        bounds_lower=tuple(cf_cfg["bounds"]["lower"]),
        bounds_upper=tuple(cf_cfg["bounds"]["upper"]),
        fix_ld2=cf_cfg.get("fix_ld2_stagger", False),
        mask_indices=np.asarray(mcmc_cfg.get("mask_indices", []), dtype=int),
    )
    log.info(f"Spec fit walltime: {_time.time() - t1:.1f}s for {w_right - w_left} channels")

    # Compare to legacy v1_LD2.txt
    v1_fit = np.loadtxt(V1 / "v1_LD2" / "v1_LD2.txt")
    log.info(f"Legacy v1_LD2.txt shape: {v1_fit.shape}")

    # v1_LD2.txt is stored densely for [left:right]; our fit_arr is (n_wvl, 4) indexed
    # at absolute wavelength pixels. Compare on the overlapping range.
    legacy_left = extraction["wavelength_left"]
    rows_in_legacy = slice(w_left - legacy_left, w_right - legacy_left)
    v1_slice = v1_fit[rows_in_legacy]
    v2_slice = result["fit"][w_left:w_right]

    diff = np.abs(v2_slice - v1_slice)
    rp_diff = diff[:, 1]
    log.info(f"Rp/Rs: max |v2 - v1| = {np.nanmax(rp_diff):.3e}  "
             f"median |v2 - v1| = {np.nanmedian(rp_diff):.3e}")

    depths_v2 = (v2_slice[:, 1] ** 2) * 1e6
    depths_v1 = (v1_slice[:, 1] ** 2) * 1e6
    log.info(f"Transit depth (ppm): v2 mean {np.nanmean(depths_v2):.0f}, "
             f"v1 mean {np.nanmean(depths_v1):.0f}, "
             f"max |diff| {np.nanmax(np.abs(depths_v2 - depths_v1)):.1f} ppm")


if __name__ == "__main__":
    main()
