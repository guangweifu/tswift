#!/usr/bin/env python3
"""End-to-end regression test on WASP-69 b.

Starting from the existing all_frame.npy ramp-fit output, runs through:
  bad_pixel -> extract -> spec_fit -> combine

Skips WL MCMC (uses legacy WL best-fit for orbital geometry so the spectral fit is
deterministic, not stochastic).

Compares the final binned spectrum against v1's existing v1_LD2.txt-derived spectrum.

Invocation (from repo root):
    /opt/anaconda3/envs/jwst_latest/bin/python tswift/scripts/test_end_to_end_wasp69b.py
"""
from __future__ import annotations

import logging
import time as _time
from pathlib import Path

import numpy as np
import yaml

from tswift import (
    mad_clip,
    run_extract,
    compute_ld_per_wavelength,
    fit_spec_curves,
    combine_spectrum,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e2e-test")

PROJ = Path("/Users/guangweifu/Documents/JWST_Tswift/WASP-69b")
V1 = PROJ / "product"
V2 = PROJ / "product_v2"


def main():
    cfg = yaml.safe_load((PROJ / "config.yaml").read_text())
    ex = cfg["extraction"]
    stellar = cfg["stellar"]
    ld_cfg = cfg["limb_darkening"]
    orbital = cfg["orbital"]
    mcmc_cfg = cfg["mcmc"]
    cf_cfg = cfg["curve_fit"]
    cb_cfg = cfg.get("combine", {})

    # Legacy WL best-fit for orbital geometry.
    v1_best = np.load(V1 / "mcmc_best_params.npy")
    _, _, _, _, a_wl, inc_wl, t0_wl = v1_best
    log.info(f"WL geometry from legacy: a={a_wl:.3f} inc={inc_wl:.3f} t0_off={t0_wl:.4f} hr")

    data_all = np.load(V1 / "all_frame.npy")
    time_all = np.load(V1 / "time_all.npy")
    wvl = np.load(V1 / "SOSS_wvl.npy")
    time_hr = (time_all - time_all[0]) * 24.0

    t0 = _time.time()
    log.info("STEP 1/4: bad_pixel MAD clip")
    data_fixed, _, _ = mad_clip(
        data_all,
        n_sigma=cfg["bad_pixel"]["n_sigma"],
        min_sigma=cfg["bad_pixel"]["min_sigma"],
    )

    log.info("STEP 2/4: extract (trace + aperture + clean)")
    er = run_extract(
        data_fixed,
        mode="SOSS", detector="nis",
        trace_half_width=ex["trace_half_width"],
        trace_poly_order=ex.get("trace_poly_order", 5),
        trace_outlier_clip=ex.get("trace_outlier_clip", 4.0),
        aperture_criterion=ex["aperture_criterion"],
        wavelength_left=ex["wavelength_left"],
        wavelength_right=ex["wavelength_right"],
        outlier_window=cfg.get("outlier_removal", {}).get("window", 10),
        outlier_threshold=cfg.get("outlier_removal", {}).get("threshold", 5.0),
    )
    log.info(f"  aperture {er['aperture']}, OOT baseline {int(er['oot_mask'].sum())} frames")

    log.info("STEP 3/4: spec_fit")
    u1, u2 = compute_ld_per_wavelength(
        wvl, ex["wavelength_left"], ex["wavelength_right"],
        stellar_teff=stellar["Teff"], stellar_logg=stellar["logg"], stellar_mh=stellar["M_H"],
        ld_model=ld_cfg["model"], ld_mode=ld_cfg["mode"],
        ld_data_path=cfg["paths"]["ld_data"],
    )
    sr = fit_spec_curves(
        er["clean_2D"], time_hr, wvl,
        wavelength_left=ex["wavelength_left"],
        wavelength_right=ex["wavelength_right"],
        period_days=orbital["period"],
        a_over_rs=a_wl, inclination_deg=inc_wl, t0_offset_hr=t0_wl,
        ecc=orbital.get("ecc", 0.0), omega=orbital.get("omega", 90.0),
        u1_arr=u1, u2_arr=u2,
        bounds_lower=tuple(cf_cfg["bounds"]["lower"]),
        bounds_upper=tuple(cf_cfg["bounds"]["upper"]),
        fix_ld2=cf_cfg.get("fix_ld2_stagger", False),
        mask_indices=np.asarray(mcmc_cfg.get("mask_indices", []), dtype=int),
    )

    log.info("STEP 4/4: combine + rebin")
    rp = sr["fit"][:, 1]
    rp_err = sr["fit_err"][:, 1]
    combined = combine_spectrum(
        wvl, rp, rp_err,
        bin_widths_nm=cb_cfg.get("bin_widths_nm", [10, 20, 50]),
        bad_wavelengths_um=cb_cfg.get("bad_wavelengths", None),
    )

    log.info(f"Total walltime: {_time.time() - t0:.1f}s")

    log.info("--- Regression diff vs v1 ---")
    v1_fit = np.loadtxt(V1 / "v1_LD2" / "v1_LD2.txt")
    v1_err = np.loadtxt(V1 / "v1_LD2" / "v1_LD2_err.txt")
    left = ex["wavelength_left"]
    right = min(ex["wavelength_right"], len(wvl))
    v2_slice = sr["fit"][left:right, 1]
    v2_err_slice = sr["fit_err"][left:right, 1]
    v1_rp = v1_fit[:, 1]
    v1_rp_err = v1_err[:, 1]

    rp_diff = np.abs(v2_slice - v1_rp)
    log.info(f"native Rp/Rs: max |v2 - v1| = {np.nanmax(rp_diff):.3e}  "
             f"median = {np.nanmedian(rp_diff):.3e}")

    v2_depths = (v2_slice ** 2) * 1e6
    v1_depths = (v1_rp ** 2) * 1e6
    depth_diff = np.abs(v2_depths - v1_depths)
    log.info(f"native depth (ppm): max |diff| = {np.nanmax(depth_diff):.3g}  "
             f"median |diff| = {np.nanmedian(depth_diff):.3g}")

    log.info("--- Binned spectrum summaries (v2) ---")
    for name, b in combined["binned"].items():
        log.info(f"  {name}: {len(b['wvl_um'])} bins, "
                 f"mean depth {np.mean(b['depth_ppm']):.0f} ± {np.mean(b['depth_err_ppm']):.0f} ppm, "
                 f"wvl {b['wvl_um'][0]:.3f}–{b['wvl_um'][-1]:.3f} µm")


if __name__ == "__main__":
    main()
