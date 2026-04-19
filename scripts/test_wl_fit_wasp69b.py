#!/usr/bin/env python3
"""Smoke test: run v2 WL MCMC on WASP-69 b with a short chain (200 steps).

Confirms the fit runs end-to-end through extraction v2's clean_2D, with the correct
normalization / LD / priors pulled from the existing WASP-69b/config.yaml. Compares
best-fit parameters loosely against the legacy mcmc_best_params.npy.

Invocation (from repo root):
    /opt/anaconda3/envs/jwst_latest/bin/python tswift/scripts/test_wl_fit_wasp69b.py
"""
from __future__ import annotations

import logging
import time as _time
from pathlib import Path

import numpy as np
import yaml
from exotic_ld import StellarLimbDarkening

from tswift import fit_wl_mcmc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wl-fit-test")

PROJ = Path("/Users/guangweifu/Documents/JWST_Tswift/WASP-69b")
V1 = PROJ / "product"
V2 = PROJ / "product_v2"


def main():
    cfg = yaml.safe_load((PROJ / "config.yaml").read_text())
    extraction = cfg["extraction"]
    mcmc_cfg = cfg["mcmc"]
    ld_cfg = cfg["limb_darkening"]
    orbital = cfg["orbital"]
    stellar = cfg["stellar"]

    # Load v2's clean_2D (matches v1 exactly).
    cl = np.load(V2 / "clean_2D.npy")
    time_all = np.load(V1 / "time_all.npy")
    wvl = np.load(V1 / "SOSS_wvl.npy")

    left, right = extraction["wavelength_left"], extraction["wavelength_right"]
    right = min(right, cl.shape[1])
    log.info(f"cl shape {cl.shape}, wvl pixels [{left}, {right})")

    wl = np.sum(cl[:, left:right], axis=1)
    time_hr = (time_all - time_all[0]) * 24.0

    # OOT mask: first + last 15%.
    n = len(wl)
    n_edge = max(10, int(n * 0.15))
    oot_full = np.zeros(n, bool)
    oot_full[:n_edge] = True
    oot_full[-n_edge:] = True

    oot_med = float(np.median(wl[oot_full]))
    wl_nor = wl / oot_med
    oot_std = float(np.std(wl_nor[oot_full]))
    err = np.ones_like(wl_nor) * oot_std
    log.info(f"OOT: {n_edge}+{n_edge} frames, median {oot_med:.0f}, std {oot_std:.6f}")

    # Mask bad indices from config (ramp settling + post-transit anomaly).
    mask_idx = np.asarray(mcmc_cfg.get("mask_indices", []), dtype=int)
    keep = np.ones(n, bool)
    keep[mask_idx] = False

    t_fit = time_hr[keep]
    wl_fit = wl_nor[keep]
    err_fit = err[keep]
    oot_fit_mask = oot_full[keep]
    oot_indices_in_fit = np.where(oot_fit_mask)[0]
    log.info(f"After masking: {keep.sum()} / {n} frames; "
             f"{oot_indices_in_fit.size} are OOT")

    # Limb darkening
    wvl_aa = wvl[left:right] * 10_000.0
    sld = StellarLimbDarkening(
        stellar["M_H"], stellar["Teff"], stellar["logg"],
        ld_cfg["model"], cfg["paths"]["ld_data"], verbose=False,
    )
    u1, u2 = sld.compute_quadratic_ld_coeffs(
        [float(wvl_aa[0]), float(wvl_aa[-1])], ld_cfg["mode"]
    )
    log.info(f"LD coeffs (bandpass {wvl_aa[0]:.0f}–{wvl_aa[-1]:.0f} Å): u1={u1:.4f}, u2={u2:.4f}")

    # Run MCMC — SHORT chain for smoke test. Real fit uses 8000 steps.
    t0 = _time.time()
    result = fit_wl_mcmc(
        time_data_hr=t_fit,
        flux_data=wl_fit,
        flux_err=err_fit,
        period_hr=orbital["period"] * 24.0,
        ecc=orbital.get("ecc", 0.0),
        omega=orbital.get("omega", 90.0),
        u1=u1, u2=u2,
        oot_indices=oot_indices_in_fit,
        initial=mcmc_cfg["initial"],
        priors=mcmc_cfg["priors"],
        fit_ld1=mcmc_cfg.get("fit_ld1", False),
        nwalkers=48,
        nsteps=200,
        nburn=50,
        rng_seed=42,
    )
    elapsed = _time.time() - t0
    log.info(f"MCMC wall time: {elapsed:.1f}s")

    log.info("--- best-fit parameters (v2, 200 steps — not converged) ---")
    for name, val, (elo, ehi) in zip(
        result["param_order"], result["best_params"], result["best_errors"]
    ):
        log.info(f"  {name:12s} = {val:+.6f}  (+{ehi:.4g} / -{elo:.4g})")

    log.info(f"Residual RMS: {result['rms_residual']:.6f} ({result['rms_residual']*1e6:.0f} ppm)")
    log.info(f"Mean acceptance: {result['acceptance'].mean():.3f} "
             f"(walker range {result['acceptance'].min():.3f}–{result['acceptance'].max():.3f})")

    # Compare to legacy converged best-fit.
    v1_best = np.load(V1 / "mcmc_best_params.npy")
    log.info(f"--- legacy converged best-fit (v1, 8000 steps): {v1_best} ---")
    if v1_best.shape == result["best_params"].shape:
        drift = np.abs(result["best_params"] - v1_best)
        for name, v2val, v1val, d in zip(result["param_order"], result["best_params"], v1_best, drift):
            log.info(f"  {name:12s} v1={v1val:+.6f}  v2={v2val:+.6f}  Δ={d:.3g}")
    else:
        log.info(f"Shape mismatch v1={v1_best.shape} v2={result['best_params'].shape} "
                 "(legacy may have used different fit_ld1 setting)")


if __name__ == "__main__":
    main()
