#!/usr/bin/env python3
"""Regression test: run v2 extraction on WASP-69 b and diff against v1 outputs.

Invocation (from repo root):
    /opt/anaconda3/envs/jwst_latest/bin/python tswift/scripts/test_extract_wasp69b.py
"""
from __future__ import annotations

import logging
import time as _time
from pathlib import Path

import numpy as np

from tswift import mad_clip, run_extract, save_extract_outputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract-test")

PROJ = Path("/Users/guangweifu/Documents/JWST_Tswift/WASP-69b")
V1 = PROJ / "product"
V2 = PROJ / "product_v2"


def _maxabs(a, b):
    return float(np.nanmax(np.abs(a - b)))


def main():
    log.info(f"Loading all_frame.npy from {V1}")
    data_all = np.load(V1 / "all_frame.npy")
    log.info(f"Loaded {data_all.shape} {data_all.dtype}")

    t0 = _time.time()
    log.info("Running MAD-robust bad-pixel clip (n_sigma=5.0, min_sigma=2.0)")
    data_fixed, bad_mask, sigma_pix = mad_clip(data_all, n_sigma=5.0, min_sigma=2.0)
    log.info(f"Flagged {int(bad_mask.sum()):,} samples "
             f"({bad_mask.mean() * 100:.3f}%) in {_time.time() - t0:.1f}s")

    # Regression diff: v1's bad_pix_fix_step_3.npy should be identical (same algo).
    v1_fixed = np.load(V1 / "bad_pix_fix_step_3.npy")
    nan_match = np.isnan(data_fixed) == np.isnan(v1_fixed)
    log.info(f"Bad-pixel mask vs v1: NaN agreement = {nan_match.mean() * 100:.3f}%")

    with np.errstate(invalid="ignore"):
        mdiff = _maxabs(data_fixed, v1_fixed)
    log.info(f"Bad-pixel values vs v1: max |diff| = {mdiff:.3g}")

    log.info("Running v2 extract: trace + aperture + clean")
    t1 = _time.time()
    result = run_extract(
        data_fixed,
        mode="SOSS",
        detector="nis",
        trace_half_width=22,
        trace_poly_order=5,
        trace_outlier_clip=4.0,
        aperture_criterion="per_channel",
        wavelength_left=50,
        wavelength_right=2040,
        outlier_window=10,
        outlier_threshold=5.0,
    )
    log.info(f"Extraction wall time: {_time.time() - t1:.1f}s")
    log.info(f"Result keys: {list(result.keys())}")
    log.info(f"Trace median row: {np.nanmedian(result['trace_fit']):.2f}")
    log.info(f"Aperture: {result['aperture']}")
    log.info(f"clean_2D shape: {result['clean_2D'].shape}")

    save_extract_outputs(result, V2)

    log.info("--- REGRESSION DIFFS vs v1 ---")
    # trace_fit
    v1_trace = np.load(V1 / "trace_fit.npy")
    log.info(f"trace_fit: max |v2 - v1| = {_maxabs(result['trace_fit'], v1_trace):.3g} rows")

    # clean_2D (numeric comparison over both values)
    v1_clean = np.load(V1 / "clean_2D.npy")
    abs_err = np.abs(result["clean_2D"] - v1_clean)
    rel_err = abs_err / np.maximum(np.abs(v1_clean), 1.0)
    log.info(f"clean_2D: max |v2 - v1| = {np.nanmax(abs_err):.3g}  "
             f"median |v2 - v1| = {np.nanmedian(abs_err):.3g}  "
             f"median rel = {np.nanmedian(rel_err) * 100:.3f}%")

    # White-light RMS as a sanity metric
    time_all = np.load(V1 / "time_all.npy")
    oot = result["oot_mask"]
    wl = np.nansum(result["clean_2D"][:, 50:2040], axis=1)
    wl_nor = wl / np.nanmedian(wl[oot])
    rms_ppm = np.nanstd(wl_nor[oot]) * 1e6
    log.info(f"White-light OOT RMS (v2): {rms_ppm:.0f} ppm  "
             f"(over {int(oot.sum())} frames)")

    wl_v1 = np.nansum(v1_clean[:, 50:2040], axis=1)
    wl_nor_v1 = wl_v1 / np.nanmedian(wl_v1[oot])
    rms_ppm_v1 = np.nanstd(wl_nor_v1[oot]) * 1e6
    log.info(f"White-light OOT RMS (v1): {rms_ppm_v1:.0f} ppm")


if __name__ == "__main__":
    main()
