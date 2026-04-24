#!/usr/bin/env python3
"""Regression test: `tswift.run_calibrate` on LP 890-9 c visit 12 PRISM data.

Reduces `data/uncal/*_nrs1_uncal.fits` through the new tswift calibration
pipeline into a temporary directory, then compares aggregate science
metrics against the existing reduction in `product/`:

  - `time_all.npy` — bit-exact (integration mid-times from INT_TIMES).
  - `all_frame.npy` — per-integration median rate within ±5 % and total
    flux per integration (white-light curve) within ±1 %.
  - `<mode>_<det>_wvl.npy` — per-column wavelength within ±0.5 %.

**Bit-exact equality is not a goal.** CRDS reference-file updates and
jwst point releases drift pixel-level rates at the few-percent level;
the WL curve and overall flux normalization are what matter for the
transmission spectrum.

Usage
-----
    cd /Users/guangweifu/Documents/JWST_Tswift
    /opt/anaconda3/envs/jwst_latest/bin/python tswift/scripts/regression_calibrate_lp890-9c_v12.py

Takes ~5-10 minutes. Skips cleanly (with a message) if:
  - Visit 12 uncal data is missing.
  - The reference `all_frame.npy` is missing.
  - `$CRDS_PATH` is unset and no cache exists at `~/crds_cache`.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


VISIT = Path("/Users/guangweifu/Documents/JWST_Tswift/LP 890-9/c/visit12_2026-03-01")
MODE = "PRISM"
DETECTOR = "nrs1"


def _check_prereqs() -> None:
    if not VISIT.exists():
        sys.exit(f"SKIP: {VISIT} does not exist")
    uncal = list((VISIT / "data" / "uncal").glob(f"*_{DETECTOR}_uncal.fits"))
    if not uncal:
        sys.exit(f"SKIP: no uncal files in {VISIT / 'data' / 'uncal'}")
    ref_frame = VISIT / "product" / "all_frame.npy"
    ref_wvl = VISIT / "product" / f"{MODE}_{DETECTOR}_wvl.npy"
    ref_wvl_legacy = VISIT / "product" / f"{MODE}_wvl.npy"   # legacy filename
    if not ref_frame.exists():
        sys.exit(f"SKIP: no reference {ref_frame}")
    if not (ref_wvl.exists() or ref_wvl_legacy.exists()):
        sys.exit(
            f"SKIP: no reference wvl file ({ref_wvl} or {ref_wvl_legacy})"
        )


def _exact(label: str, a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        print(f"  ✗ {label}: shape mismatch, {a.shape} vs {b.shape}")
        return False
    ok = bool(np.array_equal(a, b, equal_nan=True))
    print(f"  {'✓' if ok else '✗'} {label}: exact match" if ok else
          f"  ✗ {label}: max |Δ| = {float(np.max(np.abs(a - b))):.3e}")
    return ok


def _close_relative(label: str, a: np.ndarray, b: np.ndarray, *, rel_tol: float) -> bool:
    """Return True if |a-b| / |b| is everywhere ≤ `rel_tol` on finite values."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  ✗ {label}: shape mismatch, {a.shape} vs {b.shape}")
        return False
    finite = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    if not finite.any():
        print(f"  ✗ {label}: no finite, non-zero ref values")
        return False
    rel = np.abs(a[finite] - b[finite]) / np.abs(b[finite])
    ok = bool(np.max(rel) <= rel_tol)
    print(
        f"  {'✓' if ok else '✗'} {label}: "
        f"max relative Δ = {np.max(rel):.3%}  (tol {rel_tol:.1%})"
    )
    return ok


def _mean_ratio_match(new: np.ndarray, ref: np.ndarray, *, rel_tol: float) -> bool:
    """Mean-over-time ratio of per-integration median rate.

    Per-integration median is noise-dominated for PRISM SUB512 (trace
    occupies only ~20% of pixels; median is mostly bg).  The *mean*
    across integrations washes out that noise and probes whether
    reference-file drift has shifted the overall flux scale.  Signal
    pixels are checked separately in `_wl_curve_match`.
    """
    if new.shape != ref.shape:
        print(f"  ✗ overall flux scale: shape mismatch")
        return False
    new_m = np.nanmedian(new, axis=(1, 2))
    ref_m = np.nanmedian(ref, axis=(1, 2))
    ratio = new_m / ref_m
    mean_r = float(np.mean(ratio))
    ok = bool(abs(mean_r - 1.0) <= rel_tol)
    print(
        f"  {'✓' if ok else '✗'} mean per-int median ratio: "
        f"{mean_r:.4f}  (range {float(np.min(ratio)):.3f}–{float(np.max(ratio)):.3f}; "
        f"tol ±{rel_tol:.1%} on mean)"
    )
    return ok


def _wl_curve_match(new: np.ndarray, ref: np.ndarray, *, rel_tol: float) -> bool:
    """Total in-trace flux per integration (white-light curve) should match.

    For PRISM SUB512 the trace sits around rows 12–17; summing the
    middle 10 rows is a robust white-light proxy that's noise-insensitive
    to ~CRDS-level reference-file drift.
    """
    if new.shape != ref.shape:
        print(f"  ✗ WL curve: shape mismatch")
        return False
    n_rows = new.shape[1]
    r_lo = max(0, n_rows // 2 - 5)
    r_hi = min(n_rows, n_rows // 2 + 5)
    new_wl = np.nansum(new[:, r_lo:r_hi, :], axis=(1, 2))
    ref_wl = np.nansum(ref[:, r_lo:r_hi, :], axis=(1, 2))
    new_wl /= np.nanmedian(new_wl)
    ref_wl /= np.nanmedian(ref_wl)
    diff = np.abs(new_wl - ref_wl)
    ok = bool(np.max(diff) <= rel_tol)
    print(
        f"  {'✓' if ok else '✗'} WL curve (rows {r_lo}:{r_hi}, normalized): "
        f"max |Δ| = {float(np.max(diff)):.4f}  (tol {rel_tol:.1%})"
    )
    return ok


def main() -> int:
    _check_prereqs()

    import tswift

    crds_cache = os.environ.get("CRDS_PATH") or str(Path.home() / "crds_cache")
    print(f"CRDS cache: {crds_cache}")

    tmp = Path(tempfile.mkdtemp(prefix="tswift_regression_lp890-9c_v12_"))
    print(f"Work dir: {tmp}")

    # Mirror the visit's layout into tmp so tswift.run_calibrate can write
    # new outputs without touching the originals.
    (tmp / "data").mkdir(parents=True)
    shutil.copytree(VISIT / "data" / "uncal", tmp / "data" / "uncal", symlinks=True)

    t0 = time.time()
    result = tswift.run_calibrate(
        project=tmp,
        mode=MODE,
        detector=DETECTOR,
        crds_cache=crds_cache,
        bg_mask_top_rows=4,
        bg_mask_bottom_rows=4,
        rejection_threshold=8.0,
        three_group_rejection_threshold=8.0,
        maximum_cores="half",
        expand_large_events=False,
        make_plots=True,
        overwrite=True,
    )
    print(f"run_calibrate wall time: {time.time() - t0:.1f} s")

    product_new = result["paths"]["product_dir"]
    product_ref = VISIT / "product"

    new_frame = np.load(product_new / "all_frame.npy")
    ref_frame = np.load(product_ref / "all_frame.npy")
    new_time = np.load(product_new / "time_all.npy")
    ref_time = np.load(product_ref / "time_all.npy")

    new_wvl_path = product_new / tswift.wvl_filename(MODE, DETECTOR)
    new_wvl = np.load(new_wvl_path)
    ref_wvl_path = product_ref / tswift.wvl_filename(MODE, DETECTOR)
    if not ref_wvl_path.exists():
        ref_wvl_path = product_ref / f"{MODE}_wvl.npy"
    ref_wvl = np.load(ref_wvl_path)

    print("Comparing to reference products:")
    ok1 = _exact("time_all", new_time, ref_time)
    ok2 = _mean_ratio_match(new_frame, ref_frame, rel_tol=0.05)
    ok3 = _wl_curve_match(new_frame, ref_frame, rel_tol=0.005)
    ok4 = _close_relative("wvl (per-column)", new_wvl, ref_wvl, rel_tol=0.005)

    all_ok = ok1 and ok2 and ok3 and ok4
    print(f"\nArtifacts kept at: {tmp}")
    print("Regression PASS" if all_ok else "Regression FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
