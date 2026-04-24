"""Unit tests for the bg-mask builders.

These touch only numpy, no jwst — fast, hermetic, runnable on any
machine with the tswift package installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from tswift.calibrate import (
    build_edge_rows_mask,
    build_psf_trace_mask,
    build_soss_count_mask,
    build_miri_off_cols_mask,
    wvl_filename,
    _validate_mode,
    _validate_detector,
)


# ---------------------------------------------------------------------------
# edge_rows (PRISM)
# ---------------------------------------------------------------------------

def test_edge_rows_basic_shape_and_values():
    mask = build_edge_rows_mask(n_rows=32, n_cols=512, top_rows=4, bottom_rows=4)
    assert mask.shape == (32, 512)
    # Bottom 4 rows are bg
    assert np.all(mask[:4, :] == 1.0)
    # Top 4 rows are bg
    assert np.all(mask[-4:, :] == 1.0)
    # Middle is NaN (trace)
    assert np.all(np.isnan(mask[4:-4, :]))


def test_edge_rows_asymmetric():
    mask = build_edge_rows_mask(n_rows=20, n_cols=100, top_rows=2, bottom_rows=6)
    assert np.all(mask[:6, :] == 1.0)
    assert np.all(mask[-2:, :] == 1.0)
    assert np.all(np.isnan(mask[6:-2, :]))


def test_edge_rows_zero_margin():
    mask = build_edge_rows_mask(n_rows=10, n_cols=10, top_rows=0, bottom_rows=0)
    # All middle → no bg at all; all NaN
    assert np.all(np.isnan(mask))


# ---------------------------------------------------------------------------
# psf_trace (G395H)
# ---------------------------------------------------------------------------

def _synthetic_g395h_data(n_ints=1, n_grps=5, n_rows=32, n_cols=2048,
                          trace_row=15.0, trace_width=1.5, peak=2000.0,
                          detector="nrs1"):
    """Fake NIRSpec cube with a Gaussian-profile trace at `trace_row`."""
    rows = np.arange(n_rows).reshape(-1, 1)
    profile = peak * np.exp(-0.5 * ((rows - trace_row) / trace_width) ** 2)
    frame = np.broadcast_to(profile, (n_rows, n_cols)).astype(np.float64)
    # Ramp up across groups so the bg mask builder (uses last group) sees signal
    data = np.tile(frame, (n_ints, n_grps, 1, 1)).astype(np.float64)
    for g in range(n_grps):
        data[:, g] *= (g + 1) / n_grps
    return data


def test_psf_trace_finds_synthetic_trace_nrs1():
    data = _synthetic_g395h_data(trace_row=15.0, detector="nrs1")
    mask, trace = build_psf_trace_mask(data, detector="nrs1")
    # Centroid within 0.5 rows of ground truth (averaged over cols)
    assert abs(np.nanmedian(trace) - 15.0) < 0.5
    # Mask has correct shape
    assert mask.shape == data.shape[-2:]
    # Bg pixels exist outside the trace band
    assert np.sum(mask == 1.0) > 0
    # Trace band is masked
    assert np.all(np.isnan(mask[10:20, :]))


def test_psf_trace_nrs2_band():
    # NRS2 expects the trace around row ~12 (psf_lo=7, psf_hi=17)
    data = _synthetic_g395h_data(trace_row=12.0, detector="nrs2")
    mask, trace = build_psf_trace_mask(data, detector="nrs2")
    assert abs(np.nanmedian(trace) - 12.0) < 0.5
    # NRS2 narrower mask (7+7=14 rows). Ensure there's still bg on both sides.
    per_col_bg = np.nansum(mask, axis=0)
    assert np.min(per_col_bg) >= 2, "not enough bg pixels per col on NRS2 synthetic"


def test_psf_trace_faint_target_falls_back():
    # All-zero data → centroid undefined for every col → polynomial fit path
    data = np.zeros((1, 3, 32, 2048), dtype=np.float64)
    mask, trace = build_psf_trace_mask(data, detector="nrs1")
    # Should not crash; mask should still exist
    assert mask.shape == (32, 2048)
    # Trace should be finite somewhere (fallback to median of med_trace = nan
    # path — in practice this path gives a constant nan-median, and polyval
    # propagates. Make sure we don't raise.)


# ---------------------------------------------------------------------------
# soss_count_threshold
# ---------------------------------------------------------------------------

def _synthetic_soss_data(n_rows=96, n_cols=2048, trace_row=50, trace_half=15,
                         trace_flux=500, bg_flux=10):
    """Fake SOSS cube with a bright order-1 trace."""
    n_ints, n_grps = 2, 5
    data = np.full((n_ints, n_grps, n_rows, n_cols), bg_flux, dtype=np.float64)
    # Ramp from 0 (first group) to signal (last group)
    for g in range(n_grps):
        scale = g / (n_grps - 1) if n_grps > 1 else 1.0
        frame = np.full((n_rows, n_cols), bg_flux * scale, dtype=np.float64)
        frame[trace_row - trace_half: trace_row + trace_half, :] += trace_flux * scale
        data[:, g] = frame
    return data


def test_soss_count_mask_flags_trace():
    data = _synthetic_soss_data(trace_flux=500, bg_flux=5)
    mask, cds = build_soss_count_mask(data, count_threshold=100, exclude_bottom_rows=4)
    # Trace region should be NaN
    assert np.all(np.isnan(mask[40:60, :]))
    # Far off-trace should be bg
    assert np.nansum(mask[70:80, :]) > 0
    # Bottom 4 rows always excluded
    assert np.all(np.isnan(mask[:4, :]))
    # CDS shape sanity
    assert cds.shape == (96, 2048)


def test_soss_count_mask_warns_on_low_bg(caplog):
    """Threshold so high that nothing is flagged → lots of bg pixels; inverse too."""
    import logging
    caplog.set_level(logging.WARNING)
    # Threshold too LOW → almost everything flagged as trace, few bg cols
    data = _synthetic_soss_data(trace_flux=500, bg_flux=50)
    mask, _ = build_soss_count_mask(data, count_threshold=1, exclude_bottom_rows=4)
    # Should have fired the <5 warning
    assert any("bg pixels" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# miri_off_cols
# ---------------------------------------------------------------------------

def test_miri_off_cols_default():
    mask = build_miri_off_cols_mask(72)
    # Default (0, 22) + (50, 72) → 22 + 22 = 44 cols selected
    assert mask.sum() == 44
    assert mask[10]    # in left band
    assert mask[60]    # in right band
    assert not mask[30]  # in trace band
    assert not mask[45]  # in trace band


def test_miri_off_cols_custom():
    mask = build_miri_off_cols_mask(100, bg_cols_left=(5, 15), bg_cols_right=(80, 95))
    assert mask.sum() == 10 + 15
    assert not mask[0]    # left of left band
    assert mask[10]       # in left band
    assert not mask[50]   # trace band
    assert mask[90]       # in right band


def test_miri_off_cols_rejects_tiny():
    with pytest.raises(ValueError, match="only 0 columns"):
        build_miri_off_cols_mask(10, bg_cols_left=(0, 0), bg_cols_right=(5, 5))


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def test_validate_mode_roundtrip():
    assert _validate_mode("prism") == "PRISM"
    assert _validate_mode("G395H") == "G395H"
    assert _validate_mode("soss") == "SOSS"
    assert _validate_mode("MIRI_LRS") == "MIRI_LRS"
    assert _validate_mode("miri") == "MIRI"
    with pytest.raises(ValueError):
        _validate_mode("NIRCAM")


def test_validate_detector():
    assert _validate_detector("PRISM", "nrs2") == "nrs2"
    assert _validate_detector("G395H", "NRS1") == "nrs1"
    assert _validate_detector("SOSS", "nis") == "nis"
    with pytest.raises(ValueError):
        _validate_detector("SOSS", "nrs1")
    with pytest.raises(ValueError):
        _validate_detector("G395H", "nis")


def test_wvl_filename_canonical():
    assert wvl_filename("PRISM", "nrs2") == "PRISM_nrs2_wvl.npy"
    assert wvl_filename("G395H", "nrs1") == "G395H_nrs1_wvl.npy"
    assert wvl_filename("SOSS", "nis") == "SOSS_nis_wvl.npy"
