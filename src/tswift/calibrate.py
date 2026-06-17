"""Calibration — uncal.fits → ramp-fit + wavelength solution.

Wraps the `jwst` package to take raw `_uncal.fits` downloads through the
four calibration stages the rest of tswift expects:

1. **Stage 1**  (`calibrate_stage1`) — Detector1Pipeline up to (but not
   including) jump detection and ramp fitting.  Produces per-group
   calibrated 4-D ramps.
2. **Group-level background subtraction** (`calibrate_bg_subtract`) —
   mode-dispatched mask (edge-rows / PSF-centroid / SOSS count
   threshold / MIRI per-row off-cols) then per-group median subtraction.
3. **Ramp fit** (`calibrate_rampfit`) — JumpStep + RampFitStep with the
   BOTS-friendly `expand_large_events=False` default.
4. **Stage 2 wavelength** (`calibrate_wvl`) — NIRSpec `extract_2d_step`,
   NIRISS `assign_wcs` along the trace, or MIRI `assign_wcs` evaluated
   along the per-row trace centroid, depending on mode.

A final helper (`build_frame_time`) concatenates the per-segment ramp
FITS into the `all_frame.npy` + `time_all.npy` that the analysis layer
reads.  The convenience wrapper `run_calibrate(project, mode, detector)`
runs all five in order and emits diagnostic PNGs next to each stage's
output.

All functions write plain directories and numpy arrays — no pickled
models, no CRDS leakage — so partial runs are easy to inspect and
resume.  See RUNBOOK §4 for the calling pattern.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and helpers
# ---------------------------------------------------------------------------

_STAGE1_SUFFIX = "_stage1.fits"
_BGSUB_SUFFIX = "_stage1_bgsub.fits"
_RAMPFIT_SUFFIX = "_rampfit.fits"

_NIRSPEC_DETECTORS = ("nrs1", "nrs2")
_SOSS_DETECTORS = ("nis",)
_MIRI_DETECTORS = ("mirimage",)

_MODE_DETECTORS = {
    "PRISM": _NIRSPEC_DETECTORS,
    "G395H": _NIRSPEC_DETECTORS,
    "SOSS":  _SOSS_DETECTORS,
    "MIRI_LRS": _MIRI_DETECTORS,
}


def setup_crds_environment(
    crds_cache: Optional[str] = None,
    crds_server: str = "https://jwst-crds.stsci.edu",
) -> None:
    """Point CRDS at a local cache directory before importing `jwst.pipeline`.

    Idempotent — if `CRDS_PATH` is already set (e.g. in the user's shell)
    and `crds_cache` is None, this is a no-op.  Callers who provide
    `crds_cache` always override.

    Parameters
    ----------
    crds_cache : str, optional
        Path to the CRDS cache directory.  Expanded via `~` and
        `$VAR` before being written to `CRDS_PATH`.
    crds_server : str
        URL of the CRDS server.  Defaults to JWST public.
    """
    if crds_cache is not None:
        path = os.path.expandvars(os.path.expanduser(str(crds_cache)))
        Path(path).mkdir(parents=True, exist_ok=True)
        os.environ["CRDS_PATH"] = path
    os.environ.setdefault("CRDS_SERVER_URL", crds_server)


def default_paths(project: Path, detector: Optional[str] = None) -> dict:
    """Canonical calibration layout for a tswift project dir.

    All calibration functions accept explicit `*_dir` arguments, but
    when calling through `run_calibrate(project, ...)` these are the
    defaults.  G395H and other multi-detector modes keep products under
    `product_<detector>/`; single-detector modes use `product/`.
    """
    project = Path(project)
    if detector and detector in _NIRSPEC_DETECTORS:
        product_dir = project / f"product_{detector}"
    else:
        product_dir = project / "product"
    return {
        "uncal_dir":  project / "data" / "uncal",
        "stage1_dir": project / "data" / "stage1",
        "bgsub_dir":  project / "data" / "group_bg",
        "ramp_dir":   project / "data" / "ramp",
        "stage2_dir": project / "data" / "stage2",
        "product_dir": product_dir,
        "figure_dir":  product_dir / "figure",
    }


def _uncal_stem(path: Path) -> str:
    """`jw…_nis_uncal.fits` → `jw…_nis` (trailing _uncal stripped)."""
    return path.stem.replace("_uncal", "")


def discover_uncal(uncal_dir: Path, detector: str) -> list[Path]:
    """Return sorted list of `*_<detector>_uncal.fits` in `uncal_dir`.

    Excludes activity 02101 (target acquisition only).  `_04102_` is the
    science exposure activity code for BOTS time series (NIRSpec PRISM,
    NIRSpec G395H, NIRISS SOSS) and is kept.  Filter more selectively
    upstream via `tswift.fetch(..., filename_contains="_04102_")` if a
    program mixes observation types.
    """
    uncal_dir = Path(uncal_dir)
    files = sorted(uncal_dir.glob(f"*_{detector}_uncal.fits"))
    files = [f for f in files if "_02101_" not in f.name]
    if not files:
        raise FileNotFoundError(
            f"No *_{detector}_uncal.fits in {uncal_dir}. Run `tswift.fetch()` first."
        )
    return files


def _validate_mode(mode: str) -> str:
    mode = mode.upper()
    if mode in _MODE_DETECTORS:
        return mode
    if mode in ("MIRI", "MIRILRS", "MIRI_LRS"):
        return "MIRI_LRS"
    raise ValueError(
        f"Unknown instrument mode {mode!r}. Supported: PRISM, G395H, SOSS, MIRI_LRS"
    )


def _validate_detector(mode: str, detector: str) -> str:
    det = detector.lower()
    valid = _MODE_DETECTORS[mode]
    if det not in valid:
        raise ValueError(
            f"Detector {detector!r} not valid for mode {mode!r}. "
            f"Expected one of {valid}."
        )
    return det


def wvl_filename(mode: str, detector: str) -> str:
    """Canonical filename for the per-column wavelength solution."""
    return f"{mode}_{detector}_wvl.npy"


# ---------------------------------------------------------------------------
# Stage 1 — pre-jump, pre-rampfit calibration
# ---------------------------------------------------------------------------

def calibrate_stage1(
    uncal_dir: Path,
    stage1_dir: Path,
    mode: str,
    detector: str,
    *,
    crds_cache: Optional[str] = None,
    overwrite: bool = False,
) -> list[Path]:
    """Run the Detector1Pipeline, skipping JumpStep and RampFitStep.

    The resulting 4-D ramp (n_ints, n_groups, n_rows, n_cols) is saved
    as `<stem>_stage1.fits` in `stage1_dir`.  The bg-subtract stage
    reads those back.

    Skipping jump + ramp_fit is deliberate: background subtraction must
    happen BEFORE jump/ramp for group-level bg methods to work.

    Parameters
    ----------
    uncal_dir : Path
        Directory containing `*_<detector>_uncal.fits`.
    stage1_dir : Path
        Output directory.  Created if missing.
    mode, detector : str
        Instrument mode and detector name.  Validated against the
        known set.
    crds_cache : str, optional
        Passed to `setup_crds_environment`.
    overwrite : bool
        If False, skip exposures whose stage1 output already exists
        and has size > 0.  Idempotency by file hash is not yet
        implemented — delete the stage1 FITS to force a rerun.

    Returns
    -------
    list[Path]
        Output FITS in the same order as the matching uncal files.
    """
    mode = _validate_mode(mode)
    detector = _validate_detector(mode, detector)

    setup_crds_environment(crds_cache)
    from jwst.pipeline.calwebb_detector1 import Detector1Pipeline

    uncal_dir = Path(uncal_dir)
    stage1_dir = Path(stage1_dir)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    files = discover_uncal(uncal_dir, detector)
    logger.info(
        "Stage 1 [%s %s]: %d uncal file(s) → %s",
        mode, detector, len(files), stage1_dir,
    )

    outputs: list[Path] = []
    for i, uncal in enumerate(files, start=1):
        stem = _uncal_stem(uncal)
        out_path = stage1_dir / f"{stem}{_STAGE1_SUFFIX}"
        if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
            logger.info("  [%d/%d] %s — exists, skipping", i, len(files), out_path.name)
            outputs.append(out_path)
            continue
        logger.info("  [%d/%d] %s", i, len(files), uncal.name)

        result = Detector1Pipeline.call(
            str(uncal),
            save_results=False,
            steps={
                "jump":       {"skip": True},
                "ramp_fit":   {"skip": True},
                # gain_scale runs post-rampfit and reads attributes that the
                # skipped ramp_fit would have created — skip it too. The real
                # gain_scale call happens in Stage 2 (via calibrate_wvl).
                "gain_scale": {"skip": True},
            },
        )
        # jwst returns a RampModel — save it ourselves for a uniform filename.
        result.save(str(out_path))
        outputs.append(out_path)
        logger.info("    → %s", out_path.name)

    return outputs


# ---------------------------------------------------------------------------
# Background mask builders — one per mode
# ---------------------------------------------------------------------------

def _fill_nan_with_local_median(arr: np.ndarray, kernel: int = 10) -> np.ndarray:
    """Patch isolated NaNs with the median of a `kernel`-wide neighborhood."""
    out = np.copy(arr)
    nans = np.isnan(out)
    if not nans.any():
        return out
    for i, j in zip(*np.where(nans)):
        r_lo = max(0, i - kernel // 2)
        r_hi = min(arr.shape[0], i + kernel // 2)
        c_lo = max(0, j - kernel // 2)
        c_hi = min(arr.shape[1], j + kernel // 2)
        m = np.nanmedian(arr[r_lo:r_hi, c_lo:c_hi])
        if np.isfinite(m):
            out[i, j] = m
    return out


def build_edge_rows_mask(
    n_rows: int,
    n_cols: int,
    *,
    top_rows: int = 4,
    bottom_rows: int = 4,
) -> np.ndarray:
    """PRISM: treat top/bottom N rows as background, middle as trace.

    Returns a 2-D array with 1.0 in the bg region and NaN where the
    trace lives.  Column-wise median of (frame * mask) then gives the
    per-column background to subtract.
    """
    mask = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    mask[:bottom_rows, :] = 1.0
    mask[n_rows - top_rows:, :] = 1.0
    return mask


def build_psf_trace_mask(
    dark_data: np.ndarray,
    detector: str,
    *,
    mask_below: Optional[int] = None,
    mask_above: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """G395H: per-column flux-weighted centroid + polynomial fit + bg mask.

    Builds a 2-D background mask (1.0 in bg, NaN over the trace) by:

    1. Collapsing the 4-D Stage-1 cube `(n_ints, n_groups, n_rows, n_cols)`
       to a 2-D median `(n_rows, n_cols)` over both integrations and groups.
       This is much higher-SNR than the single frame `dark_data[0, -1]` we
       used to use, and it tracks the *time-median* trace (the same thing
       the analysis layer's `extract.find_trace` sees on the rate cube), so
       the bg-mask trace and the extraction trace stay aligned.

    2. Delegating per-column centroid + polynomial fit to the shared
       trace-finder `tswift.extract.find_trace`.  Same algorithm as the
       extraction stage uses, so `bg_mask` and the analysis trace can't
       silently drift apart.  CLAUDE.md pitfall #9 (mask must cover wings)
       and the "two-trace-finder divergence" lesson observed on WASP-94 A
       NRS2 (high-col trace excursion incorrectly rejected by global
       median-deviation outlier filter) drive this design.

    3. Symmetric `mask_below + mask_above`-row mask around `trace_fit`.

    Parameters
    ----------
    dark_data : 4-D array
        Stage-1 cube `(n_ints, n_groups, n_rows, n_cols)`.
    detector : str
        `nrs1` or `nrs2`.  Picks the polynomial column trim (`nrs1` has
        edge artifacts below col ~600 where there is no trace).

    Returns
    -------
    bg_mask : 2-D array
        1.0 where bg, NaN where trace.
    trace_fit : 1-D array
        Polynomial trace row per column.
    """
    # Time-median of the last (highest-signal) group across all integrations.
    # ~sqrt(n_ints) better SNR than the single frame `dark_data[0, -1]` we
    # used to use, and tracks the time-median trace — the same thing the
    # analysis layer's `find_trace` sees on the rate cube.  Using only the
    # last group avoids materializing a 5-GB intermediate that
    # `np.nanmedian(dark_data, axis=(0, 1))` would otherwise produce.
    frame = np.nanmedian(dark_data[:, -1, :, :], axis=0).astype(np.float64)
    frame = _fill_nan_with_local_median(frame)
    n_rows, n_cols = frame.shape

    if detector == "nrs1":
        poly_trim = 600 if n_cols > 1000 else 0
        mask_below = 9 if mask_below is None else mask_below
        mask_above = 9 if mask_above is None else mask_above
    else:
        poly_trim = 0
        mask_below = 7 if mask_below is None else mask_below
        mask_above = 7 if mask_above is None else mask_above

    # Use the same trace-finder the extraction stage uses, so bg-mask and
    # aperture stay tied to the same per-column centroids.
    from tswift.extract import find_trace
    poly_order = 0 if n_cols <= 512 else 3
    trace_fit = find_trace(
        frame,
        mode="G395H", detector=detector,
        poly_order=poly_order, outlier_clip=4.0,
    )
    if poly_trim > 0:
        trace_fit[:poly_trim] = trace_fit[poly_trim]
    logger.info(
        "PSF trace [%s]: poly order %d, row range %.2f–%.2f (median %.2f)",
        detector, poly_order,
        float(np.nanmin(trace_fit)), float(np.nanmax(trace_fit)),
        float(np.nanmedian(trace_fit)),
    )

    bg_mask = np.ones((n_rows, n_cols), dtype=np.float64)
    for c in range(n_cols):
        lo = max(0, int(trace_fit[c] - mask_below))
        hi = min(n_rows, int(trace_fit[c] + mask_above))
        bg_mask[lo:hi, c] = np.nan
    return bg_mask, trace_fit


def build_soss_count_mask(
    dark_data: np.ndarray,
    *,
    count_threshold: float = 250.0,
    exclude_bottom_rows: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """SOSS: any pixel brighter than `count_threshold` in the median CDS
    image is flagged as trace/contamination; everything else is bg.

    This naturally rejects orders 1, 2, 3 without needing one
    polynomial per order.  `exclude_bottom_rows` always removes the
    NIRISS reference-pixel band.

    Returns
    -------
    bg_mask : 2-D array
    cds_median : 2-D array
        CDS (last − first group) median over integrations — useful for
        the diagnostic plot.
    """
    cds = dark_data[:, -1, :, :].astype(np.float64) - dark_data[:, 0, :, :].astype(np.float64)
    cds_median = np.nanmedian(cds, axis=0)

    bg_mask = np.full_like(cds_median, np.nan)
    bg_mask[cds_median < count_threshold] = 1.0
    if exclude_bottom_rows > 0:
        bg_mask[:exclude_bottom_rows, :] = np.nan

    bg_per_col = np.nansum(bg_mask, axis=0)
    min_bg = int(np.nanmin(bg_per_col))
    logger.info(
        "SOSS bg mask: threshold=%.0f, median %.1f bg px/col, min %d",
        count_threshold, float(np.nanmedian(bg_per_col)), min_bg,
    )
    if min_bg < 5:
        logger.warning(
            "SOSS bg mask: some columns have <5 bg pixels — raise count_threshold"
        )
    return bg_mask, cds_median


def build_miri_off_cols_mask(
    n_cols: int,
    *,
    bg_cols_left: Sequence[int] = (0, 22),
    bg_cols_right: Sequence[int] = (50, 72),
) -> np.ndarray:
    """MIRI LRS: boolean mask over detector columns marking off-trace bg.

    Unlike the NIRSpec/NIRISS masks (per-column bg), MIRI LRS subtracts
    a per-ROW bg computed from the same set of off-trace columns in
    every row.  Returned is a 1-D boolean length `n_cols`.
    """
    mask = np.zeros(n_cols, dtype=bool)
    lo_lo, lo_hi = max(0, int(bg_cols_left[0])), min(n_cols, int(bg_cols_left[1]))
    hi_lo, hi_hi = max(0, int(bg_cols_right[0])), min(n_cols, int(bg_cols_right[1]))
    mask[lo_lo:lo_hi] = True
    mask[hi_lo:hi_hi] = True
    if mask.sum() < 4:
        raise ValueError(
            f"MIRI bg columns select only {mask.sum()} columns out of {n_cols}"
        )
    return mask


# ---------------------------------------------------------------------------
# Stage 2 — group-level background subtraction
# ---------------------------------------------------------------------------

def calibrate_bg_subtract(
    stage1_dir: Path,
    bgsub_dir: Path,
    mode: str,
    detector: str,
    *,
    # PRISM
    bg_mask_top_rows: int = 4,
    bg_mask_bottom_rows: int = 4,
    # SOSS
    count_threshold: float = 250.0,
    exclude_bottom_rows: int = 4,
    # MIRI LRS
    bg_cols_left: Sequence[int] = (0, 22),
    bg_cols_right: Sequence[int] = (50, 72),
    # plotting / introspection
    save_mask_path: Optional[Path] = None,
    overwrite: bool = False,
) -> dict:
    """Group-level background subtraction for stage1 FITS.

    Mode dispatch:
      - **PRISM**: edge-rows mask (`bg_mask_top_rows` + `bg_mask_bottom_rows`).
      - **G395H**: PSF-centroid mask from the first segment's last group.
      - **SOSS**: count-threshold mask on the CDS image.
      - **MIRI_LRS**: per-row median over `bg_cols_left ∪ bg_cols_right`.

    NIRSpec/NIRISS modes subtract a per-column bg (`median(frame * mask,
    axis=0)`); MIRI LRS subtracts a per-row bg.  This matches the
    geometry of each trace.

    Returns
    -------
    dict
        Keys: `mask` (2-D array or 1-D for MIRI), `trace_fit` (only
        G395H), `cds_median` (only SOSS), `outputs` (list of bgsub
        FITS paths).  Suitable for feeding into `plot_bg_subtract`.
    """
    mode = _validate_mode(mode)
    detector = _validate_detector(mode, detector)

    stage1_dir = Path(stage1_dir)
    bgsub_dir = Path(bgsub_dir)
    bgsub_dir.mkdir(parents=True, exist_ok=True)

    stage1_files = sorted(stage1_dir.glob(f"*_{detector}{_STAGE1_SUFFIX}"))
    if not stage1_files:
        raise FileNotFoundError(
            f"No *_{detector}{_STAGE1_SUFFIX} in {stage1_dir}. Run calibrate_stage1 first."
        )

    # Build the bg mask from the first segment's data.
    with fits.open(stage1_files[0]) as hdul:
        first_data = hdul["SCI"].data

    result: dict = {"trace_fit": None, "cds_median": None}

    if mode == "PRISM":
        n_rows, n_cols = first_data.shape[-2:]
        mask = build_edge_rows_mask(
            n_rows, n_cols,
            top_rows=bg_mask_top_rows,
            bottom_rows=bg_mask_bottom_rows,
        )
        result["mask"] = mask
        bg_axis = "column"  # per-column median
    elif mode == "G395H":
        mask, trace_fit = build_psf_trace_mask(first_data, detector)
        result["mask"] = mask
        result["trace_fit"] = trace_fit
        bg_axis = "column"
    elif mode == "SOSS":
        mask, cds_median = build_soss_count_mask(
            first_data,
            count_threshold=count_threshold,
            exclude_bottom_rows=exclude_bottom_rows,
        )
        result["mask"] = mask
        result["cds_median"] = cds_median
        bg_axis = "column"
    elif mode == "MIRI_LRS":
        n_cols = first_data.shape[-1]
        mask = build_miri_off_cols_mask(
            n_cols, bg_cols_left=bg_cols_left, bg_cols_right=bg_cols_right,
        )
        result["mask"] = mask
        bg_axis = "row"   # per-row median over off-trace cols
    else:
        raise AssertionError(f"unreachable mode: {mode}")

    if save_mask_path is not None:
        save_mask_path = Path(save_mask_path)
        save_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_mask_path, result["mask"])
        logger.info("Saved bg mask → %s", save_mask_path)

    # Apply the mask to each segment.
    outputs: list[Path] = []
    for i, stage1_path in enumerate(stage1_files, start=1):
        stem = stage1_path.stem.replace("_stage1", "")
        out_path = bgsub_dir / f"{stem}{_BGSUB_SUFFIX}"
        if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
            logger.info("  [%d/%d] %s — exists, skipping", i, len(stage1_files), out_path.name)
            outputs.append(out_path)
            continue
        logger.info("  [%d/%d] %s", i, len(stage1_files), stage1_path.name)

        with fits.open(stage1_path) as hdul:
            data = hdul["SCI"].data.astype(np.float64)   # (n_ints, n_grp, n_rows, n_cols)
            n_ints, n_grp = data.shape[:2]
            cleaned = np.empty_like(data, dtype=np.float32)

            if bg_axis == "column":
                # NIRSpec/NIRISS: per-(int, group) column median bg.
                mask_arr = result["mask"]  # 2-D
                for ii in range(n_ints):
                    for gg in range(n_grp):
                        bg_col = np.nanmedian(data[ii, gg] * mask_arr, axis=0)
                        cleaned[ii, gg] = (data[ii, gg] - bg_col[None, :]).astype(np.float32)
            else:
                # MIRI LRS: per-(int, group, row) median over off-trace cols.
                col_mask = result["mask"]  # 1-D bool
                for ii in range(n_ints):
                    for gg in range(n_grp):
                        # Index columns in a separate step: mixing a slice (`:`)
                        # with a boolean array in one index expression triggers
                        # numpy advanced indexing, which moves the masked axis to
                        # the front (giving (n_sel_cols, n_rows)) and breaks the
                        # per-row subtraction below.
                        bg_row = np.nanmedian(data[ii, gg][:, col_mask], axis=1)
                        cleaned[ii, gg] = (data[ii, gg] - bg_row[:, None]).astype(np.float32)

            hdul["SCI"].data = cleaned
            hdul.writeto(out_path, overwrite=True)
        outputs.append(out_path)

    result["outputs"] = outputs
    return result


# ---------------------------------------------------------------------------
# Stage 3 — jump detection + ramp fit
# ---------------------------------------------------------------------------

def calibrate_rampfit(
    bgsub_dir: Path,
    ramp_dir: Path,
    mode: str,
    detector: str,
    *,
    rejection_threshold: float = 6.0,
    three_group_rejection_threshold: float = 6.0,
    maximum_cores: str = "half",
    expand_large_events: bool = False,
    crds_cache: Optional[str] = None,
    overwrite: bool = False,
) -> list[Path]:
    """JumpStep + RampFitStep on bg-subtracted stage-1 FITS.

    `expand_large_events=False` is the important default: the snowball
    flagging sub-step is single-threaded and on BOTS TS runs adds 20×
    to JumpStep wall time.  Snowballs are rare in BOTS so skipping is
    safe (CLAUDE.md lesson #3).
    """
    mode = _validate_mode(mode)
    detector = _validate_detector(mode, detector)

    setup_crds_environment(crds_cache)
    from jwst.jump import JumpStep
    from jwst.ramp_fitting import RampFitStep

    bgsub_dir = Path(bgsub_dir)
    ramp_dir = Path(ramp_dir)
    ramp_dir.mkdir(parents=True, exist_ok=True)

    bgsub_files = sorted(bgsub_dir.glob(f"*_{detector}{_BGSUB_SUFFIX}"))
    if not bgsub_files:
        raise FileNotFoundError(
            f"No *_{detector}{_BGSUB_SUFFIX} in {bgsub_dir}. "
            "Run calibrate_bg_subtract first."
        )
    logger.info(
        "Rampfit [%s %s]: %d file(s) → %s (rej=%.1f, cores=%s, expand_events=%s)",
        mode, detector, len(bgsub_files), ramp_dir,
        rejection_threshold, maximum_cores, expand_large_events,
    )

    outputs: list[Path] = []
    for i, bgsub_path in enumerate(bgsub_files, start=1):
        stem = bgsub_path.stem.replace("_stage1_bgsub", "")
        out_path = ramp_dir / f"{stem}{_RAMPFIT_SUFFIX}"
        if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
            logger.info("  [%d/%d] %s — exists, skipping", i, len(bgsub_files), out_path.name)
            outputs.append(out_path)
            continue
        logger.info("  [%d/%d] %s", i, len(bgsub_files), bgsub_path.name)

        jumped = JumpStep.call(
            str(bgsub_path),
            save_results=False,
            rejection_threshold=rejection_threshold,
            three_group_rejection_threshold=three_group_rejection_threshold,
            maximum_cores=maximum_cores,
            expand_large_events=expand_large_events,
        )
        rate = RampFitStep.call(
            jumped,
            save_results=False,
            maximum_cores=maximum_cores,
        )
        # RampFitStep returns (rate, rateints) for multi-int observations;
        # for single-int it returns just rate.  We want rateints.
        if isinstance(rate, (tuple, list)):
            rate_model = rate[1]  # rateints: per-integration rate cube
        else:
            rate_model = rate
        rate_model.save(str(out_path))
        outputs.append(out_path)
        logger.info("    → %s", out_path.name)

    return outputs


# ---------------------------------------------------------------------------
# Stage 4 — wavelength solution
# ---------------------------------------------------------------------------

def _stage2_wvl_nirspec(
    ramp_files: list[Path],
    stage2_dir: Path,
    detector: str,
    mode: str,
    crds_cache: Optional[str] = None,
) -> np.ndarray:
    """NIRSpec: GainScale → AssignWcs → Extract2d; read WAVELENGTH ext."""
    from scipy import interpolate
    from jwst.pipeline import calwebb_detector1, calwebb_spec2

    stage2_dir = Path(stage2_dir)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # Run stage 2 on segment 1 only — wvl solution is shared across segments.
    ramp_file = ramp_files[0]
    logger.info("NIRSpec Stage 2 wvl from %s", ramp_file.name)

    gs = calwebb_detector1.gain_scale_step.GainScaleStep.call(str(ramp_file), save_results=False)
    wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep.call(gs, save_results=False)
    ext2 = calwebb_spec2.extract_2d_step.Extract2dStep.call(
        wcs, save_results=True, output_dir=str(stage2_dir),
    )

    # Find the Extract2d output and pull WAVELENGTH.
    ext2_pattern = ramp_file.stem + "_extract2dstep.fits"
    ext2_files = sorted(stage2_dir.glob(f"*{ramp_file.stem.split('_')[0]}*extract2dstep.fits"))
    # Prefer an exact stem match
    exact = [f for f in ext2_files if f.name.startswith(ramp_file.stem)]
    ext2_path = exact[0] if exact else (ext2_files[0] if ext2_files else None)
    if ext2_path is None:
        raise FileNotFoundError(
            f"Extract2dStep did not produce a file in {stage2_dir} matching {ext2_pattern!r}"
        )

    with fits.open(ext2_path) as hdul:
        wvl_2d = np.array(hdul["WAVELENGTH"].data)
        sltstrt1 = int(hdul["SCI"].header.get("SLTSTRT1", 1))

    slit_start = sltstrt1 - 1
    # Average 5 rows through the middle of the slit to beat noise in edge pixels.
    mid = wvl_2d.shape[0] // 2
    wvl_strip = np.nanmean(wvl_2d[max(0, mid - 2):mid + 3, :], axis=0)
    valid = np.where(np.isfinite(wvl_strip))[0]
    if valid.size == 0:
        raise ValueError("NIRSpec Extract2dStep emitted no finite WAVELENGTH values")

    det_width = 2048 if mode == "G395H" else 512
    x_src = valid + slit_start
    f = interpolate.interp1d(
        x_src, wvl_strip[valid], fill_value="extrapolate", kind="slinear",
    )
    wvl = f(np.arange(det_width))
    logger.info(
        "NIRSpec wvl: slit_start=%d, valid=%d–%d → %.4f–%.4f μm",
        slit_start, int(x_src[0]), int(x_src[-1]),
        float(wvl_strip[valid[0]]), float(wvl_strip[valid[-1]]),
    )
    return wvl


def _stage2_wvl_soss(
    ramp_files: list[Path],
    product_dir: Path,
    crds_cache: Optional[str] = None,
) -> np.ndarray:
    """SOSS: AssignWcs + evaluate per-column along order-1 trace."""
    from jwst.pipeline import calwebb_spec2

    ramp_file = ramp_files[0]
    logger.info("SOSS Stage 2 wvl from %s", ramp_file.name)

    output = calwebb_spec2.assign_wcs_step.AssignWcsStep.call(str(ramp_file), save_results=False)
    wcs = output.meta.wcs

    with fits.open(ramp_file) as hdul:
        rate = np.asarray(hdul["SCI"].data, dtype=float)
    if rate.ndim == 3:
        rate = np.nanmedian(rate, axis=0)  # collapse ints
    n_rows, n_cols = rate.shape

    # Trace: same finder as the analysis stage (`tswift.extract.find_trace`)
    # so the WCS is evaluated on the same row the extraction will use.
    # `mode="SOSS"` activates the centroid-mode default, which keeps the
    # trace anchored to the PSF center of mass instead of letting argmax
    # snap to whichever lobe of the double-peaked GR700XD profile is
    # momentarily brightest.
    from tswift.extract import find_trace
    trace_fit = find_trace(
        rate, mode="SOSS", detector="nis",
        poly_order=5, outlier_clip=4.0,
    )
    trace_fit = np.clip(trace_fit, 0, n_rows - 1)
    if not np.all(np.isfinite(trace_fit)):
        raise RuntimeError(
            "SOSS Stage 2: find_trace failed to return a finite trace fit."
        )

    # Save the Stage-2 trace fit for analysis use.
    product_dir = Path(product_dir)
    product_dir.mkdir(parents=True, exist_ok=True)
    np.save(product_dir / "soss_stage2_trace_fit.npy", trace_fit)

    # Evaluate WCS(x, y, order=1) → (ra, dec, wavelength) in μm.
    cols = np.arange(n_cols, dtype=float)
    try:
        _, _, lam = wcs(cols, trace_fit, np.ones_like(cols))
    except Exception as exc:
        logger.warning("Vector WCS call failed (%s), falling back to per-column", exc)
        lam = np.full(n_cols, np.nan)
        for c in range(n_cols):
            try:
                _, _, lc = wcs(float(c), float(trace_fit[c]), 1)
                lam[c] = lc
            except Exception:
                pass
    logger.info(
        "SOSS wvl: %.3f–%.3f μm, trace rows %.1f–%.1f",
        float(np.nanmin(lam)), float(np.nanmax(lam)),
        float(np.nanmin(trace_fit)), float(np.nanmax(trace_fit)),
    )
    return np.asarray(lam, dtype=float)


def _stage2_wvl_miri(
    ramp_files: list[Path],
    stage2_dir: Path,
    crds_cache: Optional[str] = None,
) -> np.ndarray:
    """MIRI LRS: AssignWcs, then evaluate the WCS along the per-row trace
    centroid to get the per-row wavelength grid (no Extract1d step)."""
    from jwst.pipeline import calwebb_spec2

    ramp_file = ramp_files[0]
    logger.info("MIRI LRS Stage 2 wvl from %s", ramp_file.name)

    stage2_dir = Path(stage2_dir)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    wcs_out = calwebb_spec2.assign_wcs_step.AssignWcsStep.call(
        str(ramp_file), save_results=False,
    )
    wcs = wcs_out.meta.wcs
    with fits.open(ramp_file) as hdul:
        rate = np.asarray(hdul["SCI"].data, dtype=float)
    if rate.ndim == 3:
        rate = np.nanmedian(rate, axis=0)
    n_rows, n_cols = rate.shape

    # MIRI LRS dispersion is along the ROW axis (y); wvl varies per row
    # around a ~fixed trace column.  Find per-row centroid column.
    trace_x = np.full(n_rows, np.nan)
    for r in range(n_rows):
        row = rate[r, :]
        if np.nanmax(row) > 0 and np.any(np.isfinite(row)):
            trace_x[r] = float(np.nanargmax(row))
    valid = np.isfinite(trace_x)
    if valid.sum() < 10:
        # Fall back to a central column
        trace_x = np.full(n_rows, n_cols // 2, dtype=float)
    rows = np.arange(n_rows, dtype=float)
    try:
        _, _, lam = wcs(trace_x, rows)
    except Exception:
        lam = np.full(n_rows, np.nan)
        for r in range(n_rows):
            try:
                _, _, lc = wcs(float(trace_x[r]), float(r))
                lam[r] = lc
            except Exception:
                pass
    logger.info(
        "MIRI LRS wvl: %.3f–%.3f μm (%d rows)",
        float(np.nanmin(lam)), float(np.nanmax(lam)), n_rows,
    )
    return np.asarray(lam, dtype=float)


def calibrate_wvl(
    ramp_dir: Path,
    product_dir: Path,
    stage2_dir: Path,
    mode: str,
    detector: str,
    *,
    crds_cache: Optional[str] = None,
) -> Path:
    """Save the per-column (or per-row for MIRI) wavelength solution.

    Writes `product_dir / <mode>_<detector>_wvl.npy`.  Returns the path.
    """
    mode = _validate_mode(mode)
    detector = _validate_detector(mode, detector)

    setup_crds_environment(crds_cache)

    ramp_dir = Path(ramp_dir)
    product_dir = Path(product_dir)
    product_dir.mkdir(parents=True, exist_ok=True)

    ramp_files = sorted(ramp_dir.glob(f"*_{detector}{_RAMPFIT_SUFFIX}"))
    if not ramp_files:
        raise FileNotFoundError(
            f"No *_{detector}{_RAMPFIT_SUFFIX} in {ramp_dir}. Run calibrate_rampfit first."
        )

    if mode in ("PRISM", "G395H"):
        wvl = _stage2_wvl_nirspec(ramp_files, stage2_dir, detector, mode, crds_cache)
    elif mode == "SOSS":
        wvl = _stage2_wvl_soss(ramp_files, product_dir, crds_cache)
    elif mode == "MIRI_LRS":
        wvl = _stage2_wvl_miri(ramp_files, stage2_dir, crds_cache)
    else:
        raise AssertionError(f"unreachable mode: {mode}")

    out_path = product_dir / wvl_filename(mode, detector)
    np.save(out_path, wvl)
    logger.info("Saved wavelength solution → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Build all_frame.npy + time_all.npy
# ---------------------------------------------------------------------------

def build_frame_time(
    ramp_dir: Path,
    product_dir: Path,
    detector: str,
    *,
    time_key: str = "INT_TIMES",
) -> tuple[Path, Path]:
    """Concatenate per-segment rateints → `all_frame.npy` + `time_all.npy`.

    Expects rateints-style output (SCI = (n_ints, n_rows, n_cols)) from
    RampFitStep.  Times come from the INT_TIMES extension (BJD, TDB).
    If that extension is missing (some older simulated data) the mid-
    integration time is inferred from the PRIMARY header `EXPSTART`
    and integration count.

    Returns
    -------
    (all_frame_path, time_all_path)
    """
    ramp_dir = Path(ramp_dir)
    product_dir = Path(product_dir)
    product_dir.mkdir(parents=True, exist_ok=True)

    ramp_files = sorted(ramp_dir.glob(f"*_{detector}{_RAMPFIT_SUFFIX}"))
    if not ramp_files:
        raise FileNotFoundError(
            f"No *_{detector}{_RAMPFIT_SUFFIX} in {ramp_dir}."
        )

    frames: list[np.ndarray] = []
    times:  list[np.ndarray] = []
    for p in ramp_files:
        with fits.open(p) as hdul:
            sci = np.asarray(hdul["SCI"].data, dtype=np.float32)
            if sci.ndim == 2:
                sci = sci[None, ...]
            frames.append(sci)

            if time_key in hdul:
                int_times = hdul[time_key].data
                # Prefer int_mid_BJD_TDB, fall back to int_mid_MJD_UTC
                for col in ("int_mid_BJD_TDB", "int_mid_MJD_UTC"):
                    if col in int_times.names:
                        times.append(np.asarray(int_times[col], dtype=np.float64))
                        break
                else:
                    raise KeyError(
                        f"INT_TIMES in {p.name} has no BJD/MJD column (got {int_times.names})"
                    )
            else:
                # Synthesize from header (rare path)
                hdr = hdul["PRIMARY"].header
                expstart_mjd = float(hdr.get("EXPSTART", np.nan))
                effint = float(hdr.get("EFFINTTM", hdr.get("TGROUP", 1.0)))
                n = sci.shape[0]
                times.append(expstart_mjd + (np.arange(n) + 0.5) * effint / 86400.0)

    all_frame = np.concatenate(frames, axis=0)
    time_all = np.concatenate(times, axis=0)

    frame_path = product_dir / "all_frame.npy"
    time_path = product_dir / "time_all.npy"
    np.save(frame_path, all_frame)
    np.save(time_path, time_all)
    logger.info(
        "Wrote %s shape=%s and %s len=%d (range %.4f–%.4f)",
        frame_path, all_frame.shape, time_path, len(time_all),
        float(time_all[0]), float(time_all[-1]),
    )
    return frame_path, time_path


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_calibrate(
    project: Path,
    mode: str,
    detector: str,
    *,
    crds_cache: Optional[str] = None,
    overwrite: bool = False,
    # bg-subtract knobs, passed through as-is
    bg_mask_top_rows: int = 4,
    bg_mask_bottom_rows: int = 4,
    count_threshold: float = 250.0,
    exclude_bottom_rows: int = 4,
    bg_cols_left: Sequence[int] = (0, 22),
    bg_cols_right: Sequence[int] = (50, 72),
    # rampfit knobs
    rejection_threshold: float = 6.0,
    three_group_rejection_threshold: float = 6.0,
    maximum_cores: str = "half",
    expand_large_events: bool = False,
    # plot
    make_plots: bool = True,
) -> dict:
    """End-to-end: uncal → stage1 → bgsub → rampfit → wvl → frame+time.

    Uses the default project layout (`default_paths`).  Returns a dict
    with keys pointing at each stage's outputs so the caller can fan
    out into diagnostics.
    """
    mode = _validate_mode(mode)
    detector = _validate_detector(mode, detector)

    paths = default_paths(project, detector)
    for d in paths.values():
        d.mkdir(parents=True, exist_ok=True)
    setup_crds_environment(crds_cache)

    result: dict = {"paths": paths, "mode": mode, "detector": detector}

    # 1. Stage 1
    result["stage1"] = calibrate_stage1(
        uncal_dir=paths["uncal_dir"],
        stage1_dir=paths["stage1_dir"],
        mode=mode, detector=detector,
        crds_cache=crds_cache, overwrite=overwrite,
    )

    # 2. Group-level background subtraction
    mask_save = paths["product_dir"] / f"{detector}_bg_mask.npy"
    result["bg"] = calibrate_bg_subtract(
        stage1_dir=paths["stage1_dir"],
        bgsub_dir=paths["bgsub_dir"],
        mode=mode, detector=detector,
        bg_mask_top_rows=bg_mask_top_rows,
        bg_mask_bottom_rows=bg_mask_bottom_rows,
        count_threshold=count_threshold,
        exclude_bottom_rows=exclude_bottom_rows,
        bg_cols_left=bg_cols_left, bg_cols_right=bg_cols_right,
        save_mask_path=mask_save,
        overwrite=overwrite,
    )

    # 3. Ramp fit
    result["ramp"] = calibrate_rampfit(
        bgsub_dir=paths["bgsub_dir"],
        ramp_dir=paths["ramp_dir"],
        mode=mode, detector=detector,
        rejection_threshold=rejection_threshold,
        three_group_rejection_threshold=three_group_rejection_threshold,
        maximum_cores=maximum_cores,
        expand_large_events=expand_large_events,
        crds_cache=crds_cache, overwrite=overwrite,
    )

    # 4. Wavelength
    result["wvl_path"] = calibrate_wvl(
        ramp_dir=paths["ramp_dir"],
        product_dir=paths["product_dir"],
        stage2_dir=paths["stage2_dir"],
        mode=mode, detector=detector,
        crds_cache=crds_cache,
    )
    result["wvl"] = np.load(result["wvl_path"])

    # 5. all_frame + time_all
    result["frame_path"], result["time_path"] = build_frame_time(
        ramp_dir=paths["ramp_dir"],
        product_dir=paths["product_dir"],
        detector=detector,
    )

    # Diagnostics
    if make_plots:
        plot_bg_subtract(
            result["bg"], mode, detector,
            fig_path=paths["figure_dir"] / f"bg_mask_{detector}.png",
        )
        plot_wvl(
            result["wvl"], mode, detector,
            fig_path=paths["figure_dir"] / "stage2_wavelength_solution.png",
        )
        plot_rampfit(
            paths["ramp_dir"], detector,
            fig_path=paths["figure_dir"] / f"rampfit_{detector}.png",
        )

    logger.info(
        "run_calibrate complete [%s %s]: %s",
        mode, detector, paths["product_dir"],
    )
    return result


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_bg_subtract(
    bg_result: dict,
    mode: str,
    detector: str,
    fig_path: Path,
) -> Path:
    """Diagnostic for `calibrate_bg_subtract`.

    Shows (for 2-D masks) the CDS image, the mask, a cleaned preview,
    and per-column bg-pixel count — so you can see at a glance whether
    the mask covers the trace and leaves enough bg pixels.  For MIRI
    LRS (1-D mask) shows the column coverage instead.
    """
    import matplotlib.pyplot as plt

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    mode = _validate_mode(mode)

    if mode == "MIRI_LRS":
        col_mask = bg_result["mask"]
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.fill_between(
            np.arange(len(col_mask)),
            0, col_mask.astype(float),
            color="tab:green", alpha=0.5, label="bg cols",
        )
        ax.set_xlabel("detector column"); ax.set_ylabel("bg (1 = used)")
        ax.set_title(f"MIRI LRS bg columns ({col_mask.sum()}/{len(col_mask)})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        return fig_path

    mask_2d = bg_result["mask"]
    cds = bg_result.get("cds_median")
    if cds is None:
        # Build a CDS proxy from the first bg-subtracted segment.  This is
        # cheap — 4-D data, last minus first group, median over ints.
        outs = bg_result.get("outputs", [])
        if outs:
            with fits.open(outs[0]) as hdul:
                d = hdul["SCI"].data
            cds = np.nanmedian(d[:, -1, :, :].astype(np.float64) - d[:, 0, :, :].astype(np.float64), axis=0)
        else:
            cds = np.zeros_like(mask_2d)

    cleaned = cds - np.nan_to_num(np.nanmedian(cds * mask_2d, axis=0))[None, :]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    ax = axes[0, 0]
    vmin, vmax = np.nanpercentile(cds, [5, 98])
    im = ax.imshow(cds, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax)
    trace = bg_result.get("trace_fit")
    if trace is not None:
        ax.plot(np.arange(len(trace)), trace, "r-", lw=0.7, label="trace")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"{mode} {detector}: CDS median (before bg sub)")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    ax = axes[0, 1]
    ax.imshow(mask_2d, aspect="auto", origin="lower", cmap="RdYlGn")
    ax.set_title("bg mask (green = bg, red = trace)")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    ax = axes[1, 0]
    cv = np.nanpercentile(cleaned, [5, 98])
    im = ax.imshow(cleaned, vmin=cv[0], vmax=cv[1], aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax)
    ax.set_title("CDS median (after bg sub)")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    ax = axes[1, 1]
    per_col = np.nansum(mask_2d, axis=0)
    ax.plot(per_col, lw=0.8)
    ax.axhline(5, color="r", ls="--", lw=0.8, label="warn < 5")
    ax.set_xlabel("column"); ax.set_ylabel("# bg pixels")
    ax.set_title("bg pixels per column")
    ax.legend(fontsize=8)

    fig.suptitle(f"Group-level bg subtraction — {mode} {detector}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_rampfit(
    ramp_dir: Path,
    detector: str,
    fig_path: Path,
) -> Path:
    """Summarize the ramp output: median rate image + per-integration flux."""
    import matplotlib.pyplot as plt

    ramp_dir = Path(ramp_dir)
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(ramp_dir.glob(f"*_{detector}{_RAMPFIT_SUFFIX}"))
    if not files:
        logger.warning("plot_rampfit: no ramp files found for %s", detector)
        return fig_path

    frames: list[np.ndarray] = []
    for p in files:
        with fits.open(p) as hdul:
            sci = np.asarray(hdul["SCI"].data, dtype=float)
            if sci.ndim == 2:
                sci = sci[None, ...]
            frames.append(sci)
    sci_all = np.concatenate(frames, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    med = np.nanmedian(sci_all, axis=0)
    vmin, vmax = np.nanpercentile(med, [5, 99])
    im = ax.imshow(med, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{detector}: median rate (per integration)")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    ax = axes[1]
    total_flux = np.nansum(sci_all, axis=(1, 2))
    ax.plot(total_flux / np.nanmedian(total_flux), lw=0.6)
    ax.set_xlabel("integration"); ax.set_ylabel("total flux / median")
    ax.set_title("per-integration total flux")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Rampfit output — {detector}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_wvl(
    wvl: np.ndarray,
    mode: str,
    detector: str,
    fig_path: Path,
) -> Path:
    """Wavelength-vs-column (or row, for MIRI LRS)."""
    import matplotlib.pyplot as plt

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(wvl, lw=1)
    xlab = "detector row" if mode == "MIRI_LRS" else "detector column"
    ax.set_xlabel(xlab); ax.set_ylabel("wavelength (μm)")
    ax.set_title(f"{mode} {detector} — Stage 2 wavelength solution")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


__all__ = [
    # core pipeline
    "calibrate_stage1",
    "calibrate_bg_subtract",
    "calibrate_rampfit",
    "calibrate_wvl",
    "build_frame_time",
    "run_calibrate",
    # mask builders (exposed for experimentation)
    "build_edge_rows_mask",
    "build_psf_trace_mask",
    "build_soss_count_mask",
    "build_miri_off_cols_mask",
    # helpers
    "setup_crds_environment",
    "default_paths",
    "discover_uncal",
    "wvl_filename",
    # plots
    "plot_bg_subtract",
    "plot_rampfit",
    "plot_wvl",
]
