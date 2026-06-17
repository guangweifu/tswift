# tswift pipeline runbook

**Who this is for:** an agent (or human) asked to produce a transmission spectrum
for a JWST transit observation. Read top to bottom. Each stage tells you exactly
what to call, what to check, and what to do when something looks wrong.

The tswift Python API is a set of composable functions — there is no one-call
`run_pipeline()`. The agent is the orchestrator. That is intentional: you decide
which stages to rerun, inspect intermediate diagnostics, and apply fixes.

Reference target throughout this doc: **WASP-69 b SOSS (program 5924)**. The
intermediate products and figures in `WASP-69b/` are ground truth.

---

## 0. Prerequisites

Before starting any pipeline work:

- [ ] Activate the env: `conda activate jwst_latest` (has jwst, batman, emcee, exotic_ld, astroquery).
- [ ] Verify CRDS cache path: `echo $CRDS_PATH` (defaults to `~/crds_cache/`).
- [ ] Verify LD data path exists: `/Users/guangweifu/Documents/Work/exotic_ld_data`.
- [ ] Verify tswift is installed: `python -c "import tswift; print(tswift.__version__)"`.

If any of these fail, stop and fix them — the pipeline can't recover from missing
calibration data.

---

## 1. Stage-by-stage overview

```
                  ┌──────────────────┐
                  │  bootstrap()     │  planet + program → project dir + target.json
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │  fetch()         │  MAST → data/uncal/
                  └────────┬─────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  run_calibrate()                     │  uncal → stage1 → bgsub → rampfit
        │    (or composable:                    │  → wvl.npy + all_frame.npy + time_all.npy
        │     calibrate_stage1, _bg_subtract,   │
        │     _rampfit, _wvl, build_frame_time) │
        └────────┬─────────────────────────────┘
                 ▼
        ┌──────────────────┐
        │  mad_clip()      │  all_frame.npy → data_fixed (NaN bad pixels)
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  run_extract()   │  data_fixed → trace_fit + extract_2D + clean_2D
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  fit_wl_mcmc()   │  white-light curve → (a, inc, t0) best-fit
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  compute_ld_... │  per-wavelength limb darkening
        │  fit_spec_curves │  per-wavelength Rp/Rs
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ combine_spectrum │  rebin to 10/20/50 nm
        │ save_spectrum    │  figure + txt files
        └──────────────────┘
```

Every stage's outputs should be inspected before moving on. The pipeline takes
a couple minutes per target; reruns are cheap.

### Diagnostic figures — where they live and what to look for

Every compute function has a paired `plot_*` function that writes a review PNG.
**Always call the plot function after the compute step** (run.py in the target
dir does this automatically for the bridge workflow). A summary:

| Stage | Plot function | Output PNG | What to look for |
|-------|---------------|------------|------------------|
| bad_pixel | `plot_bad_pixel()` | `bad_pixel.png` | Flag rate heatmap should NOT show wavelength banding (that = chromatic bias). σ-vs-signal should trend with √signal. |
| extract/trace | `plot_trace()` | `trace.png` | Trace line sits at spatial PSF peak. Aperture dashes comfortably contain PSF wings. |
| extract/aperture | `plot_aperture_scan()` | `aperture.png` | OOT-scatter-vs-width has a clear minimum (bowl). Spatial profile with aperture shaded covers the core. White-light curve shows baseline flat + transit dip. |
| extract/clean | `plot_clean()` | `clean.png` | Before/after curves overlap (cleaner just picks off outliers, doesn't smooth real signal). |
| WL MCMC | `plot_wl_fit()` | `corner.png`, `best_fit.png`, `chain.png` | Every posterior is a Gaussian blob (no rails, no flats). Best-fit residuals are white noise. Chain traces stationary after burn-in. |
| spec_fit | `plot_spec_fit()` | `spec_fit.png` | Rp(λ) has no isolated 10σ outliers. Residual RMS is at the photon floor. LD coefs smooth in λ (no huge jumps at grid edges). |
| combine | `save_spectrum()` | `spectrum.png` | Binned points sit on native scatter; no detector-boundary offset; y-axis is physical. |

Figures go to `<product_dir>/figure/<plot>.png` or `<outdir>/spectrum.png` for
the final combine. **If any figure shows an anomaly, fix the knob for THAT
stage and rerun downstream — don't move on.**

---

## 2. Bootstrap — create the project dir

**Call:**
```python
from tswift import bootstrap
project = bootstrap("WASP-69 b", program="5924", outdir="./WASP-69 b")
```

**Inputs:**
- `planet`: canonical name WITH space (e.g. `"WASP-69 b"`, not `"WASP-69b"`).
- `program`: JWST proposal id (strips `GO-`, `DD-`, `GTO-` prefixes).

**Outputs:**
- `<outdir>/target.json` — system params from NASA Exoplanet Archive pscomppars.
- `<outdir>/config.yaml` — pipeline overrides stub (empty mode; you fill it in after fetch).
- `<outdir>/data/`, `<outdir>/product/`, `<outdir>/manifests/`, `<outdir>/figures/`, `<outdir>/logs/`.

**Sanity checks:**
- [ ] `target.json` exists and has nonzero stellar.teff_k, orbital.period_days, planet.rp_rs_initial.
- [ ] Values match your expectation (eyeball against literature).

**Common pitfalls:**
- *TargetNotFoundError*: the exact planet name isn't in pscomppars. Usually a missing space
  (`WASP-69b` vs `WASP-69 b`). Consult
  [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) for the canonical name.
- *NASA EA rate limits*: rare. Retry after a minute.
- *Missing Rp/Rs*: some young/discovery-paper planets lack `pl_ratror` and `pl_radj`;
  set `planet.rp_rs_initial` manually in `target.json`.

---

## 3. Fetch data from MAST

**Call:**
```python
from tswift import list_program_products, fetch

# inspect first (cheap — no download)
prods = list_program_products("5924", target_name="WASP-69", instrument="NIRISS")
# then fetch (expensive — gigabytes)
links = fetch(project, program_id="5924", target_name="WASP-69", instrument="NIRISS",
              filename_contains="_04102_")   # science exposures only, not target acq
```

**Inputs:**
- `target_name` — MAST's naming is usually the host star without the planet letter
  (e.g. `"WASP-69"`, `"HAT-P-11"`).
- `instrument` — substring match. `"NIRISS"` matches `"NIRISS/SOSS"`.
- `filename_contains` — the 5-digit observation code lets you grab just the transit
  exposures and skip target acquisition. Get this from `list_program_products` output.

**Outputs:**
- `<project>/data/MAST_Download/` — astroquery's cache.
- `<project>/data/uncal/` — symlinks into the cache so the pipeline finds them.

**Sanity checks:**
- [ ] File count matches what you expected (SOSS ~10 segments, G395H ~10–20 per detector).
- [ ] No `_uncal.fits` over 600 MB — if it is, something wrong with the MAST query.

**Common pitfalls:**
- *Instrument filter returns 0*: MAST uses full names like `NIRISS/SOSS`, but tswift's
  prefix match should accept `"NIRISS"`. If you passed the exact MAST string it also works.
- *Target filter returns 0*: try `list_program_products(program_id)` (no target filter)
  and look at the `target_name` field of the results; MAST's naming may surprise you.
- *Large cache*: 10+ GB per transit. Make sure you have disk.

---

## 4. Calibration — uncal → ramp + wavelength

**One-shot call:**
```python
from tswift import run_calibrate
result = run_calibrate(
    project,
    mode="PRISM",            # "PRISM" | "G395H" | "SOSS" | "MIRI_LRS"
    detector="nrs1",         # "nrs1"/"nrs2" | "nis" | "mirimage"
    crds_cache="~/crds_cache/",   # optional; honors $CRDS_PATH if set
    # bg-mask knobs (mode-dependent; only the relevant ones are used)
    bg_mask_top_rows=4, bg_mask_bottom_rows=4,      # PRISM
    count_threshold=250.0, exclude_bottom_rows=4,   # SOSS
    bg_cols_left=(0, 22), bg_cols_right=(50, 72),   # MIRI LRS
    # rampfit knobs
    rejection_threshold=6.0,
    expand_large_events=False,      # disable for BOTS; 20× speedup
    maximum_cores="half",
    make_plots=True,
)
# result["paths"]["product_dir"] has:
#   all_frame.npy, time_all.npy, <mode>_<det>_wvl.npy
# plus figure/bg_mask_<det>.png, figure/rampfit_<det>.png,
#      figure/stage2_wavelength_solution.png
```

**Composable API** (for iteration / debugging):
```python
from tswift import (
    calibrate_stage1,    # uncal → data/stage1/*.fits (Detector1, jump+rampfit skipped)
    calibrate_bg_subtract, # stage1 → data/group_bg/*.fits (mode-dispatched mask)
    calibrate_rampfit,   # group_bg → data/ramp/*.fits (JumpStep + RampFitStep)
    calibrate_wvl,       # ramp → product*/<mode>_<det>_wvl.npy
    build_frame_time,    # ramp → product*/all_frame.npy + time_all.npy
)
```

**Mode dispatch** happens inside each function. The key difference across
modes is the **bg mask**:

| Mode | Mask method | Knobs |
|------|-------------|-------|
| PRISM | edge rows (top + bottom N) | `bg_mask_top_rows`, `bg_mask_bottom_rows` |
| G395H | PSF-centroid trace + polynomial fit | none (auto) |
| SOSS | count-threshold on CDS image | `count_threshold`, `exclude_bottom_rows` |
| MIRI LRS | per-row median over off-trace column bands | `bg_cols_left`, `bg_cols_right` |

NIRSpec/NIRISS subtract a per-column bg; MIRI LRS subtracts per-row.

**Sanity checks after running:**
- [ ] `product/figure/bg_mask_<det>.png`: mask covers full trace, bg region
      has no residual flux. For SOSS, min bg pixels/column ≥ 5.
- [ ] `product/figure/rampfit_<det>.png`: median rate image shows clean
      trace, per-integration flux curve has a transit-shaped dip.
- [ ] `product/figure/stage2_wavelength_solution.png`: monotonic,
      matches expected range for the mode.
- [ ] `all_frame.npy.shape` = (n_frames, n_rows, n_cols), no all-NaN frames.

**Gotchas worth remembering** (see CLAUDE.md "Lessons learned" for the full
set):
- **`expand_large_events=False`** is the default for BOTS time series.
  Snowball flagging is single-threaded and adds 20× to JumpStep wall time.
- **`AssignWcsStep` failing with "No open slits fall on detector …" is
  often correct** — for PRISM S1600A1 the spectrum may only land on one
  detector. Check the other detector before assuming a bug.
- **Calibration thresholds are tuned for post-Stage-1 ADU levels**, not
  raw uncal (~150–250 ADU bias). Validate bgsub PNGs on `data/stage1/`
  outputs, not `data/uncal/`.

**Rerunning a single stage.** Every function is idempotent when its output
already exists (pass `overwrite=True` to force). Tune `bg_mask_top_rows`,
rerun only `calibrate_bg_subtract` + `calibrate_rampfit` + `build_frame_time`,
reinspect the PNG — the iterative workflow the pipeline is built around.

---

## 5. Bad-pixel masking

**Call:**
```python
import numpy as np
from tswift import mad_clip

data_all = np.load("product/all_frame.npy")
data_fixed, bad_mask, sigma_pix = mad_clip(data_all, n_sigma=5.0, min_sigma=2.0)
```

**Inputs:**
- `n_sigma` (default 5.0): robust-σ threshold. 5σ is a reasonable default for JWST.
- `min_sigma` (default 2.0): ADU floor on per-pixel σ. Prevents flagging in quiet regions.
- `replace_with_nan` (default True): NaNs flow through `nansum` correctly downstream.

**Outputs (in-memory):**
- `data_fixed`: same shape as input, flagged samples → NaN.
- `bad_mask`: boolean, True where flagged.
- `sigma_pix`: (n_rows, n_cols), the per-pixel σ estimate.

**Sanity checks:**
- [ ] `bad_mask.mean()` is small (typical 0.001–1%). If > 5%, threshold is too tight.
- [ ] `data_fixed` looks like the input frame (no catastrophic holes).

**Common pitfalls — READ THIS:**
- **NEVER use an absolute ADU threshold.** The old pipeline had `threshold=100` which
  flagged 2.76σ of noise at the stellar SED peak but 14.5σ in the wings. Result:
  massively chromatic flag rate, transit depth silently distorted by ~hundreds of
  ppm at the bright band. MAD-robust is the only sane option.
- *Flagging too aggressive (> 5%)*: raise `n_sigma` to 6–7. You should NOT lower
  `min_sigma` below ~1.0 — that lets true noise rescue pixels from flagging.
- *Flagging too loose (no visible effect)*: probably `min_sigma` set way too high.
  Default 2.0 is tuned for SOSS; NIRSpec may want 5–10 on bright detectors.

---

## 6. Extraction — trace + aperture optimize + clean

**Call:**
```python
from tswift import run_extract

result = run_extract(
    data_fixed,
    mode="SOSS",                       # or "G395H"; "PRISM" not ported yet
    detector="nis",                    # "nrs1"/"nrs2" for G395H, "nis" for SOSS
    trace_half_width=22,               # ±22 rows for SOSS (full PSF + margin)
    trace_poly_order=5,                # 5 for SOSS (trace curves), 3 for NIRSpec
    trace_outlier_clip=4.0,
    aperture_criterion="per_channel",  # "wl_rms" or "per_channel"; see pitfalls below
    wavelength_left=50,                # trim unreliable columns
    wavelength_right=2040,
    outlier_window=10,
    outlier_threshold=5.0,
)
# result["trace_fit"]       (n_cols,)
# result["extract_2D"]      (n_frames, 2*half_width, n_cols)  wide aperture cube
# result["clean_2D"]        (n_frames, n_cols)  final per-channel light curves
# result["aperture"]        (up, down)  best sub-aperture within half_width window
# result["oot_mask"]        (n_frames,) bool for OOT baseline
```

**Sanity checks:**
- [ ] `result["trace_fit"]` monotonic/smooth — print `np.nanmin/max/median`.
- [ ] `result["aperture"]` width ≥ 10 rows and ≤ 2×half_width. A tiny width (< 5)
      means trace is drifting or aperture optimizer is avoiding contamination.
- [ ] White-light OOT RMS: `np.nanstd(np.nansum(clean_2D[:, 50:2040], axis=1)[oot_mask] / median)`
      should be within ~2× the photon noise floor. For SOSS WASP-69 b, ~200 ppm
      is the floor.

**Common pitfalls — READ THIS:**
- *Trace centroid off-center vertically*: visible in `step4_trace_extraction.png`-style
  figures (you should make one from `result["trace_fit"]` overlaid on `np.nanmedian(data_fixed, axis=0)`).
  Fix: widen `trace_half_width` to 30–40; the brute-force aperture optimizer will narrow
  back to the real PSF.
- *Aperture optimizer picks very narrow (< 5) row range*: usually means order-2
  contamination is leaking in at wider apertures and the `per_channel` criterion is
  running away from it. Correct behavior for SOSS 1.0–1.4 µm. If instead you see a
  very NARROW aperture across the WHOLE bandpass, the trace is mis-centered —
  rerun with a wider `trace_half_width`.
- *`wl_rms` vs `per_channel`*: for SOSS, use `per_channel` (penalizes order-2
  contamination). `wl_rms` minimizes white-light photon noise and happily includes
  order-2 for signal — good for NIRSpec, wrong for SOSS.

---

## 7. White-light MCMC fit

**Call:**
```python
from tswift import fit_wl_mcmc
from exotic_ld import StellarLimbDarkening

# Bandpass limb darkening (scalar for the band — not per-channel)
sld = StellarLimbDarkening(M_H, Teff, logg, "stagger", LD_DATA_PATH, verbose=False)
u1, u2 = sld.compute_quadratic_ld_coeffs(
    [wvl[left] * 1e4, wvl[right-1] * 1e4], "JWST_NIRISS_SOSSo1"
)

# OOT indices INTO THE FITTED array, not the raw array
oot_full = ... # first + last 15%
keep = ~bad_frame_mask
oot_fit = np.where(oot_full[keep])[0]

result = fit_wl_mcmc(
    time_data_hr=time_hr[keep],
    flux_data=wl_normalized[keep],
    flux_err=np.ones(keep.sum()) * oot_scatter,
    period_hr=orbital_period_days * 24,
    ecc=0.0, omega=90.0,
    u1=u1, u2=u2,
    oot_indices=oot_fit,
    initial=initial_dict,
    priors=priors_dict,
    fit_ld1=False,
    nwalkers=48, nsteps=8000, nburn=3000,
)
# result["best_params"]    (ndim,)   MAP (argmax log_prob) — pitfall #25
# result["median_params"]  (ndim,)   marginal median (reference)
# result["best_errors"]    (ndim, 2) [lower_err, upper_err] around MAP
# result["log_prob"]       (nsamples,) log-prob of each flat sample
# result["samples"]        flat chain (post-burn)
# result["chain"]          raw (nsteps, nwalkers, ndim)
# result["acceptance"]     per-walker
# result["rms_residual"]   residual std after best fit
# result["best_fit_curve"] model evaluated at data times
```

**Parameter order** (when `fit_ld1=False`):
`[slope, rp, LD2, constant, a, inc, t0_offset]`

With `fit_ld1=True`, LD1 slides in at index 2: `[slope, rp, LD1, LD2, constant, a, inc, t0_offset]`.

**Sanity checks:**
- [ ] Acceptance fraction 0.2–0.5 across all walkers.
- [ ] Corner plot: posteriors are GAUSSIAN BLOBS, not bimodal/flat/railed. Especially
      `t0_offset` — if it's flat, STOP, see the pitfall below.
- [ ] Best-fit RMS is within 2× the OOT scatter.
- [ ] `rp` posterior center within prior bounds — if rp hit the upper/lower bound,
      widen the prior.

**Common pitfalls — READ THIS:**
- **Flat posterior on `t0_offset`**: this was the bug the `tswift.transit_model`
  module is specifically built to prevent. If you see it, DO NOT assume it's a prior
  problem. Run `python -m pytest tswift/tests/test_t0_cache.py` — if it fails, you
  broke the cache design (see `transit_model.py` docstring). If it passes, then the
  prior is too tight or the data genuinely doesn't constrain t0 (rare).
- *Priors too tight*: posterior bumps against a prior wall. Widen by 2–3×.
- *Priors too wide*: multimodal posterior (two blobs in corner plot) means walkers
  got stuck in a local minimum. Re-initialize walkers from a tighter ball around the
  MAP estimate from a previous run.
- *Bad integrations driving the fit*: after step 6 (extraction), a few frames with
  ramp settling or post-transit anomalies can dominate the fit. Inspect the WL curve
  for outliers before ingress, and add their indices to `mask_indices` in config.

---

## 8. Per-wavelength spectral fit

**Call:**
```python
from tswift import compute_ld_per_wavelength, fit_spec_curves

u1_arr, u2_arr = compute_ld_per_wavelength(
    wvl, wavelength_left, wavelength_right,
    stellar_teff=4792, stellar_logg=4.57, stellar_mh=0.35,
    ld_model="stagger", ld_mode="JWST_NIRISS_SOSSo1",
    ld_data_path=LD_DATA_PATH,
)

sr = fit_spec_curves(
    clean_2D, time_hr, wvl,
    wavelength_left=50, wavelength_right=2040,
    period_days=orbital_period,
    a_over_rs=a_from_wl, inclination_deg=inc_from_wl, t0_offset_hr=t0_from_wl,
    ecc=0.0, omega=90.0,
    u1_arr=u1_arr, u2_arr=u2_arr,
    bounds_lower=(-0.005, 0.04, 0.995, -0.5),
    bounds_upper=( 0.005, 0.15, 1.005,  0.5),
    fix_ld2=False,
    mask_indices=bad_frame_indices,
)
# sr["fit"]            (n_wvl, 4)  [slope, rp, constant, LD2]  per wavelength
# sr["fit_err"]        (n_wvl, 4)  1-σ uncertainties
# sr["residuals_rms"]  (n_wvl,)    residual std per wavelength
```

**Sanity checks:**
- [ ] `sr["fit"][left:right, 1]` (Rp/Rs) has no catastrophic outliers (> 10σ vs neighbors).
- [ ] `sr["residuals_rms"][left:right]` is smooth in wavelength — not chromatic bumps.
- [ ] Count of successful fits = `right - left` (should be 100%). A few fails at
      grid edges are OK.

**Common pitfalls:**
- *Inverted/bumpy spectrum around 1.0–1.4 µm (SOSS) or near stellar absorption lines*:
  often an LD coefficient issue. Try `fix_ld2=True` — this drops LD2 from the fit
  and uses the exotic_ld stagger value per wavelength, eliminating the rp–LD2 degeneracy.
- *LD grid edge warnings*: exotic_ld's stagger model doesn't cover every wavelength.
  `compute_ld_per_wavelength` fills with nearest-neighbor, which is better than the
  legacy fallback of (0.5, 0.1). If a huge fraction is filled, the LD model is wrong
  for this star — try `ld_model="phoenix"`.
- *Systematically offset Rp/Rs vs WL fit*: the WL fit used a single-band LD; the
  spectral fit uses per-channel. A ~0.001 offset is normal. If > 0.005, check that
  the WL bandpass matches `[wavelength_left, wavelength_right)`.

---

## 8b. Bad-column repair (mandatory, run after `spec`)

Stable hot pixels in the aperture (constant ~5-10 kADU/sec) survive
`mad_clip` because that operates on time-domain outliers only.
They bias the per-channel transit depth: when the planet transits,
the stellar flux drops but the hot pixel's contribution stays the
same, so the apparent depth shifts.  The shift is positive or
negative depending on where the hot pixel sits in the aperture
relative to the trace, but is always >100 ppm at typical brightness.

**Call:**
```python
from tswift import repair_outlier_columns, plot_repair_diagnostics

repair = repair_outlier_columns(
    clean_2D, data_fixed, trace_fit, oot_mask, wvl, time_hr,
    spec_fit, spec_err, u1_per_wvl, u2_per_wvl,
    geom={"a": ..., "inc": ..., "t0_offset": ...},
    period_days=..., aperture=(ap_lo, ap_hi),
    trace_half_width=22,
    detection_n_sigma=5.0,            # outlier-channel threshold
    pixel_n_sigma_value=5.0,          # per-pixel value vs neighbour cols
    pixel_n_sigma_var=3.0,            # per-pixel time-variability
    post_repair_n_sigma=4.0,          # mask if repair leaves > Nσ outlier
    protected_wvl_um=[1.0833],        # SOSS: protect He I (real signal)
)
np.save("product/clean_2D.npy",   repair.clean_2D_repaired)
np.save("product/spec_fit.npy",   repair.spec_fit_repaired)
np.save("product/spec_fit_err.npy", repair.spec_err_repaired)
plot_repair_diagnostics(repair, wvl, outdir="product/figure",
                        planet_name=..., detector=...)
```

**Outputs:**
- `figure/bad_column_repair.png` — before/after spectrum panels with
  repaired (orange) and masked (red ✕) columns flagged.
- `figure/bad_column_repair.json` — per-column report: original
  depth, σ, identified bad rows, action (repaired/masked), final
  depth.
- Updated `clean_2D.npy`, `spec_fit.npy`, `spec_fit_err.npy` —
  downstream `combine` automatically picks up the cleaned spectrum.

**Sanity checks:**
- [ ] `n_repaired + n_masked` is small (typically <10 channels per
      detector at 5σ on bright targets; up to ~30 on faint).  If
      hundreds of channels are flagged, the detection threshold is
      too tight or the 2D bad-pixel clip in stage 5 was too loose.
- [ ] `bad_column_repair.png` after panel: flagged points sit
      cleanly inside the local-median band.
- [ ] He I 1083 nm column on SOSS is NOT in the flagged list.  If
      it is, your `protected_wvl_um` is missing or the search
      half-width is too narrow.

**Common pitfalls:**

- *Real molecular features flagged*: H₂O bands at 1.4 / 1.9 µm and
  CO₂ at 4.3 µm are smooth (they affect many channels at once), so
  the rolling-median compares each channel to its smoothed neighbours
  and won't flag them.  Sharp features like He I 1083 nm DO need
  protection — add to `protected_wvl_um`.
- *No bad pixel diagnosed but channel is outlier*: usually a snowball
  event affecting multiple rows at different times.  The fallback
  is to mask the entire column, which the module does automatically.
- *Repair makes the depth WORSE*: rare; happens if the bad pixel
  detection picked the wrong row (e.g. the trace-center row got
  flagged because the neighbour columns happened to have low values
  due to a curving trace).  The `post_repair_n_sigma` safety-check
  catches this and falls back to column-masking.

---

## 9. Combine + save

**Call:**
```python
from tswift import combine_spectrum, save_spectrum

combined = combine_spectrum(
    wvl,
    sr["fit"][:, 1],              # rp
    sr["fit_err"][:, 1],          # rp_err
    bin_widths_nm=[10, 20, 50],
    bad_wavelengths_um=[...],     # optional: list of µm to mask
)

paths = save_spectrum(combined, outdir="product/spectrum", planet_name="WASP-69 b")
# writes:
#   product/spectrum/spectrum.png
#   product/spectrum/spectrum_native.txt
#   product/spectrum/spectrum_10nm.txt  (and _20nm, _50nm)
#   product/spectrum/spectrum_summary.json
```

**Sanity checks:**
- [ ] `spectrum.png` opens and the binned points sit inside the native scatter cloud.
- [ ] Mean depth matches `(Rp/Rs)² * 1e6` from the WL fit within ~1%.
- [ ] No huge jumps at detector boundaries (for multi-detector stitching).

**Common pitfalls:**
- *Native scatter >> expected photon noise*: usually a bad-pixel threshold issue —
  loop back to §5 and tighten `n_sigma`.
- *Outlier bin at a specific wavelength*: add that wavelength (in µm) to
  `bad_wavelengths_um` and rerun combine. Common cause: contamination from another
  spectral order or a known detector defect.
- *Detector stitching offset (G395H)*: the nrs1 and nrs2 trails should overlap in
  depth within ~100 ppm. Bigger offsets indicate calibration (BG subtraction or
  wavelength solution) drift per detector — rerun calibration stage.

---

## 9a. (SOSS only, mandatory) He I 1083 nm pixel-level check

**Always run after `spec` for SOSS targets.**  He I 1083 nm is the
standard near-IR atmospheric escape tracer; you want to know if it
shows up before publishing the spectrum.

**Call:**
```python
from tswift import check_helium_1083, plot_helium

result = check_helium_1083(
    clean_2D, time_hr, wvl,
    rp=spec_fit[:, 1], rp_err=spec_err[:, 1],
    oot_mask=oot_mask, wl_left=..., wl_right=...,
    t0_hr=wl_best_params["t0_offset"],
)
plot_helium(result, outdir="product/helium", planet_name="Planet X b")
```

**Outputs (`product/helium/`):**

- `helium_pixel_spectrum.png` — per-pixel depth zoom around 1083 nm.
  A clean detection looks like one isolated pixel sticking out 5+σ
  above the local continuum band.
- `helium_lightcurves.png` — He pixel LC overplotted on WL with a
  residual panel.  In-transit residual ≈ -depth(He pixel − WL); pre
  and post should sit near zero if there's no tail.
- `helium_summary.json` — column, depth, excess significance,
  pre/post/in-transit residuals under both normalizations.

**Sanity checks (read the plots):**

- [ ] Pixel-level zoom shows one prominent excess pixel (or no
      excess).  If multiple adjacent pixels are excess, suspect a
      detector-systematic — investigate before claiming detection.
- [ ] In-transit residual is negative (channel − WL < 0) by an amount
      consistent with `excess_above_continuum_ppm`.
- [ ] Pre-transit residual sits near zero.  If it slopes up toward
      ingress, the He absorption may extend across a longer baseline
      than the WL transit and the per-channel normalization is
      contaminated.
- [ ] |post − pre residual| < 100 ppm.  If larger, check if it's an
      He tail by overplotting ``he_lc_balanced`` (pre+post norm)
      vs ``he_lc_local`` — if balanced norm gives shallower depth,
      the local norm is biased low by a post-transit tail.

**Common pitfalls:**

- *SOSS WCS offset (~1 nm)*: max-excess pixel lands at 1081-1082 nm
  rather than 1083 nm.  This is calibration, not science.  Default
  search half-width 5 nm covers it.
- *Wide bin masks the line*: never look for He at >5 nm bin width.
  The line is intrinsically narrow; binning to 10 or 50 nm dilutes
  the excess by 10–50× into the continuum.  Always look at native
  per-pixel depths.
- *Tail contamination of the spec-fit OOT*: if `post − pre` is
  significantly negative on the He pixel residual, the per-channel
  spec fit is using a contaminated baseline and reporting too-shallow
  depth at He.  Refit that single channel with pre-transit-only OOT
  to recover.

---

## 9b. (Optional) Limb-asymmetry analysis with catwoman

**Skip unless you specifically want morning vs evening limb spectra.**
Most science programs use only the symmetric `combine_spectrum` output
above.  This stage retrieves rp1 (evening / leading limb) and rp2
(morning / trailing limb) per bin via a per-bin catwoman MCMC.

**Call:**
```python
from tswift import (
    make_uniform_bin_edges, fit_limb_asymmetry,
    plot_limb_asymmetry, save_limb_asymmetry,
)

# Geometry must come from the joint white-light fit — DON'T re-fit per bin.
geom = {"a": 7.231, "inc": 88.39, "t0_offset": 4.505}

bin_lo, bin_hi = make_uniform_bin_edges(2.78, 3.72, 100)   # 100 nm bins

result = fit_limb_asymmetry(
    clean_2D, time_hr, wvl, bin_lo, bin_hi,
    geom=geom, period_days=3.95012,
    u1_per_wvl=u1_arr, u2_per_wvl=u2_arr,
    rp_mean_init=0.106, oot_mask=oot_mask,
    fit_ld1=False,                  # CLAUDE.md pitfall #21
    drp_range=(-0.05, 0.05),
    nwalkers=32, nsteps=2000, nburn=600,
)
save_limb_asymmetry(result, outdir="product/limb_asymmetry")
plot_limb_asymmetry(result, outdir="product/limb_asymmetry",
                    planet_name="Planet X b", detector="nis")
```

**Outputs (per detector):**
- `limb_asymmetry/limb_spectra.txt` — per-bin table: rp_mean, drp,
  rp1²/rp2² (ppm), Δdepth, errors, RMS.
- `limb_asymmetry/limb_asymmetry.npz` — full result dict.
- `limb_asymmetry/morning_evening_spectra.png` — rp1² (blue) and
  rp2² (red).
- `limb_asymmetry/delta_depth_vs_wavelength.png` — Δdepth with
  weighted-mean band.
- `limb_asymmetry/drp_vs_wavelength.png` — radius difference.
- `limb_asymmetry/symmetric_depth_vs_wavelength.png` — rp_mean²
  cross-check.

**Convention:** catwoman `phi=90` ⇒ rp1 = evening (leading), rp2 =
morning (trailing).  Δdepth = rp2² − rp1² is depth(morning) −
depth(evening).

**Sanity checks:**
- [ ] `morning_evening_spectra.png`: both spectra trace the symmetric
      `combine_spectrum` continuum within their error bars.
- [ ] Δdepth scatter in OOT-noise floor regions matches the per-bin
      photon-noise expectation (typically a few × 100 ppm for hot
      Jupiter targets).
- [ ] Wavelength-averaged Δ should be small (<<1σ from zero) unless
      the target genuinely has integrated asymmetry — most don't.

**Common pitfalls:**
- *drp prior too narrow*: if the posterior rails the drp_range, widen.
  Default ±0.05 covers ±5 % of rp_mean for a hot Jupiter; widen to
  ±0.1 for low-S/N or grazing geometries.
- *catwoman `fac` too coarse*: default 0.001 → ~50 ppm sampling
  noise per integration.  For sub-ppm-precision asymmetry detections
  drop to 0.0001 (~10× slower).
- *Geometry drift*: re-using a per-detector geometry instead of the
  joint one re-introduces the per-detector bias the joint WL fit
  removed (pitfall #27).  Always pass joint geometry.

---

## 9c. Red-noise diagnostics (mandatory, last stage)

Every reduction ends with a red-noise check.  Quoted error bars
assume white noise, but JWST residuals routinely have β = σ_obs / σ_white >
1.5–3 at 30-min binning, so the assumption silently inflates retrieved
significances unless you check.

**Call:**
```python
from tswift import (
    compute_wl_red_noise, plot_wl_red_noise,
    compute_spec_red_noise, plot_spec_red_noise,
    reconstruct_wl_residuals_from_curve,
)

t_kept, residuals = reconstruct_wl_residuals_from_curve(
    clean_2D, time_hr, oot_mask, best_fit_curve,
    wl_left=..., wl_right=...,
)
wl_result = compute_wl_red_noise(t_kept, residuals, detector="nis")
plot_wl_red_noise(wl_result, outdir="product/red_noise",
                  planet_name="Planet X b")

spec_result = compute_spec_red_noise(
    clean_2D, time_hr, wvl, spec_fit,
    geom={"a": ..., "inc": ..., "t0_offset": ...},
    period_days=..., u1_per_wvl=..., u2_per_wvl=...,
    oot_mask=oot_mask,
)
plot_spec_red_noise(spec_result, outdir="product/red_noise",
                    planet_name="Planet X b")
```

**Outputs:**

- `red_noise/red_noise.png` — 6-panel WL diagnostic.
- `red_noise/red_noise_summary.json` — WL β @ 5/15/30/60 min, ACF
  lag-1, PSD α, skew, kurtosis.
- `red_noise/spec_red_noise.png` — five rows (RMS, β @ 30 min,
  ACF lag-1, PSD α, kurtosis) all vs wavelength.
- `red_noise/spec_red_noise.npz` — per-channel arrays.
- `red_noise/spec_red_noise_summary.json` — wavelength-averaged
  metrics.

**Sanity checks:**

- [ ] WL `β @ 30 min` < 2.  If 2–3, the published error bars are
      ~2× too tight at retrieval cadence.  If > 3, detrending was
      probably insufficient.
- [ ] σ-vs-bin curve sits within a factor of ~2 of the dashed
      white-noise line at all bin sizes.  A wide gap at 30+ min is
      classic detector 1/f.
- [ ] ACF lag-1 < 95 % CI ≈ ±0.05 for a typical N=1500 visit.
- [ ] Q-Q plot is straight.  Curvature in the tails ⇒ outliers
      (re-check the bad-pixel and column repair stages).
- [ ] Sliding RMS in panel 6 doesn't spike during ingress/egress.
      If it does, the WL fit is mis-modelling LD.

**Common pitfalls:**

- *Per-channel β much higher than WL β*: usually fine.  Per-channel
  residuals are dominated by photon noise and even a small persistent
  correlated component looks large in fractional terms.  WL averages
  thousands of channels down to the systematic floor.  Look at the
  *trend* of spec β vs λ — flat is fine; sharp peaks at specific
  wavelengths are the warning sign.
- *β below 1 at long bins*: walker over-fitting in the WL MCMC has
  absorbed real long-timescale variance into the slope/constant
  parameters.  Not a bug per se, but it means quoted depth errors
  are slightly too tight.
- *Q-Q deviates only at ±3σ but lag-1 is fine*: residual outliers,
  not red noise.  Re-check `bad_col_repair` and the time-domain
  bad-pixel clip in stage 5.

---

## 10. Troubleshooting decision tree

**Spectrum has excess scatter (noise much higher than expected):**
1. Check `bad_pixel` threshold — tighten if flagging < 0.001% or loosen if > 5%.
2. Check aperture width — if narrow (< 10 rows), widen `trace_half_width` and rerun §6.
3. Check OOT baseline — is it actually out of transit? `find_ingress_index` might be wrong.

**Spectrum has an unphysical feature at a specific wavelength:**
1. Is it at a detector boundary (G395H)? → stitching issue, recalibrate.
2. Is it at 1.0–1.4 µm (SOSS)? → order-2 contamination. Try `aperture_criterion="per_channel"`.
3. Is it at a known stellar line? → LD issue, try `fix_ld2=True` or different `ld_model`.
4. Isolated single bin? → add to `bad_wavelengths_um` and rebin.

**WL MCMC corner plot shows flat `t0_offset` posterior:**
1. Run `pytest tswift/tests/test_t0_cache.py`. If failing, DO NOT patch around it —
   the transit_model cache design is broken. See `transit_model.py` docstring.
2. If tests pass, widen `t0_offset` prior by 2–3×. Ensure initial value isn't
   at the prior boundary.

**WL MCMC doesn't converge (multimodal, wide, or railed):**
1. Widen priors → `[0.12, 0.145]` became `[0.10, 0.16]`, etc.
2. Check initial values are inside priors.
3. Increase `nsteps`/`nburn` — 8000/3000 should be enough; if it doesn't help,
   the model is underdetermined (probably need tighter mask_indices).

**Depth at one channel looks wrong but neighbors are fine:**
1. Check `sr["residuals_rms"][i]` — is it an outlier?
2. Check LD coefficients at that channel — `u1_arr[i]`, `u2_arr[i]` shouldn't be NaN.
3. Add `wvl[i]` to `bad_wavelengths_um` and rebin.

---

## 11. When in doubt — regression tests

Run these to verify you haven't broken anything at a known-good target:

```bash
# 1. t0 cache design (< 1 s)
/opt/anaconda3/envs/jwst_latest/bin/python tswift/tests/test_t0_cache.py

# 2. Bit-identical WASP-69 b extraction + fit + combine (~2.5 min)
/opt/anaconda3/envs/jwst_latest/bin/python tswift/scripts/test_end_to_end_wasp69b.py
```

**If either fails, stop and fix before proceeding.** The WASP-69 b regression is
the canonical "the pipeline still works" check — its v1 outputs are exactly what
v2 should reproduce.

---

## 12. What goes in target.json vs config.yaml

**target.json** (auto-populated from NASA EA; regenerate with `bootstrap` if you
change source):
- Stellar: Teff, log g, [Fe/H], radius, mass
- Orbital: period, t0, a/Rs prior, inclination prior, eccentricity
- Planet: Rp/Rs initial, Mp, Rp, Teq

**config.yaml** (human/agent editable — this is where tuning happens):
- `extraction.trace_half_width`, `.aperture_criterion`, `.wavelength_left/right`
- `bad_pixel.n_sigma`, `.min_sigma`
- `mcmc.nwalkers`, `.nsteps`, `.nburn`, `.initial`, `.priors`, `.mask_indices`, `.fit_ld1`
- `curve_fit.bounds`, `.fix_ld2`
- `combine.bin_widths_nm`, `.bad_wavelengths`
- `paths.crds_cache`, `.ld_data`

**Never put system parameters in config.yaml.** If pscomppars is wrong for your
target, override in target.json (and note the override source — the agent will
see this via `target.source`).

---

## 13. Where to find deeper context

- **Project-level agent guide (architecture, folder layout, lessons learned):**
  [`../CLAUDE.md`](../CLAUDE.md)
- **Per-target reduction quirks:** `<Planet>/NOTES.md` in each planet folder.
- **Reference reductions:** `WASP-69b/` (SOSS), `HAT-P-11 b/` (G395H
  multi-visit), `LP 890-9/{b,c}/` (PRISM multi-visit).
