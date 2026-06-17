# tswift — agent guide

**Goal:** turn a raw JWST transit observation into a transmission spectrum.

tswift is **AI-agent-native**: there is deliberately no `run_pipeline()`. You (the
agent) are the orchestrator — call one stage, **open the diagnostic PNG it wrote**,
decide whether to continue or adjust `config.yaml` and rerun. Most failures look
fine in the log and only become obvious in the plot, so the inspect-every-PNG loop
is the whole method, not an optional nicety.

This guide is the cross-target playbook. Pair it with:
- [`RUNBOOK.md`](RUNBOOK.md) — per-stage call signatures, return keys, and the
  "what the PNG should look like" checklist.
- [`examples/`](examples/) — runnable, config-driven driver templates
  (`analyze_soss.py`, `analyze_g395h.py`) you copy and adapt.
- [`README.md`](README.md) — install + data setup.

Supported modes: **NIRSpec PRISM, NIRSpec G395H, NIRISS SOSS, MIRI LRS.** Adding a
mode is a config branch, not a new script.

> **Scope today:** SOSS and G395H go end-to-end through the package
> (`run_extract` → spectrum). PRISM and MIRI LRS are supported in **calibration**
> but their extraction is not yet wired through `run_extract` (it raises
> `NotImplementedError` with a clear message). Plan SOSS/G395H reductions with
> this package as-is.

---

## The workflow

```bash
# 0. install + data caches (see README.md): pip install tswift,
#    set CRDS_PATH and EXOTIC_LD_DATA, download the exotic_ld grid.

# 1. inspect the program's observations (cheap, no download)
tswift list 1234 --target "Star" --instrument NIRSpec

# 2. create the project: target.json (NASA archive) + config.yaml stub
tswift bootstrap "Planet X b" 1234 --outdir "./Planet X b"

# 3. download the science exposures (narrow with --contains once you see obs ids)
tswift fetch "./Planet X b" 1234 --target "Star" --instrument NIRSpec --contains _04102_

# 4. set instrument.mode + paths in ./Planet X b/config.yaml, copy a driver
#    from examples/, then run stages ONE AT A TIME, inspecting each PNG:
python analyze_g395h.py "./Planet X b" extract     # then open product*/figure/*.png
python analyze_g395h.py "./Planet X b" wl          # ... and so on
```

Calibration (`run_calibrate`) takes `uncal.fits` → ramp + per-column wavelength and
writes `all_frame.npy`, `time_all.npy`, `<mode>_<det>_wvl.npy`, plus `bg_mask`,
`rampfit`, and `stage2_wavelength_solution` diagnostic PNGs. Then the analysis
stages: `mad_clip` → `run_extract` → white-light MCMC → per-wavelength fit →
bad-column repair → combine → (SOSS) helium → red-noise.

---

## Core principles

### 1. Iterative workflow — the most important rule
Run one stage, open its PNG (`product*/figure/` or `product/spectrum/`), apply the
RUNBOOK checklist, and only then continue. If something is wrong, identify the knob
in `config.yaml`, edit it, and rerun **only that stage + downstream**. When you add
code to a stage, add a diagnostic plot that makes the relevant choice visible —
one `savefig` now saves hours of debugging a bad spectrum later.

### 2. Config-driven, not code-forked
Stellar/orbital params (`target.json`), extraction windows, MCMC priors, and
calibration thresholds (`config.yaml`) are data. Instrument mode is dispatched at
runtime on `instrument.mode`. Never fork a file per mode.

### 3. Geometry is shared; depth and limb-darkening vary with color
The per-wavelength fit holds orbital geometry (`a`, `inc`, `t0`) fixed at the
white-light best fit and only lets depth + LD vary. So the white-light fit must be
trustworthy before you touch the spectrum.

---

## Mode comparison (where the modes differ)

| Aspect | PRISM | G395H | SOSS |
|--------|-------|-------|------|
| Instrument | NIRSpec | NIRSpec | NIRISS |
| Detectors | NRS1 **or** NRS2 (check!) | NRS1 + NRS2 | NIS only |
| Wavelength | 0.6–5.3 µm | NRS1 2.8–3.7, NRS2 3.8–5.2 | 0.85–2.85 µm (order 1) |
| Resolution | R ~ 100 | R ~ 1000 | R ~ 700 |
| Trace geometry | narrow, straight | medium, slight curve | wide ~30 rows, strong curve |
| `trace_poly_order` | 0 | 3 | 5 |
| BG method | edge rows | PSF-centroid mask | `count_threshold` |
| LD mode | `JWST_NIRSpec_PRISM` | `JWST_NIRSpec_G395H` | `JWST_NIRISS_SOSSo1` |
| MCMC LD fit | fix u1, fit u2 | fix u1, fit u2 | fix u1, fit u2 |
| White-light fit | single-detector | **joint** NRS1+NRS2 (`fit_wl_mcmc_joint`) | single-detector |

**PRISM BOTS S1600A1 can land on either NRS1 or NRS2** depending on the visit.
Download both detectors and verify with `mean(last_group − first_group)` over a few
hundred integrations before calibrating — one detector is often blank.

---

## Per-stage diagnostics (what the PNG should show, and the knob)

- **bg_mask** — mask symmetric around the trace and wide enough to cover the wings.
  Knobs: `bg_mask_top/bottom_rows` (PRISM), `count_threshold` (SOSS), `bg_cols_*`
  (MIRI).
- **rampfit / wavelength** — clean ramp; the wavelength solution monotonic and the
  right length for the detector.
- **trace** — the polynomial sits on the spectrum across all illuminated columns.
  Knobs: `trace_poly_order` (match the curvature), `wavelength_left/right`, and
  `restrict_trace_fit_to_window` when a large dark block would drag the global fit.
- **aperture_scan** — the chosen sub-aperture matches the criterion. SOSS:
  `per_channel` (order-2 contamination above 1 µm); NIRSpec: `wl_rms`.
- **wl corner / best_fit / chain** — Gaussian posteriors, **no railed priors**,
  flat residuals (no ingress/egress bumps = geometry off; no sinusoid = LD ok).
- **spec_fit** — Rp/Rs(λ) without sharp adjacent-channel inversions; residual-RMS
  near the photon floor; LD curves continuous.
- **spectrum** — physically plausible; spikes are candidates for
  `combine.bad_wavelengths`.
- **red_noise** — `β @ 30 min` near 1; values > 2 mean published error bars at
  retrieval resolution should be inflated; > 3 flags insufficient detrending.

---

## Pitfalls that look fine in logs

These bias the final spectrum and won't show up in residuals alone. The mitigations
are baked into the default code paths; keep the diagnostics for new targets.

### Bad-pixel masking
- **Never use absolute-ADU thresholds.** A fixed cut flags wildly different amounts
  at different wavelengths (Poisson scales with √signal), producing dense scatter in
  the SED-peak band or diluting depth elsewhere. Use **MAD-based `n_sigma`** (default
  5σ), replace with NaN, propagate through `nansum`. (`mad_clip` does this.)
- **Stable hot pixels survive the time-domain clip.** A pixel contributing a
  near-constant excess every integration looks stable in time and passes `mad_clip`,
  but dilutes the depth at its column when the star dims in transit. The
  `bad_col_repair` stage (run after `spec`) finds these by comparing to neighbour
  columns, masks them, and re-extracts — **always protect the He I 1083 nm column on
  SOSS** (`protected_wvl_um=[1.0833]`) so a real outflow isn't "repaired" away.

### Trace + background
- **Cross-correlation trace finders snap to wrong rows on faint targets.** Use the
  per-column flux-weighted centroid within a trusted window; it is the SOSS/MIRI
  default and far more robust.
- **A large contiguous dark block biases the trace polyfit.** `wavelength_left/right`
  only trims the light-curve window, not the trace fit. Set
  `restrict_trace_fit_to_window=True` (e.g. G395H NRS1's blue ~650 cols) so the
  argmax-of-noise block is excluded from the centroid fit.
- **Run a background-method sweep on each new target** — the optimum depends on the
  SED. `per_row_per_int` captures 1/f drifts a single 2D template can't.

### Extraction / aperture
- **Plain `nansum` beats fancier extraction on faint targets** (weighted/median/
  optimal rarely help unless the PSF template is genuinely clean).
- **`per_channel` minimizes photon noise; `wl_rms` tracks the white-light depth.**
  The per-channel criterion can pick an aperture several pixels too narrow; for
  NIRSpec prefer `wl_rms`.

### White-light MCMC
- **Save the MAP (`argmax log_prob`), not the marginal median.** On the
  `a`–`inc` transit-duration ridge the marginal median lands *off* the ridge and fits
  worse than any real sample. Both `fit_wl_mcmc` and `fit_wl_mcmc_joint` save the MAP.
- **Check for railed priors after every fit.** If any parameter's posterior sits
  within ~2% of a prior edge, the prior is railed and silently absorbs systematics —
  a low residual does NOT rule this out. Compare each parameter's **p1/p99** (not abs
  min/max — burn-in makes tail outliers) against its prior; widen and refit. Common
  culprits: `a/Rs`, `t0_offset`, `rp`. (`inc` bumping 90° is a physical edge, not a
  prior problem.)
- **Fix LD1 at the exotic_ld mean, fit only LD2.** u1+u2 is well constrained but
  u1−u2 is nearly degenerate; fitting both opens a ridge that produces fake "modes."
- **Use wide shared priors + enough burn-in,** not narrow ones near a degenerate
  ridge (which force fake bimodality). Normalize by a **pre-ingress** window and let
  the free slope absorb post-transit drift; bound `tc` tightly (±~3 min) with a good
  initial guess.

### LD, spectrum, combine
- **LD grid edges fail silently.** `exotic_ld` raises off-grid; the code fills with
  the nearest valid neighbor (a flat hold across an edge block, not interpolation). If
  a large fraction of a band is filled, the LD model is wrong for the star — try
  `ld_model="phoenix"`.
- **Per-channel normalization window matters.** `fit_spec_curves` accepts an
  `oot_mask`; pass the white-light-derived OOT window. A symmetric pre+post median can
  bias the depth if one side is contaminated (e.g. a He outflow tail).
- **curve_fit formal errors underestimate channel-to-channel scatter 2–5×.** The
  red-noise stage reports a per-channel β; rescale errors by `max(1, β)` before
  retrieval.

### Multi-detector / multi-visit
- **G395H — always use the joint NRS1+NRS2 white-light fit.** Both detectors see the
  same transit, so `a`, `inc`, `t0` are physically identical. Independent per-detector
  fits drift ~1% in `a/Rs` and ~0.2° in `inc`, creating a fake ~50–100 ppm step at the
  2.87↔3.82 µm gap. The joint fit writes the shared geometry; the per-detector spec
  stage reads it as fixed input; `combine_dets` produces the public stitched spectrum.
- **Multi-visit:** reduce each visit, then run a shared-geometry joint fit across
  visits before inverse-variance combining — per-visit orbital solutions otherwise
  drift enough to look atmospheric.

### Helium / escape (SOSS)
- **Always run the He I 1083 nm pixel-level check.** The line is sub-nm; binning to
  10–50 nm dilutes a real detection 10–50×. A few-hundred-ppm excess at 1.083 µm is a
  classic outflow signature you won't see in the binned spectrum.
- **Outflow tails bias the spec-fit depth low.** If absorption extends past the
  optical t1/t4, the post-transit OOT median pulls down with the tail. Read the
  helium residual panel: |post − pre| ≳ 100 ppm means refit that pixel with
  pre-transit-only OOT.

### MIRI LRS (calibration)
- **Strong wavelength-dependent settling ramp** at the start of each obs (largest
  ~9–11 µm). A linear slope is insufficient; use a 2-exponential + linear detrend per
  channel, fit on pre-ingress OOT and divided out before the transit fit.

### Ops
- **One busy CPU core is a red flag** — some step is serialized (e.g. JumpStep's
  snowball flagging; disable `expand_large_events` for BOTS time series). Check the
  log for where it's stuck.
- **Never copy a wavelength solution from a prior reduction.** Get it from the
  current Stage-2 output; CRDS-context or detector-region changes make a copied
  solution silently drift.

---

## Where to find things

| What you need | Where |
|---|---|
| Per-stage cookbook | [`RUNBOOK.md`](RUNBOOK.md) |
| Runnable driver templates | [`examples/`](examples/) |
| Calibration | `tswift.calibrate` (`run_calibrate`) |
| Bad-pixel MAD clip | `tswift.bad_pixel.mad_clip` |
| Trace + aperture | `tswift.extract.run_extract` |
| White-light MCMC (single / joint) | `tswift.wl_fit` |
| Per-wavelength fit + LD | `tswift.spec_fit` |
| Bad-column repair | `tswift.bad_columns` |
| Red-noise diagnostics | `tswift.red_noise` |
| He I 1083 nm check (SOSS) | `tswift.helium` |
| Limb asymmetry (optional, catwoman) | `tswift.limb_asymmetry` |
| Rebin + stitch | `tswift.combine` |
| Final figure + text output | `tswift.spectrum` |
| Project bootstrap / MAST fetch | `tswift.bootstrap`, `tswift.mast` |

If you discover a target-specific quirk during a reduction, write it in a
`NOTES.md` in that project so the next agent doesn't rediscover it.
