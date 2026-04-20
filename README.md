# tswift

**JWST transit spectroscopy pipeline — from raw MAST data to a transmission spectrum.**

tswift is an agent-native pipeline: each stage has a clear Python API, writes
structured outputs + review figures, and the failure modes are documented right
next to the knob that fixes them. It's designed so an LLM agent (or a human)
can inspect intermediate products, diagnose problems, and iterate — not just
push a button and pray.

Supports **NIRSpec G395H**, **NIRSpec PRISM**, and **NIRISS SOSS**.

> **Status**: development (`2.0.0.dev0`). The analysis stages (extraction →
> fitting → combining → spectrum) are ported and regression-tested bit-identical
> against a mature predecessor pipeline on WASP-69 b. Stage-1/2 calibration
> (uncal → ramp) is not yet ported to the v2 API — v2 starts from existing
> ramp-fit products. See [RUNBOOK.md §4](RUNBOOK.md) for the current bridging
> approach.

## Install

The package depends on `jwst`, `exotic_ld`, `batman-package`, `emcee`,
`astroquery`, `pydantic` and a CRDS cache. Easiest path is a dedicated conda env:

```bash
conda create -n tswift python=3.12
conda activate tswift
conda install -c conda-forge jwst astroquery emcee batman-package
pip install exotic-ld
pip install git+https://github.com/guangweifu/tswift.git
```

Configure CRDS (once per machine):

```bash
export CRDS_PATH=$HOME/crds_cache
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
```

For proprietary MAST downloads, save a token at `~/.mast_token` (owner-only,
`chmod 600`). `tswift.fetch()` reads it automatically.

## Quickstart

Read [RUNBOOK.md](RUNBOOK.md) — it walks through every pipeline stage with
the exact Python call, sanity checks, and failure modes.

```python
from tswift import bootstrap, fetch

# 1. Bootstrap a project dir, auto-populate target.json from NASA Exoplanet Archive
project = bootstrap("WASP-69 b", program="5924", outdir="./WASP-69b_v2")

# 2. Download uncal.fits from MAST (the token-gated call also works for
#    proprietary data if you've saved a token at ~/.mast_token)
fetch(project, program_id="5924", target_name="WASP-69", instrument="NIRISS")

# 3. Run stage1/2 calibration (jwst.pipeline — not yet wrapped in tswift v2;
#    for now use the legacy wrapper described in RUNBOOK.md §4, or call jwst
#    directly to produce all_frame.npy + time_all.npy + wvl.npy)

# 4. v2 analysis — the ported stages:
from tswift import (
    mad_clip, run_extract,
    fit_wl_mcmc, compute_ld_per_wavelength, fit_spec_curves,
    combine_spectrum, save_spectrum,
)
```

Every stage has a paired `plot_*` function that writes a review PNG — see
[RUNBOOK.md](RUNBOOK.md) for the list.

## Regression tests

```bash
# t0-cache design invariant (runs in ~1 s)
python tests/test_t0_cache.py

# Full analysis pipeline (~2.5 min) requires a pre-calibrated WASP-69 b
# ramp product; see RUNBOOK.md §5 onwards for running it on your own data.
```

## What's in this package

| Module | What it does |
|--------|--------------|
| `tswift.contracts` | Pydantic models for `Target`, `Manifest`, `Diagnostics` |
| `tswift.target_db` | NASA Exoplanet Archive query → populated `Target` |
| `tswift.mast` | List + fetch JWST observations (public + proprietary) |
| `tswift.bootstrap` | `bootstrap(planet, program)` → project dir + target.json |
| `tswift.bad_pixel` | MAD-robust per-pixel clipping (`mad_clip`) + diagnostic plot |
| `tswift.extract` | Trace find + aperture optimize + per-channel clean |
| `tswift.transit_model` | batman wrapper with correct t0 caching |
| `tswift.wl_fit` | White-light `emcee` MCMC |
| `tswift.spec_fit` | Per-wavelength `scipy.curve_fit` with `exotic_ld` LD |
| `tswift.combine` | Rebin + stitch detectors + inverse-variance weighting |
| `tswift.spectrum` | Save final figure + text files + summary JSON |

## License

MIT — see [LICENSE](LICENSE).

## Provenance

Extracted from the author's JWST transit spectroscopy working repo. The full
research history (planets, experiments, calibration notes) lives in a private
repo; this public release is the pipeline code only.
