# tswift

AI-agent-native JWST transit spectroscopy pipeline. Raw MAST data → transmission spectrum,
with structured diagnostics at every step so an agent can inspect, troubleshoot, and iterate.

Supports NIRSpec G395H, NIRSpec PRISM, and NIRISS SOSS.

## Design intent

- **Inputs**: planet name + program ID. Everything else is derived.
- **Outputs**: transmission spectrum + per-step diagnostics (JSON + figures).
- **Interface**: Python API is primary. An agent (Claude Code) reads `diagnostics.json`
  files and orchestrates the pipeline. A `tswift` CLI is available for manual use.

## Install

```bash
pip install -e ./tswift
```

The package expects `jwst`, `exotic_ld`, and a CRDS cache to already be configured — see
project-level `CLAUDE.md` for environment setup.

## Quickstart

**For an agent or new user**: read [RUNBOOK.md](RUNBOOK.md) start to finish. It walks
through every pipeline stage with the exact Python call, sanity checks, and the
common failure modes to watch for.

```python
from tswift import bootstrap, fetch

project = bootstrap("WASP-69 b", program="5924", outdir="./WASP-69b_v2")
fetch(project, program_id="5924", target_name="WASP-69", instrument="NIRISS")
# ... then calibration (port pending) → mad_clip → run_extract → fit_wl_mcmc →
#     fit_spec_curves → combine_spectrum → save_spectrum
# See RUNBOOK.md for the full step-by-step.
```

## Regression tests

```bash
# t0 cache design (unit test, < 1 s)
python tswift/tests/test_t0_cache.py

# Full WASP-69 b pipeline from ramp to spectrum (~2.5 min)
python tswift/scripts/test_end_to_end_wasp69b.py
```
