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

```python
from tswift import bootstrap, fetch

project = bootstrap("WASP-69 b", program="5924", outdir="./WASP-69b_v2")
fetch(project)
# ... then calibration / extraction / fit (coming soon)
```
