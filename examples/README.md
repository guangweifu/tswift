# tswift examples

Runnable, config-driven driver templates for the two most common setups. They
implement the canonical "call one stage, inspect its diagnostic PNG, then
continue" flow end to end. Copy one next to your project and adapt — most edits
live in `config.yaml` / `target.json`, not in the script.

| File | Setup | Flow |
|------|-------|------|
| [`analyze_soss.py`](analyze_soss.py) | NIRISS SOSS, single detector | calibrate → badpix → extract → wl → spec → bad_col_repair → combine → **helium** → red_noise |
| [`analyze_g395h.py`](analyze_g395h.py) | NIRSpec G395H, NRS1+NRS2 | per-det calibrate/badpix/extract → **joint** wl → per-det spec → combine_dets → red_noise |
| [`config.yaml`](config.yaml) | sample knobs | copy into your project dir and edit |

## Run

```bash
# 0. install + set data paths (see ../README.md)
export EXOTIC_LD_DATA=$HOME/exotic_ld_data
export CRDS_PATH=$HOME/crds_cache

# 1. create the project and download data
tswift bootstrap "WASP-69 b" 5924 --outdir "./WASP-69 b"
tswift fetch "./WASP-69 b" 5924 --target WASP-69 --instrument NIRISS

# 2. set instrument.mode + paths in ./WASP-69 b/config.yaml, then run:
python analyze_soss.py "./WASP-69 b"            # full flow
python analyze_soss.py "./WASP-69 b" extract    # a single stage, to iterate
```

`analyze_g395h.py` takes the same arguments; its stages are
`calibrate badpix extract wl spec combine_dets red_noise`.

## Notes

- **These are templates, not turnkey.** They run as-is on a correctly set-up
  project, but you should still open each `product*/figure/*.png` before trusting
  the next stage — the whole point of the pipeline is human/agent-in-the-loop
  inspection. See [`../RUNBOOK.md`](../RUNBOOK.md) for the per-stage checklist.
- The white-light `t0` here is initialized to the baseline midpoint; for real
  reductions, refine it from the observed ingress/egress (RUNBOOK §7) and check
  the corner plot for railed priors (pitfall #17) before continuing to `spec`.
- G395H **must** use the joint WL fit (pitfall #27); `analyze_g395h.py` does this
  by default and feeds the shared geometry to every channel.
