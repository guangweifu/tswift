#!/usr/bin/env python
"""Runnable example driver — NIRISS SOSS, single detector, end to end.

This is the canonical "call one stage, look at its PNG, continue" flow distilled
into one script. Adapt it per target; it is config-driven, so most edits live in
the project's ``config.yaml`` / ``target.json``, not here.

Usage
-----
    # 1. create the project + download data (see the top-level README / CLI):
    tswift bootstrap "WASP-69 b" 5924 --outdir "./WASP-69 b"
    tswift fetch "./WASP-69 b" 5924 --target WASP-69 --instrument NIRISS
    # 2. set instrument.mode: "SOSS" and paths.ld_data in ./WASP-69 b/config.yaml
    # 3. run this driver (or a single stage):
    python analyze_soss.py "./WASP-69 b"            # full flow
    python analyze_soss.py "./WASP-69 b" extract    # one stage

Every compute step has a paired ``plot_*`` — open the PNG it writes under
``product/figure/`` before trusting the next stage. See RUNBOOK.md for the
per-stage checklist and the knobs each stage exposes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

import tswift as ts

DETECTOR = "nis"
MODE = "SOSS"
LD_MODE = "JWST_NIRISS_SOSSo1"


def load_ctx(project_dir: str) -> dict:
    project = Path(project_dir).resolve()
    cfg = yaml.safe_load((project / "config.yaml").read_text())
    target = json.loads((project / "target.json").read_text())
    product = project / "product"
    product.mkdir(exist_ok=True)
    return {"project": project, "cfg": cfg, "target": target, "product": product}


def stage_calibrate(ctx: dict) -> None:
    """uncal -> ramp + per-column wavelength. Writes product/all_frame.npy etc."""
    ts.run_calibrate(
        ctx["project"], mode=MODE, detector=DETECTOR,
        crds_cache=ctx["cfg"]["paths"]["crds_cache"],
    )


def stage_badpix(ctx: dict) -> None:
    """MAD-robust per-pixel clip (pitfall #6 — never absolute-ADU thresholds)."""
    product = ctx["product"]
    bp = ctx["cfg"]["extraction"]["bad_pixel"]
    data = np.load(product / "all_frame.npy")
    data_fixed, bad_mask, sigma = ts.mad_clip(
        data, n_sigma=bp["n_sigma"], min_sigma=bp["min_sigma"])
    np.save(product / "data_fixed.npy", data_fixed)
    np.save(product / "sigma_pix.npy", sigma)
    ts.plot_bad_pixel(data, data_fixed, bad_mask, sigma, product / "figure",
                      n_sigma=bp["n_sigma"], detector=DETECTOR)


def stage_extract(ctx: dict) -> None:
    """Trace + aperture optimize + per-channel clean."""
    product, ex = ctx["product"], ctx["cfg"]["extraction"]
    data_fixed = np.load(product / "data_fixed.npy")
    result = ts.run_extract(
        data_fixed, mode=MODE, detector=DETECTOR,
        trace_half_width=ex.get("trace_half_width", 22),
        trace_poly_order=ex.get("trace_poly_order", 5),
        aperture_criterion=ex.get("aperture_criterion", "per_channel"),
        wavelength_left=ex.get("wavelength_left", 50),
        wavelength_right=ex.get("wavelength_right", 2040),
        outlier_window=ex.get("outlier_window", 20),
        outlier_threshold=ex.get("outlier_threshold", 4.0),
    )
    ts.save_extract_outputs(result, product)
    json.dump({"aperture": list(result["aperture"]),
               "ingress_idx": int(result["ingress_idx"])},
              open(product / "aperture.json", "w"), indent=2)
    # Diagnostics — inspect these before the WL fit.
    time_all = np.load(product / "time_all.npy")
    time_hr = (time_all - time_all[0]) * 24.0
    up, down = result["aperture"]
    wl_left = ex.get("wavelength_left", 50); wl_right = ex.get("wavelength_right", 2040)
    figdir = product / "figure"
    ts.plot_trace(data_fixed, result["trace_fit"],
                  ex.get("trace_half_width", 22), figdir, detector=DETECTOR)
    ts.plot_aperture_scan(result["extract_2D"], result["all_aperture_results"],
                          up, down, result["oot_mask"], result["ingress_idx"],
                          time_hr, wl_left, wl_right, figdir, detector=DETECTOR)
    cl_before = np.nansum(result["extract_2D"][:, up:down, :], axis=1)
    ts.plot_clean(cl_before, result["clean_2D"], time_hr, wl_left, wl_right,
                  figdir, detector=DETECTOR)


def _wl_band_ld(ctx, wvl, wl_left, wl_right):
    """Per-wavelength LD (reused by spec) + the band-median for the WL fit."""
    s = ctx["target"]["stellar"]
    u1, u2 = ts.compute_ld_per_wavelength(
        wvl, wl_left, wl_right,
        stellar_teff=s["teff_k"], stellar_logg=s["logg"], stellar_mh=s.get("feh", 0.0),
        ld_model="stagger", ld_mode=LD_MODE,
        ld_data_path=ctx["cfg"]["paths"]["ld_data"],
    )
    u1_wl = float(np.nanmedian(u1[wl_left:wl_right]))
    u2_wl = float(np.nanmedian(u2[wl_left:wl_right]))
    return u1, u2, u1_wl, u2_wl


def stage_wl(ctx: dict) -> None:
    """White-light emcee fit. Saves the MAP geometry to wl_best.json."""
    product, ex = ctx["product"], ctx["cfg"]["extraction"]
    orb = ctx["target"]["orbital"]
    wl_left = ex.get("wavelength_left", 50)
    wl_right = ex.get("wavelength_right", 2040)

    clean = np.load(product / "clean_2D.npy")
    oot_mask = np.load(product / "oot_mask.npy")
    time_all = np.load(product / "time_all.npy")
    wvl = np.load(product / ts.wvl_filename(MODE, DETECTOR))
    time_hr = (time_all - time_all[0]) * 24.0

    # White-light curve = OOT-normalized sum over the bandpass.
    wl_flux = np.nansum(clean[:, wl_left:wl_right], axis=1)
    wl_nor = wl_flux / np.nanmedian(wl_flux[oot_mask])
    keep = np.isfinite(wl_nor)
    oot_idx = np.where(oot_mask[keep])[0]
    oot_scatter = float(np.nanstd(wl_nor[oot_mask]))

    _, _, u1_wl, u2_wl = _wl_band_ld(ctx, wvl, wl_left, wl_right)
    t0_init = float(np.median(time_hr))   # refine with the observed ingress/egress
    mc = ctx["cfg"]["mcmc"]
    result = ts.fit_wl_mcmc(
        time_data_hr=time_hr[keep], flux_data=wl_nor[keep],
        flux_err=np.full(int(keep.sum()), oot_scatter),
        period_hr=orb["period_days"] * 24.0,
        ecc=orb.get("eccentricity", 0.0), omega=orb.get("omega_deg", 90.0),
        u1=u1_wl, u2=u2_wl, oot_indices=oot_idx,
        initial={"slope": 0.0, "rp": ctx["target"]["planet"]["rp_rs_initial"],
                 "LD2": u2_wl, "constant": 1.0,
                 "a": orb["a_over_rs"], "inc": orb["inclination_deg"],
                 "t0_offset": t0_init},
        priors={"slope": (-0.01, 0.01), "rp": (0.06, 0.15), "LD2": (-0.5, 0.5),
                "constant": (0.98, 1.02), "a": (3.0, 30.0), "inc": (80.0, 89.99),
                "t0_offset": (t0_init - 1.0, t0_init + 1.0)},
        fit_ld1=mc.get("fit_ld1", False),
        nwalkers=mc["nwalkers"], nsteps=mc["nsteps"], nburn=mc["nburn"],
    )
    # CLAUDE.md pitfall #17: eyeball corner.png for railed priors before trusting this.
    ts.plot_wl_fit(result, time_hr[keep], wl_nor[keep],
                   outdir=product / "figure", detector=DETECTOR)
    np.save(product / "wl_best_fit_curve.npy", result["best_fit_curve"])
    params = dict(zip(result["param_order"], result["best_params"].tolist()))
    json.dump({"param_order": result["param_order"],
               "best_params": result["best_params"].tolist(),
               "oot_scatter_ppm": oot_scatter * 1e6,
               "wavelength_left": wl_left, "wavelength_right": wl_right,
               **{k: params[k] for k in ("a", "inc", "t0_offset")}},
              open(product / "wl_best.json", "w"), indent=2)


def stage_spec(ctx: dict) -> None:
    """Per-wavelength depth, geometry fixed at the WL MAP."""
    product, orb = ctx["product"], ctx["target"]["orbital"]
    wl = json.loads((product / "wl_best.json").read_text())
    wl_left, wl_right = wl["wavelength_left"], wl["wavelength_right"]

    clean = np.load(product / "clean_2D.npy")
    oot_mask = np.load(product / "oot_mask.npy")
    time_all = np.load(product / "time_all.npy")
    wvl = np.load(product / ts.wvl_filename(MODE, DETECTOR))
    time_hr = (time_all - time_all[0]) * 24.0

    u1, u2, _, _ = _wl_band_ld(ctx, wvl, wl_left, wl_right)
    np.save(product / "u1_per_wvl.npy", u1)
    np.save(product / "u2_per_wvl.npy", u2)
    sr = ts.fit_spec_curves(
        clean, time_hr, wvl, wavelength_left=wl_left, wavelength_right=wl_right,
        period_days=orb["period_days"], a_over_rs=wl["a"],
        inclination_deg=wl["inc"], t0_offset_hr=wl["t0_offset"],
        ecc=orb.get("eccentricity", 0.0), omega=orb.get("omega_deg", 90.0),
        u1_arr=u1, u2_arr=u2, fix_ld2=False,
        oot_mask=oot_mask,            # WL-derived OOT window (pitfall #23)
    )
    np.save(product / "spec_fit.npy", sr["fit"])
    np.save(product / "spec_fit_err.npy", sr["fit_err"])
    ts.plot_spec_fit(sr, wvl, u1, u2, outdir=product / "figure", detector=DETECTOR)


def stage_bad_col_repair(ctx: dict) -> None:
    """Repair stable hot-pixel channels (protect He I 1083 nm)."""
    product, orb = ctx["product"], ctx["target"]["orbital"]
    wl = json.loads((product / "wl_best.json").read_text())
    ap = json.loads((product / "aperture.json").read_text())
    g = {"a": wl["a"], "inc": wl["inc"], "t0_offset": wl["t0_offset"]}
    time_all = np.load(product / "time_all.npy")
    time_hr = (time_all - time_all[0]) * 24.0
    wvl = np.load(product / ts.wvl_filename(MODE, DETECTOR))
    rep = ts.repair_outlier_columns(
        np.load(product / "clean_2D.npy"), np.load(product / "data_fixed.npy"),
        np.load(product / "trace_fit.npy"), np.load(product / "oot_mask.npy"),
        wvl, time_hr, np.load(product / "spec_fit.npy"),
        np.load(product / "spec_fit_err.npy"),
        np.load(product / "u1_per_wvl.npy"), np.load(product / "u2_per_wvl.npy"),
        geom=g, period_days=orb["period_days"],
        aperture=tuple(ap["aperture"]),
        trace_half_width=ctx["cfg"]["extraction"].get("trace_half_width", 22),
        protected_wvl_um=[1.0833],     # never "repair" a real He outflow
    )
    np.save(product / "clean_2D.npy", rep.clean_2D_repaired)
    np.save(product / "spec_fit.npy", rep.spec_fit_repaired)
    np.save(product / "spec_fit_err.npy", rep.spec_err_repaired)
    ts.plot_repair_diagnostics(rep, wvl, outdir=product / "figure",
                               planet_name=ctx["target"]["name"], detector=DETECTOR)


def stage_combine(ctx: dict) -> None:
    """Depth + rebin + save the transmission spectrum."""
    product = ctx["product"]
    wvl = np.load(product / ts.wvl_filename(MODE, DETECTOR))
    sr = np.load(product / "spec_fit.npy")
    sre = np.load(product / "spec_fit_err.npy")
    combined = ts.combine_spectrum(
        wvl, sr[:, 1], sre[:, 1],
        bin_widths_nm=ctx["cfg"]["combine"]["bin_widths_nm"],
        bad_wavelengths_um=ctx["cfg"]["combine"].get("bad_wavelengths") or None,
    )
    ts.save_spectrum(combined, outdir=product / "spectrum",
                     planet_name=ctx["target"]["name"])


def stage_helium(ctx: dict) -> None:
    """SOSS-only He I 1083 nm pixel check (mandatory; pitfall #32/#33)."""
    product = ctx["product"]
    wl = json.loads((product / "wl_best.json").read_text())
    time_all = np.load(product / "time_all.npy")
    time_hr = (time_all - time_all[0]) * 24.0
    sr = np.load(product / "spec_fit.npy")
    sre = np.load(product / "spec_fit_err.npy")
    res = ts.check_helium_1083(
        np.load(product / "clean_2D.npy"), time_hr,
        np.load(product / ts.wvl_filename(MODE, DETECTOR)),
        rp=sr[:, 1], rp_err=sre[:, 1], oot_mask=np.load(product / "oot_mask.npy"),
        wl_left=wl["wavelength_left"], wl_right=wl["wavelength_right"],
        t0_hr=wl["t0_offset"],
    )
    ts.plot_helium(res, outdir=product / "helium", planet_name=ctx["target"]["name"])


def stage_red_noise(ctx: dict) -> None:
    """White-light + per-channel red-noise / beta-factor diagnostics (last stage)."""
    product, orb = ctx["product"], ctx["target"]["orbital"]
    wl = json.loads((product / "wl_best.json").read_text())
    g = {"a": wl["a"], "inc": wl["inc"], "t0_offset": wl["t0_offset"]}
    time_all = np.load(product / "time_all.npy")
    time_hr = (time_all - time_all[0]) * 24.0
    clean = np.load(product / "clean_2D.npy")
    oot_mask = np.load(product / "oot_mask.npy")
    wvl = np.load(product / ts.wvl_filename(MODE, DETECTOR))

    t_kept, resid = ts.reconstruct_wl_residuals_from_curve(
        clean, time_hr, oot_mask, np.load(product / "wl_best_fit_curve.npy"),
        wl_left=wl["wavelength_left"], wl_right=wl["wavelength_right"])
    wl_rn = ts.compute_wl_red_noise(t_kept, resid, detector=DETECTOR,
                                    oot_scatter_ppm=wl.get("oot_scatter_ppm"))
    ts.plot_wl_red_noise(wl_rn, outdir=product / "red_noise",
                         planet_name=ctx["target"]["name"])
    spec_rn = ts.compute_spec_red_noise(
        clean, time_hr, wvl, np.load(product / "spec_fit.npy"),
        geom=g, period_days=orb["period_days"],
        u1_per_wvl=np.load(product / "u1_per_wvl.npy"),
        u2_per_wvl=np.load(product / "u2_per_wvl.npy"),
        oot_mask=oot_mask, fit_ld2=True)
    ts.plot_spec_red_noise(spec_rn, outdir=product / "red_noise",
                           planet_name=ctx["target"]["name"])


STAGES = {
    "calibrate": stage_calibrate, "badpix": stage_badpix, "extract": stage_extract,
    "wl": stage_wl, "spec": stage_spec, "bad_col_repair": stage_bad_col_repair,
    "combine": stage_combine, "helium": stage_helium, "red_noise": stage_red_noise,
}
ALL = ["calibrate", "badpix", "extract", "wl", "spec", "bad_col_repair",
       "combine", "helium", "red_noise"]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 1
    project_dir = argv[0]
    stages = argv[1:] or ALL
    ctx = load_ctx(project_dir)
    for name in stages:
        if name not in STAGES:
            print(f"unknown stage {name!r}; choices: {', '.join(ALL)}", file=sys.stderr)
            return 2
        print(f"=== {name} ===")
        STAGES[name](ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
