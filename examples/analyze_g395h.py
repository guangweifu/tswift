#!/usr/bin/env python
"""Runnable example driver — NIRSpec G395H, joint NRS1 + NRS2, end to end.

G395H is dual-detector. NRS1 and NRS2 see the SAME transit, so the orbital
geometry (a, inc, t0) is physically identical — fit it JOINTLY (pitfall #27),
then hold the shared geometry fixed for every spectroscopic channel and stitch
the two detectors. Independent per-detector fits create a fake ~50-100 ppm step
at the 2.87<->3.82 um gap.

Usage
-----
    tswift bootstrap "WASP-94 A b" 5924 --outdir "./WASP-94 A b"
    tswift fetch "./WASP-94 A b" 5924 --target "WASP-94 A" --instrument NIRSpec
    # set instrument.mode: "G395H" + paths.ld_data in config.yaml, then:
    python analyze_g395h.py "./WASP-94 A b"                 # full flow
    python analyze_g395h.py "./WASP-94 A b" extract wl      # selected stages

Per-detector products live in product_nrs1/ and product_nrs2/; the joint fit and
the stitched spectrum live in joint/ and combined/. See RUNBOOK.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

import tswift as ts

MODE = "G395H"
DETECTORS = ["nrs1", "nrs2"]
LD_MODE = "JWST_NIRSpec_G395H"
# NRS1's blue ~650 cols are unilluminated; NRS2 is clean across.
DEFAULT_WINDOW = {"nrs1": (650, 2040), "nrs2": (0, 2040)}


def load_ctx(project_dir: str) -> dict:
    project = Path(project_dir).resolve()
    cfg = yaml.safe_load((project / "config.yaml").read_text())
    target = json.loads((project / "target.json").read_text())
    return {"project": project, "cfg": cfg, "target": target}


def _pdir(ctx, det) -> Path:
    p = ctx["project"] / f"product_{det}"
    p.mkdir(exist_ok=True)
    return p


def _window(ctx, det):
    ex = ctx["cfg"]["extraction"]
    return (ex.get("wavelength_left", DEFAULT_WINDOW[det][0]),
            ex.get("wavelength_right", DEFAULT_WINDOW[det][1]))


def _wl_curve(product, det, wl_left, wl_right):
    """OOT-normalized white-light curve + bookkeeping for one detector."""
    clean = np.load(product / "clean_2D.npy")
    oot = np.load(product / "oot_mask.npy")
    time_all = np.load(product / "time_all.npy")
    time_hr = (time_all - time_all[0]) * 24.0
    flux = np.nansum(clean[:, wl_left:wl_right], axis=1)
    nor = flux / np.nanmedian(flux[oot])
    keep = np.isfinite(nor)
    return dict(time_hr=time_hr[keep], flux=nor[keep],
                oot_idx=np.where(oot[keep])[0],
                scatter=float(np.nanstd(nor[oot])))


def _ld(ctx, wvl, wl_left, wl_right):
    s = ctx["target"]["stellar"]
    return ts.compute_ld_per_wavelength(
        wvl, wl_left, wl_right,
        stellar_teff=s["teff_k"], stellar_logg=s["logg"], stellar_mh=s.get("feh", 0.0),
        ld_model="stagger", ld_mode=LD_MODE,
        ld_data_path=ctx["cfg"]["paths"]["ld_data"])


# --------------------------------------------------------------------------
# Per-detector stages
# --------------------------------------------------------------------------

def stage_calibrate(ctx: dict) -> None:
    for det in DETECTORS:
        ts.run_calibrate(ctx["project"], mode=MODE, detector=det,
                         crds_cache=ctx["cfg"]["paths"]["crds_cache"])


def stage_badpix(ctx: dict) -> None:
    bp = ctx["cfg"]["extraction"]["bad_pixel"]
    for det in DETECTORS:
        product = _pdir(ctx, det)
        data = np.load(product / "all_frame.npy")
        data_fixed, bad_mask, sigma = ts.mad_clip(
            data, n_sigma=bp["n_sigma"], min_sigma=bp["min_sigma"])
        np.save(product / "data_fixed.npy", data_fixed)
        ts.plot_bad_pixel(data, data_fixed, bad_mask, sigma, product / "figure",
                          n_sigma=bp["n_sigma"], detector=det)


def stage_extract(ctx: dict) -> None:
    ex = ctx["cfg"]["extraction"]
    for det in DETECTORS:
        product = _pdir(ctx, det)
        wl_left, wl_right = _window(ctx, det)
        data_fixed = np.load(product / "data_fixed.npy")
        result = ts.run_extract(
            data_fixed, mode=MODE, detector=det,
            trace_half_width=ex.get("trace_half_width", 16),
            trace_poly_order=ex.get("trace_poly_order", 3),
            aperture_criterion=ex.get("aperture_criterion", "wl_rms"),
            wavelength_left=wl_left, wavelength_right=wl_right,
            # NRS1's dark block biases the global trace polyfit — restrict it.
            restrict_trace_fit_to_window=ex.get("restrict_trace_fit_to_window", det == "nrs1"),
            outlier_window=ex.get("outlier_window", 20),
            outlier_threshold=ex.get("outlier_threshold", 4.0))
        ts.save_extract_outputs(result, product)
        json.dump({"aperture": list(result["aperture"])},
                  open(product / "aperture.json", "w"), indent=2)
        # Diagnostics — inspect these before the WL fit.
        time_all = np.load(product / "time_all.npy")
        time_hr = (time_all - time_all[0]) * 24.0
        up, down = result["aperture"]
        figdir = product / "figure"
        ts.plot_trace(data_fixed, result["trace_fit"],
                      ex.get("trace_half_width", 16), figdir, detector=det)
        ts.plot_aperture_scan(result["extract_2D"], result["all_aperture_results"],
                              up, down, result["oot_mask"], result["ingress_idx"],
                              time_hr, wl_left, wl_right, figdir, detector=det)
        cl_before = np.nansum(result["extract_2D"][:, up:down, :], axis=1)
        ts.plot_clean(cl_before, result["clean_2D"], time_hr, wl_left, wl_right,
                      figdir, detector=det)


def stage_wl(ctx: dict) -> None:
    """JOINT NRS1+NRS2 white-light fit -> shared geometry in joint/wl_geometry.json."""
    orb = ctx["target"]["orbital"]
    mc = ctx["cfg"]["mcmc"]
    fit_ld1 = mc.get("fit_ld1", False)
    detectors, initial, priors = [], {}, {}
    t0_init = None
    for det in DETECTORS:
        product = _pdir(ctx, det)
        wl_left, wl_right = _window(ctx, det)
        wvl = np.load(product / ts.wvl_filename(MODE, det))
        u1, u2 = _ld(ctx, wvl, wl_left, wl_right)
        u1_wl = float(np.nanmedian(u1[wl_left:wl_right]))
        u2_wl = float(np.nanmedian(u2[wl_left:wl_right]))
        c = _wl_curve(product, det, wl_left, wl_right)
        if t0_init is None:
            t0_init = float(np.median(c["time_hr"]))
        detectors.append(dict(name=det, time_data_hr=c["time_hr"], flux_data=c["flux"],
                              flux_err=np.full(c["flux"].size, c["scatter"]),
                              oot_indices=c["oot_idx"], u1=u1_wl, u2=u2_wl))
        initial.update({f"slope_{det}": 0.0, f"rp_{det}": ctx["target"]["planet"]["rp_rs_initial"],
                        f"LD2_{det}": u2_wl, f"constant_{det}": 1.0})
        priors.update({f"slope_{det}": (-0.01, 0.01), f"rp_{det}": (0.06, 0.15),
                       f"LD2_{det}": (-0.5, 0.5), f"constant_{det}": (0.98, 1.02)})
        if fit_ld1:
            initial[f"LD1_{det}"] = u1_wl
            priors[f"LD1_{det}"] = (0.0, 1.0)
    initial.update({"a": orb["a_over_rs"], "inc": orb["inclination_deg"], "t0_offset": t0_init})
    priors.update({"a": (3.0, 30.0), "inc": (80.0, 89.99),
                   "t0_offset": (t0_init - 1.0, t0_init + 1.0)})

    result = ts.fit_wl_mcmc_joint(
        detectors, period_hr=orb["period_days"] * 24.0,
        ecc=orb.get("eccentricity", 0.0), omega=orb.get("omega_deg", 90.0),
        initial=initial, priors=priors, fit_ld1=fit_ld1,
        nwalkers=mc.get("nwalkers", 64), nsteps=mc.get("nsteps", 10000),
        nburn=mc.get("nburn", 4000))

    joint = ctx["project"] / "joint"
    joint.mkdir(exist_ok=True)
    ts.plot_wl_fit_joint(result, detectors, outdir=joint / "figure")
    json.dump({"shared": result["shared"], "best_per_det": result["best_per_det"],
               "wavelength_left": {d: _window(ctx, d)[0] for d in DETECTORS},
               "wavelength_right": {d: _window(ctx, d)[1] for d in DETECTORS}},
              open(joint / "wl_geometry.json", "w"), indent=2)
    for det in DETECTORS:
        np.save(_pdir(ctx, det) / "wl_best_fit_curve.npy",
                result["best_fit_curve_per_det"][det])


def stage_spec(ctx: dict) -> None:
    """Per-detector per-channel fit using the SHARED joint geometry."""
    orb = ctx["target"]["orbital"]
    geom = json.loads((ctx["project"] / "joint" / "wl_geometry.json").read_text())
    shared = geom["shared"]
    for det in DETECTORS:
        product = _pdir(ctx, det)
        wl_left, wl_right = geom["wavelength_left"][det], geom["wavelength_right"][det]
        wvl = np.load(product / ts.wvl_filename(MODE, det))
        time_all = np.load(product / "time_all.npy")
        time_hr = (time_all - time_all[0]) * 24.0
        u1, u2 = _ld(ctx, wvl, wl_left, wl_right)
        np.save(product / "u1_per_wvl.npy", u1)
        np.save(product / "u2_per_wvl.npy", u2)
        sr = ts.fit_spec_curves(
            np.load(product / "clean_2D.npy"), time_hr, wvl,
            wavelength_left=wl_left, wavelength_right=wl_right,
            period_days=orb["period_days"], a_over_rs=shared["a"],
            inclination_deg=shared["inc"], t0_offset_hr=shared["t0_offset"],
            u1_arr=u1, u2_arr=u2, fix_ld2=False,
            oot_mask=np.load(product / "oot_mask.npy"))
        np.save(product / "spec_fit.npy", sr["fit"])
        np.save(product / "spec_fit_err.npy", sr["fit_err"])
        ts.plot_spec_fit(sr, wvl, u1, u2, outdir=product / "figure", detector=det)


def stage_bad_col_repair(ctx: dict) -> None:
    """Repair stable hot-pixel channels per detector (G395H has no He line to protect)."""
    orb = ctx["target"]["orbital"]
    g = json.loads((ctx["project"] / "joint" / "wl_geometry.json").read_text())["shared"]
    for det in DETECTORS:
        product = _pdir(ctx, det)
        ap = json.loads((product / "aperture.json").read_text())
        time_all = np.load(product / "time_all.npy")
        time_hr = (time_all - time_all[0]) * 24.0
        wvl = np.load(product / ts.wvl_filename(MODE, det))
        rep = ts.repair_outlier_columns(
            np.load(product / "clean_2D.npy"), np.load(product / "data_fixed.npy"),
            np.load(product / "trace_fit.npy"), np.load(product / "oot_mask.npy"),
            wvl, time_hr, np.load(product / "spec_fit.npy"),
            np.load(product / "spec_fit_err.npy"),
            np.load(product / "u1_per_wvl.npy"), np.load(product / "u2_per_wvl.npy"),
            geom=g, period_days=orb["period_days"], aperture=tuple(ap["aperture"]),
            trace_half_width=ctx["cfg"]["extraction"].get("trace_half_width", 16),
            protected_wvl_um=[])     # G395H: no He I 1083 nm line in band
        np.save(product / "clean_2D.npy", rep.clean_2D_repaired)
        np.save(product / "spec_fit.npy", rep.spec_fit_repaired)
        np.save(product / "spec_fit_err.npy", rep.spec_err_repaired)
        ts.plot_repair_diagnostics(rep, wvl, outdir=product / "figure",
                                   planet_name=ctx["target"]["name"], detector=det)


def stage_combine_dets(ctx: dict) -> None:
    """Inverse-variance stitch of NRS1 + NRS2 into the public G395H spectrum."""
    dets = {}
    for det in DETECTORS:
        product = _pdir(ctx, det)
        sr = np.load(product / "spec_fit.npy")
        sre = np.load(product / "spec_fit_err.npy")
        dets[det] = {"wvl_um": np.load(product / ts.wvl_filename(MODE, det)),
                     "rp": sr[:, 1], "rp_err": sre[:, 1]}
    combined = ts.combine_detectors(
        dets, bin_widths_nm=ctx["cfg"]["combine"]["bin_widths_nm"],
        bad_wavelengths_um=ctx["cfg"]["combine"].get("bad_wavelengths") or None)
    ts.save_spectrum(combined, outdir=ctx["project"] / "combined" / "spectrum",
                     planet_name=ctx["target"]["name"])


def stage_red_noise(ctx: dict) -> None:
    orb = ctx["target"]["orbital"]
    geom = json.loads((ctx["project"] / "joint" / "wl_geometry.json").read_text())
    g = geom["shared"]
    for det in DETECTORS:
        product = _pdir(ctx, det)
        wl_left, wl_right = geom["wavelength_left"][det], geom["wavelength_right"][det]
        wvl = np.load(product / ts.wvl_filename(MODE, det))
        time_all = np.load(product / "time_all.npy")
        time_hr = (time_all - time_all[0]) * 24.0
        clean = np.load(product / "clean_2D.npy")
        oot_mask = np.load(product / "oot_mask.npy")
        t_kept, resid = ts.reconstruct_wl_residuals_from_curve(
            clean, time_hr, oot_mask, np.load(product / "wl_best_fit_curve.npy"),
            wl_left=wl_left, wl_right=wl_right)
        ts.plot_wl_red_noise(ts.compute_wl_red_noise(t_kept, resid, detector=det),
                             outdir=product / "red_noise", planet_name=ctx["target"]["name"])
        spec_rn = ts.compute_spec_red_noise(
            clean, time_hr, wvl, np.load(product / "spec_fit.npy"), geom=g,
            period_days=orb["period_days"],
            u1_per_wvl=np.load(product / "u1_per_wvl.npy"),
            u2_per_wvl=np.load(product / "u2_per_wvl.npy"),
            oot_mask=oot_mask, fit_ld2=True, detector=det)
        ts.plot_spec_red_noise(spec_rn, outdir=product / "red_noise",
                               planet_name=ctx["target"]["name"])


STAGES = {
    "calibrate": stage_calibrate, "badpix": stage_badpix, "extract": stage_extract,
    "wl": stage_wl, "spec": stage_spec, "bad_col_repair": stage_bad_col_repair,
    "combine_dets": stage_combine_dets, "red_noise": stage_red_noise,
}
ALL = ["calibrate", "badpix", "extract", "wl", "spec", "bad_col_repair",
       "combine_dets", "red_noise"]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 1
    ctx = load_ctx(argv[0])
    for name in (argv[1:] or ALL):
        if name not in STAGES:
            print(f"unknown stage {name!r}; choices: {', '.join(ALL)}", file=sys.stderr)
            return 2
        print(f"=== {name} ===")
        STAGES[name](ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
