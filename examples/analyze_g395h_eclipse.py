#!/usr/bin/env python
"""Runnable example driver — NIRSpec G395H SECONDARY ECLIPSE (emission), NRS1+NRS2.

A secondary eclipse measures the planet's dayside flux (an emission spectrum),
NOT transmission. Calibration + extraction are identical to the transit flow, but
the FIT is different: an `batman` ``transittype="secondary"`` model with NO limb
darkening, fitting the eclipse depth Fp/Fs (the emission spectrum) and the eclipse
mid-time, with orbital geometry fixed from the known transit parameters.

    tswift bootstrap "KELT-20 b" 6978 --outdir "./KELT-20 b"
    tswift fetch "./KELT-20 b" 6978 --target KELT-20 --instrument NIRSpec --contains _04102_
    # set instrument.mode: "G395H" + paths in config.yaml, then:
    python analyze_g395h_eclipse.py "./KELT-20 b"            # full flow
    python analyze_g395h_eclipse.py "./KELT-20 b" eclipse_wl # one stage

Stages: calibrate badpix extract eclipse_wl eclipse_spec combine_emission.
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
DEFAULT_WINDOW = {"nrs1": (650, 2040), "nrs2": (0, 2040)}


def load_ctx(project_dir):
    project = Path(project_dir).resolve()
    return {"project": project,
            "cfg": yaml.safe_load((project / "config.yaml").read_text()),
            "target": json.loads((project / "target.json").read_text())}


def _pdir(ctx, det):
    p = ctx["project"] / f"product_{det}"; p.mkdir(exist_ok=True); return p


def _dets(ctx):
    """Detectors to process — config `instrument.detectors`, else both."""
    return ctx["cfg"].get("instrument", {}).get("detectors", DETECTORS)


def _window(ctx, det):
    ex = ctx["cfg"]["extraction"]
    return (ex.get("wavelength_left", DEFAULT_WINDOW[det][0]),
            ex.get("wavelength_right", DEFAULT_WINDOW[det][1]))


def _geom_rp(ctx):
    orb, pl = ctx["target"]["orbital"], ctx["target"]["planet"]
    return {"a": orb["a_over_rs"], "inc": orb["inclination_deg"]}, pl["rp_rs_initial"]


# --- calibration + extraction: identical to the transit G395H flow ---

def stage_calibrate(ctx):
    for det in _dets(ctx):
        ts.run_calibrate(ctx["project"], mode=MODE, detector=det,
                         crds_cache=ctx["cfg"]["paths"]["crds_cache"])


def stage_badpix(ctx):
    bp = ctx["cfg"]["extraction"]["bad_pixel"]
    for det in _dets(ctx):
        product = _pdir(ctx, det)
        data = np.load(product / "all_frame.npy")
        data_fixed, bad_mask, sigma = ts.mad_clip(
            data, n_sigma=bp["n_sigma"], min_sigma=bp["min_sigma"])
        np.save(product / "data_fixed.npy", data_fixed)
        ts.plot_bad_pixel(data, data_fixed, bad_mask, sigma, product / "figure",
                          n_sigma=bp["n_sigma"], detector=det)


def stage_extract(ctx):
    ex = ctx["cfg"]["extraction"]
    for det in _dets(ctx):
        product = _pdir(ctx, det)
        wl_left, wl_right = _window(ctx, det)
        data_fixed = np.load(product / "data_fixed.npy")
        result = ts.run_extract(
            data_fixed, mode=MODE, detector=det,
            trace_half_width=ex.get("trace_half_width", 16),
            trace_poly_order=ex.get("trace_poly_order", 3),
            aperture_criterion=ex.get("aperture_criterion", "wl_rms"),
            wavelength_left=wl_left, wavelength_right=wl_right,
            restrict_trace_fit_to_window=ex.get("restrict_trace_fit_to_window", det == "nrs1"),
            outlier_window=ex.get("outlier_window", 20),
            outlier_threshold=ex.get("outlier_threshold", 4.0))
        ts.save_extract_outputs(result, product)
        json.dump({"aperture": list(result["aperture"])},
                  open(product / "aperture.json", "w"), indent=2)
        time_all = np.load(product / "time_all.npy")
        time_hr = (time_all - time_all[0]) * 24.0
        up, down = result["aperture"]; figdir = product / "figure"
        ts.plot_trace(data_fixed, result["trace_fit"],
                      ex.get("trace_half_width", 16), figdir, detector=det)
        cl_before = np.nansum(result["extract_2D"][:, up:down, :], axis=1)
        ts.plot_clean(cl_before, result["clean_2D"], time_hr, wl_left, wl_right,
                      figdir, detector=det)


# --- eclipse-specific fitting ---

def _wl_curve(product, wl_left, wl_right):
    clean = np.load(product / "clean_2D.npy")
    oot = np.load(product / "oot_mask.npy")
    time_all = np.load(product / "time_all.npy")
    time_hr = (time_all - time_all[0]) * 24.0
    flux = np.nansum(clean[:, wl_left:wl_right], axis=1)
    nor = flux / np.nanmedian(flux[oot])
    keep = np.isfinite(nor)
    return time_hr[keep], nor[keep], float(np.nanstd(nor[oot]))


def stage_eclipse_wl(ctx):
    """White-light eclipse fit per detector (depth + eclipse time; geometry fixed)."""
    orb = ctx["target"]["orbital"]; geom, rp = _geom_rp(ctx)
    P = orb["period_days"]
    for det in _dets(ctx):
        product = _pdir(ctx, det)
        wl_left, wl_right = _window(ctx, det)
        t, f, sc = _wl_curve(product, wl_left, wl_right)
        t_sec_guess = float(np.median(t))     # eclipse near mid-observation; refine via prior
        res = ts.fit_eclipse_wl(
            t, f, np.full(f.size, sc), period_days=P, geom=geom, rp=rp,
            ecc=orb.get("eccentricity", 0.0), omega=orb.get("omega_deg", 90.0),
            initial={"slope": 0.0, "fp": 0.0015, "constant": 1.0, "t_sec_offset": t_sec_guess},
            priors={"slope": (-0.01, 0.01), "fp": (0.0, 0.01),
                    "constant": (0.98, 1.02),
                    "t_sec_offset": (t_sec_guess - 1.5, t_sec_guess + 1.5)})
        ts.plot_eclipse_wl(res, t, f, product / "figure", detector=det)
        json.dump({"eclipse_depth_ppm": res["eclipse_depth_ppm"],
                   "t_sec_offset": res["t_sec_offset"],
                   "rms_ppm": res["rms_residual"] * 1e6,
                   "wavelength_left": wl_left, "wavelength_right": wl_right},
                  open(product / "eclipse_wl.json", "w"), indent=2)


def stage_eclipse_spec(ctx):
    """Per-channel eclipse depth = emission spectrum (geometry + eclipse time fixed)."""
    orb = ctx["target"]["orbital"]; geom, rp = _geom_rp(ctx)
    for det in _dets(ctx):
        product = _pdir(ctx, det)
        ew = json.loads((product / "eclipse_wl.json").read_text())
        wl_left, wl_right = ew["wavelength_left"], ew["wavelength_right"]
        wvl = np.load(product / ts.wvl_filename(MODE, det))
        time_all = np.load(product / "time_all.npy")
        time_hr = (time_all - time_all[0]) * 24.0
        sr = ts.fit_eclipse_curves(
            np.load(product / "clean_2D.npy"), time_hr, wvl,
            wavelength_left=wl_left, wavelength_right=wl_right,
            period_days=orb["period_days"], geom=geom, rp=rp,
            t_sec_offset=ew["t_sec_offset"],
            ecc=orb.get("eccentricity", 0.0), omega=orb.get("omega_deg", 90.0),
            oot_mask=np.load(product / "oot_mask.npy"))
        np.save(product / "eclipse_depth_ppm.npy", sr["depth_ppm"])
        np.save(product / "eclipse_depth_err_ppm.npy", sr["depth_err_ppm"])


def stage_combine_emission(ctx):
    """Stitch NRS1+NRS2 eclipse depths into the dayside emission spectrum."""
    wvl_all, dep_all, err_all = [], [], []
    for det in _dets(ctx):
        product = _pdir(ctx, det)
        wvl_all.append(np.load(product / ts.wvl_filename(MODE, det)))
        dep_all.append(np.load(product / "eclipse_depth_ppm.npy"))
        err_all.append(np.load(product / "eclipse_depth_err_ppm.npy"))
    wvl = np.concatenate(wvl_all); dep = np.concatenate(dep_all); err = np.concatenate(err_all)
    order = np.argsort(wvl)
    comb = ts.combine_emission(wvl[order], dep[order], err[order],
                               bin_widths_nm=ctx["cfg"]["combine"]["bin_widths_nm"])
    ts.save_emission_spectrum(comb, ctx["project"] / "combined" / "emission",
                              planet_name=ctx["target"]["name"])


STAGES = {"calibrate": stage_calibrate, "badpix": stage_badpix, "extract": stage_extract,
          "eclipse_wl": stage_eclipse_wl, "eclipse_spec": stage_eclipse_spec,
          "combine_emission": stage_combine_emission}
ALL = ["calibrate", "badpix", "extract", "eclipse_wl", "eclipse_spec", "combine_emission"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__); return 1
    ctx = load_ctx(argv[0])
    for name in (argv[1:] or ALL):
        if name not in STAGES:
            print(f"unknown stage {name!r}; choices: {', '.join(ALL)}", file=sys.stderr); return 2
        print(f"=== {name} ==="); STAGES[name](ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
