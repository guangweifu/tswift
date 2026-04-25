"""White-light transit MCMC with emcee + multiprocessing.

Two flavours
------------

1. **Per-detector fit** (`fit_wl_mcmc`).  Fits one light curve at a time.
   Parameters (theta) follow `transit_model.param_order(fit_ld1)`:

   - fit_ld1=False : ['slope', 'rp', 'LD2', 'constant', 'a', 'inc', 't0_offset']
   - fit_ld1=True  : ['slope', 'rp', 'LD1', 'LD2', 'constant', 'a', 'inc', 't0_offset']

2. **Joint multi-detector fit** (`fit_wl_mcmc_joint`).  Fits N light curves
   simultaneously with **shared** orbital geometry (`a`, `inc`, `t0_offset`)
   and **per-detector** wavelength-dependent params (`slope, rp, [LD1,] LD2,
   constant`).  Use this for G395H where NRS1 + NRS2 see the same transit on
   the same star at the same time — independent per-detector fits will
   disagree by ~1% in `a/Rs` and ~0.2° in `inc`, biasing the relative depth
   between detectors and creating fake "atmospheric" offsets at the
   2.87↔3.82 µm gap.  Joint param order:

       [slope_d1, rp_d1, [LD1_d1,] LD2_d1, constant_d1,
        slope_d2, rp_d2, [LD1_d2,] LD2_d2, constant_d2,
        ...
        a, inc, t0_offset]

   Per-detector params are name-suffixed with the detector key
   (e.g. ``rp_nrs1``, ``LD2_nrs2``) in the priors / initial dicts.

All normalization/time referencing conventions match `transit_model`.  In
particular, `oot_indices` is a list of indices into the *fitted* (post-masking)
data array that should serve as the OOT baseline; the model's predicted flux
is normalized against the same indices so residuals are centered on 1.0.

For multiprocessing safety, every worker process is initialized via
`transit_model.init_globals(...)` — the batman TransitModel can't be pickled
so we rebuild it per worker.  The joint fit additionally caches a per-detector
batman model (one per `id(time_array)`) inside that same module.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import emcee
import numpy as np

from tswift import transit_model as tm

logger = logging.getLogger(__name__)


def fit_wl_mcmc(
    time_data_hr: np.ndarray,
    flux_data: np.ndarray,
    flux_err: np.ndarray,
    *,
    period_hr: float,
    ecc: float,
    omega: float,
    u1: float,
    u2: float,
    oot_indices: np.ndarray,
    initial: dict,
    priors: dict,
    time_data_ref: Optional[float] = None,
    fit_ld1: bool = False,
    nwalkers: int = 48,
    nsteps: int = 8000,
    nburn: int = 3000,
    nprocesses: Optional[int] = None,
    init_scatter: float = 1e-4,
    rng_seed: Optional[int] = None,
) -> dict:
    """Run the emcee white-light MCMC fit.

    Parameters
    ----------
    time_data_hr : (N,)
        Time (hours from visit start) for every fitted integration. Masked-out
        integrations must already be removed.
    flux_data : (N,)
        OOT-normalized white-light flux at those times.
    flux_err : (N,)
        Per-point uncertainty. Legacy code uses a constant OOT std here.
    period_hr : float
        Orbital period in hours.
    ecc, omega : float
        Fixed eccentricity and argument of periastron.
    u1, u2 : float
        Limb-darkening coefficients for the bandpass (from exotic_ld).
    oot_indices : (K,) int
        Indices INTO time_data_hr that form the OOT baseline — used to normalize
        the model flux and keep model/data on the same footing.
    initial : dict of name -> float
        Initial values for every parameter in `param_order(fit_ld1)`.
    priors : dict of name -> (lo, hi)
        Uniform prior bounds.
    time_data_ref : float, optional
        Reference time for the linear slope baseline. Defaults to median(time_data_hr).
    fit_ld1 : bool
        If False (default) LD1 is fixed to `u1`; if True, LD1 is a fit parameter.
    nwalkers, nsteps, nburn : int
    nprocesses : int, optional
        Pool size for multiprocessing. Defaults to `mp.cpu_count()`.
    init_scatter : float
        Gaussian scatter on the initial walker positions.
    rng_seed : int, optional
        For reproducible walker initialization (the MCMC itself remains stochastic
        via emcee's internal RNG unless you also seed that separately).

    Returns
    -------
    dict
        best_params   : (ndim,) median over post-burn samples
        best_errors   : (ndim, 2) 16/84-percentile errors
        samples       : (nsteps*nwalkers - nburn*nwalkers, ndim) flattened chain
        chain         : (nsteps, nwalkers, ndim) raw chain
        param_order   : list[str]
        ndim          : int
        acceptance    : (nwalkers,) mean acceptance per walker
        rms_residual  : float, RMS of data - best-fit model
        best_fit_curve: (N,) best-fit model evaluated at time_data_hr
        residuals     : (N,)
    """
    param_order = tm.param_order(fit_ld1)
    missing = [p for p in param_order if p not in initial]
    if missing:
        raise KeyError(f"Missing initial values for: {missing}")
    initial_arr = np.array([initial[p] for p in param_order], dtype=float)

    if time_data_ref is None:
        time_data_ref = float(np.median(time_data_hr))

    oot_indices = np.asarray(oot_indices, dtype=int)
    ndim = len(initial_arr)
    ncpu = nprocesses if nprocesses is not None else mp.cpu_count()

    rng = np.random.default_rng(rng_seed)
    pos = initial_arr + init_scatter * rng.standard_normal((nwalkers, ndim))

    logger.info(
        f"WL MCMC: {nwalkers} walkers × {nsteps} steps (burn={nburn}), "
        f"ndim={ndim}, fit_ld1={fit_ld1}, pool={ncpu}, "
        f"u1={u1:.4f}, u2={u2:.4f}, period_hr={period_hr:.4f}, ecc={ecc}"
    )

    # Positional initargs — nested closures can't be pickled by mp.spawn.
    init_args = (u1, u2, fit_ld1, period_hr, ecc, omega,
                 oot_indices, time_data_ref, priors)
    # Also initialize in the parent process so `transit_model` evaluations outside
    # the pool (e.g. best-fit rendering) work.
    tm.init_globals(*init_args)

    with mp.Pool(ncpu, initializer=tm.init_globals, initargs=init_args) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, tm.log_probability,
            args=(time_data_hr, flux_data, flux_err),
            pool=pool,
        )
        sampler.run_mcmc(pos, nsteps, progress=False)

    chain = sampler.get_chain()                       # (nsteps, nwalkers, ndim)
    flat = sampler.get_chain(discard=nburn, flat=True)
    best = np.percentile(flat, 50, axis=0)
    lo = np.percentile(flat, 16, axis=0)
    hi = np.percentile(flat, 84, axis=0)
    errors = np.stack([best - lo, hi - best], axis=1)  # (ndim, 2)

    # Render best-fit (in main process — globals already initialized).
    best_fit = tm.transit_model(best, time_data_hr)
    residuals = flux_data - best_fit
    rms = float(np.std(residuals))

    return {
        "best_params": best,
        "best_errors": errors,
        "samples": flat,
        "chain": chain,
        "param_order": param_order,
        "ndim": ndim,
        "acceptance": sampler.acceptance_fraction,
        "rms_residual": rms,
        "best_fit_curve": best_fit,
        "residuals": residuals,
    }


# ----------------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------------

def plot_wl_fit(
    result: dict,
    time_data_hr: np.ndarray,
    flux_data: np.ndarray,
    outdir: str | Path,
    *,
    detector: str | None = None,
    priors: dict | None = None,
) -> dict:
    """Write 3 diagnostic PNGs: corner, best-fit + residuals, chain traces.

    Read the plots
    --------------
    - `corner.png`: marginal posteriors + pairwise. Every parameter should show
      a Gaussian blob. **Flat or railed posteriors = the prior is wrong or the
      data don't constrain the parameter.** In particular: `t0_offset` must have
      structure — if it's flat, run `pytest tswift/tests/test_t0_cache.py`
      before anything else.
    - `best_fit.png`: data (gray), binned data (blue), best-fit model (red),
      residuals below. Residual RMS in the title is the metric; look for
      systematic sinusoidal residuals (limb-darkening mismatch) or bumps at
      ingress/egress (orbital geometry off).
    - `chain.png`: time series of each walker's parameter value through steps.
      After burn-in walkers should be stationary around the median. Divergent
      lines = insufficient burn-in or multimodal posterior.

    Returns map of plot name → absolute path.
    """
    import matplotlib.pyplot as plt
    import corner
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    labels = result["param_order"]
    samples = result["samples"]
    chain = result["chain"]
    ndim = result["ndim"]
    best = result["best_params"]
    det_tag = f" — {detector}" if detector else ""

    paths: dict[str, Path] = {}

    # --- corner ---
    # Let corner auto-scale the axes to the posterior extent — earlier we
    # passed `range=priors`, which forced every panel to show the full prior
    # support.  With the wide priors needed to avoid railing (a:4-20,
    # inc:80-89.99, t0:±1h), the photon-noise-limited posterior shrinks to
    # an invisible spike inside that window.  Rail detection is now done
    # programmatically by the analyze-stage inline check (CLAUDE.md pitfall
    # #17), so the plot doesn't have to double as a rail diagnostic.  If
    # `priors` is supplied, overlay them as faint dashed lines on the
    # diagonal so visual rail-spotting is still easy when the posterior
    # happens to land near an edge.
    fig = corner.corner(
        samples, labels=labels, truths=best,
        show_titles=True, title_fmt=".4f",
    )
    if priors is not None:
        # corner builds an N×N grid of axes; the diagonals are i*N+i.
        axes_grid = np.array(fig.axes).reshape((ndim, ndim))
        for i, name in enumerate(labels):
            if name not in priors:
                continue
            lo, hi = priors[name]
            ax = axes_grid[i, i]
            ax.axvline(lo, color="tab:red", ls=":", lw=0.8, alpha=0.6)
            ax.axvline(hi, color="tab:red", ls=":", lw=0.8, alpha=0.6)
    fig.suptitle(f"WL MCMC posteriors{det_tag}", y=1.02)
    p_corner = outdir / "corner.png"
    fig.savefig(p_corner, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["corner"] = p_corner.resolve()

    # --- best fit + residuals ---
    best_curve = result["best_fit_curve"]
    resid = result["residuals"]
    rms = result["rms_residual"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    axes[0].scatter(time_data_hr, flux_data, s=3, alpha=0.4, color="gray", label="data")
    # binned data
    bin_w = max(1, len(time_data_hr) // 100)
    n_bins = len(time_data_hr) // bin_w
    tb = time_data_hr[: n_bins * bin_w].reshape(-1, bin_w).mean(axis=1)
    fb = flux_data[: n_bins * bin_w].reshape(-1, bin_w).mean(axis=1)
    axes[0].scatter(tb, fb, s=20, color="tab:blue", zorder=5, label=f"binned (×{bin_w})")
    axes[0].plot(time_data_hr, best_curve, "r-", lw=1.5, label="best-fit", zorder=10)
    axes[0].set_ylabel("normalized flux")
    axes[0].legend()
    axes[0].set_title(f"WL best-fit{det_tag}   "
                      f"rp={best[labels.index('rp')]:.4f}   "
                      f"RMS={rms * 1e6:.0f} ppm")

    axes[1].scatter(time_data_hr, resid * 1e6, s=3, alpha=0.4, color="gray")
    rb = resid[: n_bins * bin_w].reshape(-1, bin_w).mean(axis=1)
    axes[1].scatter(tb, rb * 1e6, s=20, color="tab:blue", zorder=5)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set_xlabel("time (hours)")
    axes[1].set_ylabel("residuals (ppm)")
    plt.tight_layout()
    p_fit = outdir / "best_fit.png"
    fig.savefig(p_fit, dpi=120)
    plt.close(fig)
    paths["best_fit"] = p_fit.resolve()

    # --- chain traces ---
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 1.6 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], alpha=0.3, color="0.3", lw=0.5)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("step")
    axes[0].set_title(f"WL chain traces{det_tag}   "
                      f"acceptance {result['acceptance'].mean():.2f}")
    plt.tight_layout()
    p_chain = outdir / "chain.png"
    fig.savefig(p_chain, dpi=120)
    plt.close(fig)
    paths["chain"] = p_chain.resolve()

    return paths


# ============================================================================
# Joint multi-detector fit
# ============================================================================
#
# Module-level state used by the multiprocessing workers.  Mirrors the
# `transit_model._G` pattern but holds the joint-fit data + prior fn.

_JOINT: dict = {}


def joint_param_order(detector_names: list[str], fit_ld1: bool) -> list[str]:
    """Joint-fit parameter order.

    Per-detector params come first (each detector grouped together, in the
    order given by `detector_names`), then the three shared parameters.
    """
    per = ["slope", "rp", "LD1", "LD2", "constant"] if fit_ld1 else \
          ["slope", "rp", "LD2", "constant"]
    order: list[str] = []
    for d in detector_names:
        order.extend(f"{p}_{d}" for p in per)
    order.extend(["a", "inc", "t0_offset"])
    return order


def _split_joint_theta(
    theta: np.ndarray, n_det: int, fit_ld1: bool
) -> tuple[list[np.ndarray], float, float, float]:
    per = 5 if fit_ld1 else 4
    chunks = [theta[i * per:(i + 1) * per] for i in range(n_det)]
    a, inc, t0 = theta[per * n_det:per * n_det + 3]
    return chunks, float(a), float(inc), float(t0)


def _eval_joint_model(theta: np.ndarray) -> list[np.ndarray]:
    """Evaluate one model curve per detector for the given joint theta.

    Reuses `transit_model.transit_model` by mutating `tm._G` per detector —
    this is safe inside one worker because each call within a single
    `log_probability_joint` evaluation is sequential.  The shared params
    (period_hr, ecc, omega, fit_ld1) are set once in `_init_joint_globals`
    and never touched here.
    """
    detectors = _JOINT["detectors"]
    fit_ld1 = _JOINT["fit_ld1"]
    chunks, a, inc, t0 = _split_joint_theta(theta, len(detectors), fit_ld1)

    out: list[np.ndarray] = []
    for det, chunk in zip(detectors, chunks):
        if fit_ld1:
            slope, rp, LD1, LD2, constant = chunk
            single_theta = np.array([slope, rp, LD1, LD2, constant, a, inc, t0])
        else:
            slope, rp, LD2, constant = chunk
            LD1 = det["u1"]
            single_theta = np.array([slope, rp, LD2, constant, a, inc, t0])

        # Per-detector u1/u2/oot/time-ref — must be set before each call.
        tm._G["u1"] = det["u1"]
        tm._G["u2"] = det["u2"]
        tm._G["norm_range"] = det["oot_indices"]
        tm._G["time_data_ref"] = det["time_data_ref"]
        out.append(tm.transit_model(single_theta, det["time_data_hr"]))
    return out


def _log_probability_joint(theta: np.ndarray) -> float:
    lp = _JOINT["log_prior_fn"](theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        models = _eval_joint_model(theta)
    except Exception:
        return -np.inf
    ll = 0.0
    for det, model in zip(_JOINT["detectors"], models):
        sigma2 = det["flux_err"] ** 2
        ll += float(-0.5 * np.sum((det["flux_data"] - model) ** 2 / sigma2
                                  + np.log(2 * np.pi * sigma2)))
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def _init_joint_globals(detectors, fit_ld1, period_hr, ecc, omega, priors):
    """Pool-worker initializer for the joint fit.

    Sets up both `_JOINT` (this module's joint state) and `tm._G` (the
    underlying transit_model module's shared state).  Per-detector u1/u2/
    norm_range/time_data_ref get re-set inside `_eval_joint_model` per
    detector per evaluation.
    """
    _JOINT["detectors"] = detectors
    _JOINT["fit_ld1"] = fit_ld1

    # transit_model module-level shared state (set once per worker).
    tm._G["fit_ld1"] = fit_ld1
    tm._G["ecc"] = ecc
    tm._G["omega"] = omega
    tm._G["period_hr"] = period_hr
    tm._G["_tm_cache"] = {}

    # Joint log-prior — uniform bounds on every parameter in the joint order.
    order = joint_param_order([d["name"] for d in detectors], fit_ld1)
    missing = [p for p in order if p not in priors]
    if missing:
        raise KeyError(f"Joint priors missing for: {missing}")
    bounds = [priors[p] for p in order]

    def log_prior_joint(theta):
        for v, (lo, hi) in zip(theta, bounds):
            if not (lo < v < hi):
                return -np.inf
        return 0.0

    _JOINT["log_prior_fn"] = log_prior_joint


def fit_wl_mcmc_joint(
    detectors: list[dict],
    *,
    period_hr: float,
    ecc: float,
    omega: float,
    initial: dict,
    priors: dict,
    fit_ld1: bool = False,
    nwalkers: int = 64,
    nsteps: int = 10000,
    nburn: int = 4000,
    nprocesses: Optional[int] = None,
    init_scatter: float = 1e-4,
    rng_seed: Optional[int] = None,
) -> dict:
    """Joint white-light MCMC across N detectors with shared orbital geometry.

    Use case
    --------
    G395H NRS1 + NRS2 see the same transit on the same star at the same time.
    Independent per-detector fits typically disagree by ~1 % in `a/Rs` and
    ~0.2° in `inc` (transit-duration degeneracy + photon noise) — that
    propagates into the per-channel `rp` fits as a fake step at the 2.87 ↔
    3.82 µm detector gap.  Sharing geometry across detectors removes the
    step entirely.

    Parameters
    ----------
    detectors : list of dict
        One per detector.  Each dict must contain:

        - ``name`` : str — detector key (e.g. "nrs1"); used as suffix in
          parameter names.
        - ``time_data_hr`` : (N_i,) — time stamps (hours from visit start).
        - ``flux_data``    : (N_i,) — OOT-normalized white-light flux.
        - ``flux_err``     : (N_i,) — per-point uncertainty.
        - ``oot_indices``  : (K_i,) int — indices into the fitted array used
          as the OOT baseline.
        - ``u1``, ``u2``   : float — limb-darkening coefficients for that
          bandpass (used as fixed LD1 when ``fit_ld1=False``, and to
          initialize the LD2 prior centre).
        - ``time_data_ref`` : float, optional — reference time for the
          slope baseline; defaults to median of ``time_data_hr``.
    period_hr, ecc, omega : float
        Shared orbital constants.
    initial : dict
        Flat dict with keys for every parameter in
        ``joint_param_order([d["name"] for d in detectors], fit_ld1)``.
    priors : dict
        Same keys as `initial`, values are ``(lo, hi)`` uniform bounds.
    fit_ld1 : bool
        If True LD1 is a free parameter per detector; if False LD1 is fixed
        to the detector's ``u1``.
    nwalkers, nsteps, nburn : int
        emcee sampler config.  Joint fit needs more walkers and steps than
        per-detector — defaults are 64 / 10000 / 4000.

    Returns
    -------
    dict, with the same keys as `fit_wl_mcmc` plus:

        - ``best_per_det`` : dict[str, dict] — per-detector best-fit params
          mapped to their bare names (slope, rp, LD2, …) for downstream
          spec-fit consumption.
        - ``shared`` : dict — best-fit `a`, `inc`, `t0_offset`.
        - ``rms_residual_per_det`` : dict[str, float]
        - ``best_fit_curve_per_det`` : dict[str, np.ndarray]
        - ``residuals_per_det``      : dict[str, np.ndarray]
    """
    det_names = [d["name"] for d in detectors]
    if len(set(det_names)) != len(det_names):
        raise ValueError(f"Detector names must be unique: {det_names}")

    # Fill defaults + freeze arrays.
    detectors_norm: list[dict] = []
    for d in detectors:
        td = dict(d)
        td["time_data_hr"] = np.asarray(d["time_data_hr"], dtype=float)
        td["flux_data"]    = np.asarray(d["flux_data"], dtype=float)
        td["flux_err"]     = np.asarray(d["flux_err"], dtype=float)
        td["oot_indices"]  = np.asarray(d["oot_indices"], dtype=int)
        td["u1"]           = float(d["u1"])
        td["u2"]           = float(d["u2"])
        td["time_data_ref"] = float(d.get("time_data_ref",
                                          np.median(td["time_data_hr"])))
        detectors_norm.append(td)

    order = joint_param_order(det_names, fit_ld1)
    missing = [p for p in order if p not in initial]
    if missing:
        raise KeyError(f"Joint initial values missing for: {missing}")
    initial_arr = np.array([initial[p] for p in order], dtype=float)

    ndim = len(initial_arr)
    ncpu = nprocesses if nprocesses is not None else mp.cpu_count()

    rng = np.random.default_rng(rng_seed)
    pos = initial_arr + init_scatter * rng.standard_normal((nwalkers, ndim))

    logger.info(
        "Joint WL MCMC: %d detectors %s, %d walkers × %d steps (burn=%d), "
        "ndim=%d, fit_ld1=%s, pool=%d",
        len(detectors_norm), det_names, nwalkers, nsteps, nburn,
        ndim, fit_ld1, ncpu,
    )

    init_args = (detectors_norm, fit_ld1, period_hr, ecc, omega, priors)
    # Initialize in main process too — for best-fit rendering after MCMC.
    _init_joint_globals(*init_args)

    with mp.Pool(ncpu, initializer=_init_joint_globals, initargs=init_args) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, _log_probability_joint, pool=pool,
        )
        sampler.run_mcmc(pos, nsteps, progress=False)

    chain = sampler.get_chain()
    flat = sampler.get_chain(discard=nburn, flat=True)
    log_prob_flat = sampler.get_log_prob(discard=nburn, flat=True)

    # CLAUDE.md pitfall #25: save the MAP (argmax log_prob), not the marginal
    # median.  On the a-inc transit-duration ridge the marginal median lands
    # OFF the ridge, fitting worse than any actual sample.  Multimodal
    # posteriors on the ridge (pitfall #26) make this even more important —
    # MAP picks the dominant mode, median averages between them.
    map_idx = int(np.nanargmax(log_prob_flat))
    best = flat[map_idx].copy()
    median = np.percentile(flat, 50, axis=0)
    lo16 = np.percentile(flat, 16, axis=0)
    hi84 = np.percentile(flat, 84, axis=0)
    # Errors quoted around MAP using ±34% credible interval.
    errors = np.stack([best - lo16, hi84 - best], axis=1)

    # Per-detector best-fit curves + residuals — at MAP.
    models = _eval_joint_model(best)
    best_fit_curve_per_det = {d["name"]: m for d, m in zip(detectors_norm, models)}
    residuals_per_det = {
        d["name"]: d["flux_data"] - m for d, m in zip(detectors_norm, models)
    }
    rms_per_det = {
        n: float(np.std(r)) for n, r in residuals_per_det.items()
    }
    rms_joint = float(np.sqrt(
        sum(np.sum(r ** 2) for r in residuals_per_det.values())
        / sum(len(r) for r in residuals_per_det.values())
    ))

    # Map joint best params → per-detector flat dicts (bare names) for spec.
    chunks, a_b, inc_b, t0_b = _split_joint_theta(best, len(detectors_norm), fit_ld1)
    per_det_names = ["slope", "rp", "LD1", "LD2", "constant"] if fit_ld1 else \
                    ["slope", "rp", "LD2", "constant"]
    best_per_det: dict[str, dict] = {}
    for det, chunk in zip(detectors_norm, chunks):
        best_per_det[det["name"]] = {
            name: float(v) for name, v in zip(per_det_names, chunk)
        }
        # Always include LD1 in the per-det summary (even when fixed) so
        # downstream code doesn't need a fit_ld1 branch.
        if not fit_ld1:
            best_per_det[det["name"]]["LD1"] = float(det["u1"])
    shared = {"a": float(a_b), "inc": float(inc_b), "t0_offset": float(t0_b)}

    return {
        "best_params": best,                     # MAP, used for spec stage
        "median_params": median,                 # marginal median, for reference
        "best_errors": errors,
        "samples": flat,
        "log_prob": log_prob_flat,
        "chain": chain,
        "param_order": order,
        "ndim": ndim,
        "acceptance": sampler.acceptance_fraction,
        "rms_residual": rms_joint,
        "rms_residual_per_det": rms_per_det,
        "best_fit_curve_per_det": best_fit_curve_per_det,
        "residuals_per_det": residuals_per_det,
        "best_per_det": best_per_det,
        "shared": shared,
        "detector_names": det_names,
        "fit_ld1": fit_ld1,
    }


def plot_wl_fit_joint(
    result: dict,
    detectors: list[dict],
    outdir: str | Path,
    *,
    priors: dict | None = None,
) -> dict:
    """Diagnostic PNGs for a joint multi-detector WL fit.

    Writes:

    - ``corner.png``        — full joint posterior (all params).
    - ``corner_shared.png`` — zoomed corner of just `a`, `inc`, `t0_offset`.
    - ``best_fit.png``      — per-detector data + model + residual subplots
      stacked vertically.
    - ``chain.png``         — walker traces for every parameter.
    """
    import matplotlib.pyplot as plt
    import corner
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    labels = result["param_order"]
    samples = result["samples"]
    chain = result["chain"]
    ndim = result["ndim"]
    best = result["best_params"]
    det_names = result["detector_names"]
    paths: dict[str, Path] = {}

    # --- full corner ---
    fig = corner.corner(
        samples, labels=labels, truths=best,
        show_titles=True, title_fmt=".4f",
        label_kwargs={"fontsize": 8},
        title_kwargs={"fontsize": 7},
    )
    if priors is not None:
        axes_grid = np.array(fig.axes).reshape((ndim, ndim))
        for i, name in enumerate(labels):
            if name not in priors:
                continue
            lo, hi = priors[name]
            axes_grid[i, i].axvline(lo, color="tab:red", ls=":", lw=0.6, alpha=0.6)
            axes_grid[i, i].axvline(hi, color="tab:red", ls=":", lw=0.6, alpha=0.6)
    fig.suptitle(f"Joint WL MCMC posteriors — {' + '.join(det_names)}", y=1.005)
    p = outdir / "corner.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    paths["corner"] = p.resolve()

    # --- shared-only corner (a, inc, t0_offset) ---
    shared_labels = ["a", "inc", "t0_offset"]
    shared_idx = [labels.index(n) for n in shared_labels]
    fig = corner.corner(
        samples[:, shared_idx], labels=shared_labels,
        truths=best[shared_idx],
        show_titles=True, title_fmt=".4f",
    )
    if priors is not None:
        axes_grid = np.array(fig.axes).reshape((3, 3))
        for i, name in enumerate(shared_labels):
            if name not in priors:
                continue
            lo, hi = priors[name]
            axes_grid[i, i].axvline(lo, color="tab:red", ls=":", lw=0.8, alpha=0.6)
            axes_grid[i, i].axvline(hi, color="tab:red", ls=":", lw=0.8, alpha=0.6)
    fig.suptitle("Joint WL — shared geometry", y=1.02)
    p = outdir / "corner_shared.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["corner_shared"] = p.resolve()

    # --- per-detector best-fit + residuals ---
    n_det = len(detectors)
    fig, axes = plt.subplots(
        2 * n_det, 1, figsize=(12, 4 * n_det),
        gridspec_kw={"height_ratios": [3, 1] * n_det}, sharex=False,
    )
    if n_det == 1:
        axes = np.atleast_1d(axes)
    for i, det in enumerate(detectors):
        ax_top = axes[2 * i]
        ax_bot = axes[2 * i + 1]
        name = det["name"]
        t = np.asarray(det["time_data_hr"])
        f = np.asarray(det["flux_data"])
        m = np.asarray(result["best_fit_curve_per_det"][name])
        r = np.asarray(result["residuals_per_det"][name])
        rms = result["rms_residual_per_det"][name]

        ax_top.scatter(t, f, s=3, alpha=0.4, color="gray", label="data")
        bw = max(1, len(t) // 100)
        nb = len(t) // bw
        tb = t[: nb * bw].reshape(-1, bw).mean(axis=1)
        fb = f[: nb * bw].reshape(-1, bw).mean(axis=1)
        ax_top.scatter(tb, fb, s=18, color="tab:blue", zorder=5,
                       label=f"binned (×{bw})")
        ax_top.plot(t, m, "r-", lw=1.3, label="best-fit", zorder=10)
        ax_top.set_ylabel("normalized flux")
        ax_top.legend(loc="lower right", fontsize=8)
        ax_top.set_title(
            f"{name}: rp={result['best_per_det'][name]['rp']:.4f}   "
            f"RMS={rms * 1e6:.0f} ppm"
        )

        ax_bot.scatter(t, r * 1e6, s=3, alpha=0.4, color="gray")
        rb = r[: nb * bw].reshape(-1, bw).mean(axis=1)
        ax_bot.scatter(tb, rb * 1e6, s=18, color="tab:blue", zorder=5)
        ax_bot.axhline(0, color="r", ls="--")
        ax_bot.set_ylabel("residuals (ppm)")
        if i == n_det - 1:
            ax_bot.set_xlabel("time (hours)")

    plt.tight_layout()
    p = outdir / "best_fit.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["best_fit"] = p.resolve()

    # --- chain traces ---
    fig, axes = plt.subplots(ndim, 1, figsize=(11, 1.2 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], alpha=0.3, color="0.3", lw=0.4)
        ax.set_ylabel(labels[i], fontsize=8)
    axes[-1].set_xlabel("step")
    axes[0].set_title(
        f"Joint WL chain traces — {' + '.join(det_names)}   "
        f"acceptance {result['acceptance'].mean():.2f}"
    )
    plt.tight_layout()
    p = outdir / "chain.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["chain"] = p.resolve()

    return paths
