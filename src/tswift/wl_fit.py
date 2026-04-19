"""White-light transit MCMC with emcee + multiprocessing.

The fit parameters (theta) follow the order in `transit_model.param_order(fit_ld1)`:

- fit_ld1 = False  (default): ['slope', 'rp', 'LD2', 'constant', 'a', 'inc', 't0_offset']
- fit_ld1 = True              : ['slope', 'rp', 'LD1', 'LD2', 'constant', 'a', 'inc', 't0_offset']

All normalization/time referencing conventions match `transit_model`. In particular,
`oot_indices` is a list of indices into the *fitted* (post-masking) data array that
should serve as the OOT baseline; the model's predicted flux is normalized against
the same indices so residuals are centered on 1.0.

For multiprocessing safety, every worker process is initialized via
`transit_model.init_globals(...)` — the batman TransitModel can't be pickled so we
rebuild it per worker.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
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
