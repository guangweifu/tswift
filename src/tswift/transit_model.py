"""Batman transit model wrapper with CORRECT t0 caching.

Background — the bug this module is written to prevent
-------------------------------------------------------
`batman.TransitModel` precomputes geometry on a phase grid. If that grid depends on
the fitted parameter `t0`, it must be recomputed every evaluation (slow). The legacy
code cached it by `id(time_array)` but baked `t0` into `time_phase = (time - t0) /
period_hr`, so the cache key didn't include t0 but the cached phase grid did.

Result: every walker saw the same cached phase grid, which corresponded to some
**fixed** t0. The `t0_offset` parameter in `theta` changed, but had no effect on the
predicted flux. The posterior on `t0_offset` was flat, even though the model
reported a seemingly-reasonable best-fit value.

The fix
-------
- `time_phase = time_data / period_hr` — t0-independent, cacheable.
- `params.t0 = t0_offset / period_hr` passed through batman's native t0 handling,
  which shifts the flux without touching the cached geometry.

A regression test lives in `tests/test_t0_cache.py` that asserts varying `t0` in
`theta` changes the output flux by at least 1%. Keep it green.

Multiprocessing note
--------------------
emcee's multiprocessing pools run log_prob in worker processes that can't share the
parent's cached TransitModel. We use module-level state initialized per worker via
`init_globals(...)` passed as `initializer` to `multiprocessing.Pool`.
"""
from __future__ import annotations

from typing import Optional

import batman
import numpy as np


# Module-level worker state. Populated by `init_globals` at the start of each emcee
# worker process. Do not touch directly from user code; use the public API.
_G: dict = {}


def _param_order(fit_ld1: bool) -> list[str]:
    if fit_ld1:
        return ["slope", "rp", "LD1", "LD2", "constant", "a", "inc", "t0_offset"]
    return ["slope", "rp", "LD2", "constant", "a", "inc", "t0_offset"]


def init_globals(
    u1: float,
    u2: float,
    fit_ld1: bool,
    period_hr: float,
    ecc: float,
    omega: float,
    norm_range: np.ndarray | list,
    time_data_ref: float,
    priors: Optional[dict] = None,
) -> None:
    """Initialize module state — called once per worker process.

    Parameters
    ----------
    u1, u2 : float
        Fixed limb-darkening coefficients for the bandpass. `u1` is used as LD1 if
        `fit_ld1=False`; both are always passed through to batman.
    fit_ld1 : bool
        If True, LD1 is a free parameter in theta; if False, LD1 is fixed to `u1`.
    period_hr : float
        Orbital period in hours.
    ecc, omega : float
        Eccentricity and argument of periastron (degrees). Both fixed, not fitted.
    norm_range : array-like of int
        Indices (into the FITTED array after masking) used to normalize the model's
        flux to OOT median — must match how data was normalized.
    time_data_ref : float
        Reference time (hours from visit start) for the linear slope baseline.
    priors : dict, optional
        Per-parameter prior bounds as `{name: (lo, hi)}`. Required if this worker
        will evaluate `log_probability`; optional if only forward-modeling.
    """
    _G["u1"] = u1
    _G["u2"] = u2
    _G["fit_ld1"] = fit_ld1
    _G["ecc"] = ecc
    _G["omega"] = omega
    _G["period_hr"] = period_hr
    _G["norm_range"] = np.asarray(norm_range, dtype=int) if len(norm_range) else np.array([], dtype=int)
    _G["time_data_ref"] = time_data_ref
    _G["_tm_cache"] = {}
    if priors is not None:
        _G["log_prior_fn"] = _make_log_prior(priors, fit_ld1)


def _make_log_prior(priors: dict, fit_ld1: bool):
    order = _param_order(fit_ld1)
    missing = [p for p in order if p not in priors]
    if missing:
        raise KeyError(f"Missing priors for: {missing}")
    bounds = [priors[p] for p in order]

    def log_prior(theta):
        for v, (lo, hi) in zip(theta, bounds):
            if not (lo < v < hi):
                return -np.inf
        return 0.0

    return log_prior


def transit_model(theta: np.ndarray, time_data_hr: np.ndarray) -> np.ndarray:
    """Evaluate the transit light curve at `time_data_hr` with parameters `theta`.

    Follows the t0-cache design in the module docstring — DO NOT change the cache
    key or phase grid without also updating the regression test.
    """
    fit_ld1: bool = _G["fit_ld1"]
    if fit_ld1:
        slope, rp, LD1, LD2, constant, a, inc, t0_offset = theta
    else:
        slope, rp, LD2, constant, a, inc, t0_offset = theta
        LD1 = _G["u1"]

    period_hr = _G["period_hr"]

    # t0-INDEPENDENT phase grid. This is what makes caching across walkers safe.
    time_phase = time_data_hr / period_hr

    # Cache the TransitModel geometry per unique time array (id + length).
    # Walker's t0 goes through params.t0 below, not through time_phase.
    cache_key = (id(time_data_hr), len(time_data_hr))
    if cache_key not in _G["_tm_cache"]:
        p0 = batman.TransitParams()
        p0.t0 = time_data_hr[len(time_data_hr) // 2] / period_hr  # arbitrary mid-point
        p0.per = 1.0
        p0.rp = 0.1
        p0.a = 10.0
        p0.inc = 88.0
        p0.ecc = _G["ecc"]
        p0.w = _G["omega"]
        p0.u = [0.3, 0.3]
        p0.limb_dark = "quadratic"
        _G["_tm_cache"][cache_key] = batman.TransitModel(p0, time_phase)

    m = _G["_tm_cache"][cache_key]

    # Per-walker params. t0 goes through batman's native handling — this is the
    # ONLY way the walker's t0 actually influences flux.
    p = batman.TransitParams()
    p.t0 = t0_offset / period_hr
    p.per = 1.0
    p.rp = rp
    p.a = a
    p.inc = inc
    p.ecc = _G["ecc"]
    p.w = _G["omega"]
    p.u = [LD1, LD2]
    p.limb_dark = "quadratic"

    flux = m.light_curve(p)

    # Normalize model by OOT median, same as the data normalization.
    norm_range = _G["norm_range"]
    if norm_range.size > 0:
        valid = norm_range[norm_range < len(flux)]
        norm = np.median(flux[valid]) if valid.size else 1.0
    else:
        norm = 1.0

    flux_norm = flux / norm
    time_ref = _G["time_data_ref"]
    return flux_norm * constant + slope * (time_data_hr - time_ref)


def log_likelihood(theta, time_data_hr, flux_data, flux_err) -> float:
    model = transit_model(theta, time_data_hr)
    sigma2 = flux_err ** 2
    return float(-0.5 * np.sum((flux_data - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2)))


def log_probability(theta, time_data_hr, flux_data, flux_err) -> float:
    lp = _G["log_prior_fn"](theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, time_data_hr, flux_data, flux_err)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def param_order(fit_ld1: bool) -> list[str]:
    """Public accessor for the parameter order used in `theta` arrays."""
    return _param_order(fit_ld1)
