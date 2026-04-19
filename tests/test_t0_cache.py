"""Regression test for the batman t0-cache bug.

If this ever fails, it means the TransitModel cache is again baking t0 into the phase
grid — which freezes the walker's t0_offset parameter without warning. See the long
comment in tswift/transit_model.py for the history.

Run:
    pytest tswift/tests/test_t0_cache.py -q
or ad hoc:
    python tswift/tests/test_t0_cache.py
"""
from __future__ import annotations

import numpy as np

from tswift.transit_model import init_globals, transit_model


def _setup():
    # Constants representative of WASP-69 b (doesn't need to be exact).
    period_hr = 3.8681 * 24.0
    # 4 hours of data spanning the transit centered at 2 hours.
    time_hr = np.linspace(0.0, 4.0, 400)
    # Fully populated globals; normalize to the first 30 points (pre-ingress-ish).
    init_globals(
        0.3,               # u1
        0.25,              # u2
        False,             # fit_ld1
        period_hr,
        0.0,               # ecc
        90.0,              # omega
        np.arange(30),     # norm_range
        2.0,               # time_data_ref
    )
    return time_hr


# Parameter order (fit_ld1=False): [slope, rp, LD2, constant, a, inc, t0_offset]
# The fit sometimes uses fit_ld1=True: [slope, rp, LD1, LD2, constant, a, inc, t0_offset].
# We test the no-LD1 variant (the default).

def _theta(t0_offset: float):
    return [0.0, 0.13, 0.25, 1.0, 12.0, 86.8, t0_offset]


def test_t0_changes_flux():
    """Varying t0_offset MUST change the predicted flux."""
    time_hr = _setup()
    flux_early = transit_model(_theta(1.5), time_hr)
    flux_late  = transit_model(_theta(2.5), time_hr)

    delta = np.max(np.abs(flux_early - flux_late))
    assert delta > 0.01, (
        f"t0 has no effect on flux (max delta {delta:.3e}). "
        "The TransitModel cache is probably baking t0 into the phase grid — see "
        "tswift/transit_model.py docstring."
    )


def test_identical_t0_identical_flux():
    """Baseline consistency — same inputs should give bit-identical outputs."""
    time_hr = _setup()
    f1 = transit_model(_theta(2.0), time_hr)
    f2 = transit_model(_theta(2.0), time_hr)
    assert np.array_equal(f1, f2), "transit_model is not deterministic"


def test_rp_changes_depth():
    """Sanity: larger rp should produce a deeper transit."""
    time_hr = _setup()

    def _theta_rp(rp):
        return [0.0, rp, 0.25, 1.0, 12.0, 86.8, 2.0]

    shallow = transit_model(_theta_rp(0.10), time_hr)
    deep    = transit_model(_theta_rp(0.14), time_hr)
    assert shallow.min() > deep.min(), "deeper rp did not produce a deeper transit"


if __name__ == "__main__":
    test_t0_changes_flux()
    test_identical_t0_identical_flux()
    test_rp_changes_depth()
    print("All t0-cache regression tests passed.")
