"""Per-pixel MAD-robust bad-pixel masking.

Why MAD and not an absolute ADU threshold
------------------------------------------
Absolute ADU thresholds are chromatically biased. At the stellar SED peak, a 100-ADU
threshold may be 3σ; in the wings it's 15σ. So the flagging rate varies with wavelength
for no good reason, and replacing flagged pixels with the time-median silently erases
transit signal at bright channels (because the time-median is dominated by out-of-transit
frames). Per-pixel MAD clipping fixes both problems.

The `min_sigma` floor prevents flagging on artificially quiet pixels (e.g. reference
pixels or masked regions).
"""
from __future__ import annotations

import numpy as np


# 1.4826 · MAD is the standard robust estimator of σ for Gaussian noise.
_MAD_TO_SIGMA = 1.4826


def mad_clip(
    data_all: np.ndarray,
    *,
    n_sigma: float = 5.0,
    min_sigma: float = 2.0,
    replace_with_nan: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify and replace bad pixels via per-pixel time-MAD robust clipping.

    Parameters
    ----------
    data_all : np.ndarray, shape (n_frames, n_rows, n_cols)
        Time series of 2D frames (ramp-fit output).
    n_sigma : float, default 5.0
        Threshold in σ units.
    min_sigma : float, default 2.0
        ADU floor to avoid flagging pixels whose MAD is artificially small
        (reference pixels, zero regions, etc.).
    replace_with_nan : bool, default True
        If True, flagged values → NaN. Downstream nansum/nanmedian handles this
        correctly. Set False to replace with the per-pixel time-median (legacy;
        biases transit depth at bright channels — avoid).

    Returns
    -------
    data_fixed : np.ndarray
        Copy of input with flagged pixels replaced.
    bad_mask : np.ndarray, bool, same shape
        True where a sample was flagged.
    sigma_pix : np.ndarray, shape (n_rows, n_cols)
        The per-pixel σ estimate used for thresholding.
    """
    median = np.nanmedian(data_all, axis=0)
    diff = data_all - median[None, :, :]

    # MAD across time. As long as in-transit samples are a minority (< 40% for
    # any JWST transit observation), the MAD is unaffected by transit depth.
    mad = np.nanmedian(np.abs(diff), axis=0)
    sigma_pix = np.maximum(_MAD_TO_SIGMA * mad, min_sigma)
    bad_mask = np.abs(diff) > n_sigma * sigma_pix[None, :, :]

    data_fixed = np.array(data_all, copy=True)
    if replace_with_nan:
        data_fixed[bad_mask] = np.nan
    else:
        data_fixed[bad_mask] = np.broadcast_to(median, data_all.shape)[bad_mask]

    return data_fixed, bad_mask, sigma_pix
