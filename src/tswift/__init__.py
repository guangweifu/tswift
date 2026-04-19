"""tswift — AI-agent-native JWST transit spectroscopy pipeline."""
from tswift.contracts import (
    Target,
    StellarParams,
    OrbitalParams,
    PlanetParams,
    Coordinates,
    Manifest,
    Diagnostics,
    CheckResult,
    FileRef,
)
from tswift.bootstrap import bootstrap
from tswift.mast import fetch, list_program_products
from tswift.bad_pixel import mad_clip
from tswift.extract import (
    find_trace,
    extract_trace_2d,
    optimize_aperture,
    clean_per_channel,
    run_extract,
    save_extract_outputs,
)
from tswift.wl_fit import fit_wl_mcmc
from tswift.spec_fit import compute_ld_per_wavelength, fit_spec_curves
from tswift.combine import (
    rp_to_depth_ppm,
    bin_inverse_variance,
    combine_spectrum,
    combine_detectors,
)

__version__ = "2.0.0.dev0"

__all__ = [
    "Target",
    "StellarParams",
    "OrbitalParams",
    "PlanetParams",
    "Coordinates",
    "Manifest",
    "Diagnostics",
    "CheckResult",
    "FileRef",
    "bootstrap",
    "fetch",
    "list_program_products",
    "mad_clip",
    "find_trace",
    "extract_trace_2d",
    "optimize_aperture",
    "clean_per_channel",
    "run_extract",
    "save_extract_outputs",
    "fit_wl_mcmc",
    "compute_ld_per_wavelength",
    "fit_spec_curves",
    "rp_to_depth_ppm",
    "bin_inverse_variance",
    "combine_spectrum",
    "combine_detectors",
    "__version__",
]
