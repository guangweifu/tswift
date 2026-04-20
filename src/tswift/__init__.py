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
from tswift.mast import fetch, list_program_products, ensure_mast_login, load_mast_token
from tswift.bad_pixel import mad_clip, plot_bad_pixel
from tswift.extract import (
    find_trace,
    extract_trace_2d,
    optimize_aperture,
    clean_per_channel,
    run_extract,
    save_extract_outputs,
    plot_trace,
    plot_aperture_scan,
    plot_clean,
)
from tswift.wl_fit import fit_wl_mcmc, plot_wl_fit
from tswift.spec_fit import compute_ld_per_wavelength, fit_spec_curves, plot_spec_fit
from tswift.combine import (
    rp_to_depth_ppm,
    bin_inverse_variance,
    combine_spectrum,
    combine_detectors,
)
from tswift.spectrum import plot_spectrum, save_spectrum

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
    "ensure_mast_login",
    "load_mast_token",
    "mad_clip",
    "plot_bad_pixel",
    "find_trace",
    "extract_trace_2d",
    "optimize_aperture",
    "clean_per_channel",
    "run_extract",
    "save_extract_outputs",
    "plot_trace",
    "plot_aperture_scan",
    "plot_clean",
    "fit_wl_mcmc",
    "plot_wl_fit",
    "compute_ld_per_wavelength",
    "fit_spec_curves",
    "plot_spec_fit",
    "rp_to_depth_ppm",
    "bin_inverse_variance",
    "combine_spectrum",
    "combine_detectors",
    "plot_spectrum",
    "save_spectrum",
    "__version__",
]
