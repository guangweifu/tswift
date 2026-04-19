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
    "__version__",
]
