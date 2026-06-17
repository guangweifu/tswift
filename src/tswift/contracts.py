"""Pydantic data contracts for tswift.

`Target` is the load-bearing schema: system parameters auto-populated from the
NASA Exoplanet Archive (`target_db.query_target`) and consumed by every fitting
stage. Write a target.json by hand if you must, but prefer `bootstrap()` and
override individual fields via `config.yaml`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------------------
# Target
# ----------------------------------------------------------------------------

class Coordinates(BaseModel):
    ra_deg: float
    dec_deg: float


class StellarParams(BaseModel):
    teff_k: float
    teff_k_err: Optional[float] = None
    logg: float
    logg_err: Optional[float] = None
    feh: float = Field(description="[Fe/H] metallicity")
    feh_err: Optional[float] = None
    radius_rsun: float
    radius_rsun_err: Optional[float] = None
    mass_msun: Optional[float] = None


class OrbitalParams(BaseModel):
    period_days: float
    period_err_days: Optional[float] = None
    t0_bjd_tdb: Optional[float] = Field(
        default=None, description="Reference transit mid-time (BJD_TDB)"
    )
    t0_err_days: Optional[float] = None
    a_over_rs: Optional[float] = None
    a_over_rs_err: Optional[float] = None
    inclination_deg: Optional[float] = None
    inclination_deg_err: Optional[float] = None
    eccentricity: float = 0.0
    omega_deg: float = 90.0


class PlanetParams(BaseModel):
    rp_rs_initial: float = Field(description="Initial guess for Rp/Rs in fits")
    mass_mjup: Optional[float] = None
    radius_rjup: Optional[float] = None
    equilibrium_temp_k: Optional[float] = None


class Target(BaseModel):
    """Everything about the system an agent needs to set up fits and interpret results.

    Auto-populated by `tswift.target_db.query_target(planet_name)`. Writing a target.json
    by hand is supported but discouraged — prefer `bootstrap()` and then override
    individual fields via `config.yaml` if the archive values are wrong.
    """

    name: str = Field(description="Planet name, e.g. 'WASP-69 b'")
    hostname: str = Field(description="Host star name, e.g. 'WASP-69'")
    program: str = Field(description="JWST program id, e.g. 'GO-5924' or '5924'")
    coordinates: Coordinates
    stellar: StellarParams
    orbital: OrbitalParams
    planet: PlanetParams
    source: str = "NASA_Exoplanet_Archive/pscomppars"
    queried_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("queried_at", mode="before")
    @classmethod
    def _parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "Target":
        return cls.model_validate_json(Path(path).read_text())
