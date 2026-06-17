"""NASA Exoplanet Archive adapter.

Queries the `pscomppars` (planetary systems, composite parameters) table and produces
a populated `Target` object. The composite table aggregates best-available values from
multiple references, which is what we want for a starting point — users can override
in `config.yaml` if they have reason to prefer a specific reference.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

from tswift.contracts import (
    Coordinates,
    OrbitalParams,
    PlanetParams,
    StellarParams,
    Target,
)


# Columns pulled from the pscomppars table. Keep this list narrow — we only want what
# the downstream fit code actually consumes plus a handful of diagnostic fields.
_PSCOMPPARS_COLUMNS = [
    "pl_name", "hostname",
    "ra", "dec",
    "pl_orbper", "pl_orbpererr1",
    "pl_tranmid", "pl_tranmiderr1",
    "pl_ratdor", "pl_ratdorerr1",
    "pl_orbincl", "pl_orbinclerr1",
    "pl_orbeccen",
    "pl_ratror",
    "pl_bmassj", "pl_radj", "pl_eqt",
    "st_teff", "st_tefferr1",
    "st_logg", "st_loggerr1",
    "st_met", "st_meterr1",
    "st_rad", "st_raderr1",
    "st_mass",
]


class TargetNotFoundError(ValueError):
    pass


def query_target(planet_name: str, program: Optional[str] = None) -> Target:
    """Fetch system parameters from NASA Exoplanet Archive and build a Target.

    Parameters
    ----------
    planet_name : str
        Full planet name, e.g. "WASP-69 b". Spaces are required where the canonical
        name has them.
    program : str, optional
        JWST program id. Stored in the Target for provenance; not used for the query.

    Raises
    ------
    TargetNotFoundError
        If the planet is not in pscomppars or returns no rows.
    """
    # astroquery's NasaExoplanetArchive accepts a 'where' clause in ADQL/SQL syntax.
    # Escape quotes just in case: planet names with apostrophes are unlikely but
    # defensive coding is cheap here.
    safe_name = planet_name.replace("'", "''")
    where = f"pl_name='{safe_name}'"
    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select=",".join(_PSCOMPPARS_COLUMNS),
        where=where,
    )

    if len(table) == 0:
        raise TargetNotFoundError(
            f"No rows in pscomppars for planet '{planet_name}'. "
            "Check spelling and required spaces (e.g. 'WASP-69 b', not 'WASP-69b')."
        )
    if len(table) > 1:
        # pscomppars is composite-per-planet so this shouldn't happen, but don't crash.
        pass

    row = table[0]

    def _f(col: str) -> Optional[float]:
        """Coerce masked/empty/Quantity values to a plain float or None."""
        v = row[col]
        try:
            if hasattr(v, "mask") and v.mask:
                return None
        except Exception:
            pass
        # astropy Quantity (e.g. ra/dec carry deg) — strip units
        if hasattr(v, "value"):
            v = v.value
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v != v:  # NaN
            return None
        return v

    def _req(col: str, label: str) -> float:
        v = _f(col)
        if v is None:
            raise ValueError(
                f"Required field '{label}' ({col}) is null for {planet_name}. "
                f"Set it manually in target.json or config.yaml."
            )
        return v

    # Rp/Rs initial: prefer pl_ratror, else derive from Rp (Rjup) / Rs (Rsun)
    rp_rs = _f("pl_ratror")
    if rp_rs is None:
        rp_rj = _f("pl_radj")
        rs_rsun = _f("st_rad")
        if rp_rj is not None and rs_rsun is not None:
            # R_jupiter / R_sun = 0.10045
            rp_rs = (rp_rj * 0.10045) / rs_rsun
    if rp_rs is None:
        raise ValueError(
            f"Cannot determine initial Rp/Rs for {planet_name}: "
            "neither pl_ratror nor (pl_radj + st_rad) are available."
        )

    return Target(
        name=str(row["pl_name"]),
        hostname=str(row["hostname"]),
        program=program or "unknown",
        coordinates=Coordinates(
            ra_deg=_req("ra", "ra"),
            dec_deg=_req("dec", "dec"),
        ),
        stellar=StellarParams(
            teff_k=_req("st_teff", "Teff"),
            teff_k_err=_f("st_tefferr1"),
            logg=_req("st_logg", "log g"),
            logg_err=_f("st_loggerr1"),
            feh=_f("st_met") or 0.0,
            feh_err=_f("st_meterr1"),
            radius_rsun=_req("st_rad", "Rstar"),
            radius_rsun_err=_f("st_raderr1"),
            mass_msun=_f("st_mass"),
        ),
        orbital=OrbitalParams(
            period_days=_req("pl_orbper", "period"),
            period_err_days=_f("pl_orbpererr1"),
            t0_bjd_tdb=_f("pl_tranmid"),
            t0_err_days=_f("pl_tranmiderr1"),
            # a/Rs and inclination are the WL-fit initial guesses; a null value
            # here would coerce to NaN in the MCMC initial vector and silently
            # wreck the fit several stages later, so require them at query time
            # (override manually in target.json for systems the archive lacks).
            a_over_rs=_req("pl_ratdor", "a/Rs"),
            a_over_rs_err=_f("pl_ratdorerr1"),
            inclination_deg=_req("pl_orbincl", "inclination"),
            inclination_deg_err=_f("pl_orbinclerr1"),
            eccentricity=_f("pl_orbeccen") or 0.0,
        ),
        planet=PlanetParams(
            rp_rs_initial=rp_rs,
            mass_mjup=_f("pl_bmassj"),
            radius_rjup=_f("pl_radj"),
            equilibrium_temp_k=_f("pl_eqt"),
        ),
        source="NASA_Exoplanet_Archive/pscomppars",
        queried_at=datetime.now(timezone.utc),
    )
