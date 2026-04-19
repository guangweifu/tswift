"""MAST adapter: list JWST program products and download uncal.fits files.

Design notes
------------
- List is cheap and safe to call repeatedly; download is expensive and resumable.
- `list_program_products` returns a plain list of dicts so an agent can reason about
  what's available without needing an astroquery Table.
- `fetch` downloads only `_uncal.fits` files (the starting point for the pipeline),
  verifies sizes against MAST's reported size, and symlinks everything into
  `<project>/data/uncal/` so downstream code can find it.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from astroquery.mast import Observations

logger = logging.getLogger(__name__)


def list_program_products(
    program_id: str,
    target_name: Optional[str] = None,
    instrument: Optional[str] = None,
    product_type: str = "uncal",
) -> list[dict]:
    """Return a list of product metadata for a JWST program.

    Parameters
    ----------
    program_id : str
        JWST proposal ID, e.g. "5924" or "GO-5924" (the GO- prefix is stripped).
    target_name : str, optional
        Filter by target name (MAST's target_name field; e.g. "WASP-69").
    instrument : str, optional
        Filter by instrument: "NIRISS", "NIRSPEC", "MIRI", "NIRCAM".
    product_type : str, default "uncal"
        Pipeline stage suffix to filter product filenames on (uncal/rate/cal/...).

    Returns
    -------
    list[dict]
        One entry per product with keys: obs_id, productFilename, size, dataURI,
        productType, instrument_name, target_name.
    """
    prop_id = program_id.upper().removeprefix("GO-").removeprefix("DD-").removeprefix("GTO-").strip()

    criteria = {"obs_collection": "JWST", "proposal_id": prop_id}
    if target_name:
        criteria["target_name"] = target_name

    obs = Observations.query_criteria(**criteria)
    if len(obs) == 0:
        return []

    # Post-query filter on instrument_name so "NIRISS" matches "NIRISS/SOSS" etc.
    # MAST's exact-match instrument filter is brittle and surprises callers.
    if instrument:
        inst_lower = instrument.lower()
        keep_obs = [i for i, v in enumerate(obs["instrument_name"])
                    if str(v).lower().startswith(inst_lower)]
        if not keep_obs:
            return []
        obs = obs[keep_obs]

    prods = Observations.get_product_list(obs)

    # Attach observation-level target_name / instrument_name to each product so
    # callers can reason about what each file is without a second query.
    obs_map = {str(row["obsid"]): (str(row["target_name"]), str(row["instrument_name"]))
               for row in obs}

    suffix = f"_{product_type}.fits"
    out = []
    for row in prods:
        name = str(row["productFilename"])
        if not name.endswith(suffix):
            continue
        pobsid = str(row["parent_obsid"]) if "parent_obsid" in prods.colnames else None
        tn, ins = obs_map.get(pobsid, (None, None))
        out.append({
            "obs_id": str(row["obs_id"]),
            "productFilename": name,
            "size": int(row["size"]) if row["size"] else None,
            "dataURI": str(row["dataURI"]),
            "productType": str(row["productType"]),
            "instrument_name": ins,
            "target_name": tn,
        })
    return out


def fetch(
    project_dir: Path,
    program_id: str,
    target_name: Optional[str] = None,
    instrument: Optional[str] = None,
    filename_contains: Optional[str] = None,
) -> list[Path]:
    """Download all `_uncal.fits` files for a program into `project_dir/data/uncal/`.

    Downloads go to a MAST cache under `project_dir/data/MAST_Download/` (astroquery's
    natural layout), then every fetched file is symlinked into `data/uncal/` flat.
    Re-running is idempotent: astroquery skips files that already exist with the
    expected size.

    Parameters
    ----------
    project_dir : Path
        Per-planet project directory.
    program_id : str
        JWST proposal ID.
    target_name : str, optional
        Filter by target_name.
    instrument : str, optional
        Filter by instrument (NIRISS/NIRSPEC/etc.).
    filename_contains : str, optional
        Extra filter on the productFilename, e.g. "_04102_" to pull only science
        exposures from a specific observation.

    Returns
    -------
    list[Path]
        Absolute paths to symlinks in `data/uncal/` for every fetched file.
    """
    project_dir = Path(project_dir).resolve()
    uncal_dir = project_dir / "data" / "uncal"
    mast_cache = project_dir / "data" / "MAST_Download"
    uncal_dir.mkdir(parents=True, exist_ok=True)
    mast_cache.mkdir(parents=True, exist_ok=True)

    prop_id = program_id.upper().removeprefix("GO-").removeprefix("DD-").removeprefix("GTO-").strip()
    criteria = {"obs_collection": "JWST", "proposal_id": prop_id}
    if target_name:
        criteria["target_name"] = target_name

    obs = Observations.query_criteria(**criteria)
    if len(obs) == 0:
        raise RuntimeError(f"No JWST observations for program {program_id} with given filters")

    if instrument:
        inst_lower = instrument.lower()
        keep_obs = [i for i, v in enumerate(obs["instrument_name"])
                    if str(v).lower().startswith(inst_lower)]
        if not keep_obs:
            raise RuntimeError(
                f"No observations matching instrument='{instrument}' for program {program_id} "
                f"target={target_name}"
            )
        obs = obs[keep_obs]

    prods = Observations.get_product_list(obs)

    mask = [
        str(name).endswith("_uncal.fits")
        and (filename_contains is None or filename_contains in str(name))
        for name in prods["productFilename"]
    ]
    if not any(mask):
        raise RuntimeError(
            f"No _uncal.fits products match for program {program_id} "
            f"(target={target_name}, instrument={instrument}, contains={filename_contains})"
        )
    to_get = prods[mask]

    logger.info(f"Downloading {len(to_get)} uncal files to {mast_cache}")
    result = Observations.download_products(
        to_get,
        download_dir=str(mast_cache),
    )

    symlinks: list[Path] = []
    for row in result:
        if row["Status"] != "COMPLETE":
            logger.warning(f"Download not complete: {row['Local Path']} ({row['Status']})")
            continue
        local = Path(str(row["Local Path"])).resolve()
        if not local.exists():
            logger.warning(f"Reported local path missing: {local}")
            continue
        link = uncal_dir / local.name
        if link.is_symlink() or link.exists():
            # idempotent: leave existing correct link alone
            try:
                if link.resolve() == local:
                    symlinks.append(link)
                    continue
            except OSError:
                pass
            link.unlink()
        try:
            link.symlink_to(local)
        except OSError as e:
            # Fall back to hardlink if symlinks are disabled (rare on macOS)
            logger.warning(f"Symlink failed ({e}); hardlinking instead")
            os.link(local, link)
        symlinks.append(link)

    logger.info(f"Linked {len(symlinks)} files into {uncal_dir}")
    return symlinks
