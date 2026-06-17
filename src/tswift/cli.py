"""Command-line interface for the offline-safe, one-shot tswift steps.

    tswift bootstrap "WASP-69 b" 5924 [--outdir ./WASP-69b]
    tswift list 5924 [--target WASP-69] [--instrument NIRISS]
    tswift fetch ./WASP-69b 5924 [--target WASP-69] [--instrument NIRISS] [--contains _04102_]

The iterative analysis stages (calibrate, extract, fit, combine, ...) are
intentionally NOT exposed here: each one writes a diagnostic PNG you are meant to
inspect before deciding whether to continue or retune, so they live in the Python
API and your per-target ``analyze.py`` driver. See RUNBOOK.md and examples/.
"""
from __future__ import annotations

import argparse
import logging
import sys


def _cmd_bootstrap(args) -> int:
    from tswift.bootstrap import bootstrap

    path = bootstrap(args.planet, args.program, outdir=args.outdir)
    print(path)
    return 0


def _cmd_list(args) -> int:
    from tswift.mast import list_program_products

    prods = list_program_products(
        args.program, target_name=args.target, instrument=args.instrument
    )
    if not prods:
        print("No uncal products found for that query.", file=sys.stderr)
        return 1
    print(f"{len(prods)} uncal product(s):")
    for p in prods:
        size_mb = (p.get("size") or 0) / 1e6
        print(
            f"  {str(p.get('obs_id', '?')):28s} "
            f"{str(p.get('target_name')):16s} "
            f"{str(p.get('instrument')):10s} "
            f"{p.get('productFilename', '?')}  ({size_mb:.0f} MB)"
        )
    return 0


def _cmd_fetch(args) -> int:
    from tswift.mast import fetch

    files = fetch(
        args.project_dir,
        args.program,
        target_name=args.target,
        instrument=args.instrument,
        filename_contains=args.contains,
    )
    print(f"Downloaded {len(files)} file(s) into {args.project_dir}/data/uncal/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tswift",
        description="JWST transit spectroscopy pipeline — project setup + MAST commands. "
        "Run the analysis stages from Python (see RUNBOOK.md / examples/).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser(
        "bootstrap",
        help="Create a project dir + target.json from the NASA Exoplanet Archive.",
    )
    b.add_argument("planet", help='Planet name, e.g. "WASP-69 b"')
    b.add_argument("program", help="JWST program id, e.g. 5924")
    b.add_argument("--outdir", default=None, help="Project dir (default ./<planet>)")
    b.set_defaults(func=_cmd_bootstrap)

    ls = sub.add_parser(
        "list", help="List uncal products for a program (cheap; no download)."
    )
    ls.add_argument("program", help="JWST program id")
    ls.add_argument("--target", default=None, help="Host star name to filter on")
    ls.add_argument("--instrument", default=None, help="NIRISS | NIRSpec | MIRI")
    ls.set_defaults(func=_cmd_list)

    f = sub.add_parser(
        "fetch", help="Download uncal.fits for a program into a project dir."
    )
    f.add_argument("project_dir", help="Project dir created by `tswift bootstrap`")
    f.add_argument("program", help="JWST program id")
    f.add_argument("--target", default=None, help="Host star name to filter on")
    f.add_argument("--instrument", default=None, help="NIRISS | NIRSpec | MIRI")
    f.add_argument(
        "--contains",
        default=None,
        help="Only files whose name contains this (e.g. _04102_)",
    )
    f.set_defaults(func=_cmd_fetch)

    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
