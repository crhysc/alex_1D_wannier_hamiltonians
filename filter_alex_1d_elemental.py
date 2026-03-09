#!/usr/bin/env python3

# python filter_alex_1d_elemental.py \
#   --dataset alex_pbe_1d_all \
#   --store-dir "./alex_pbe_1d_all_cache" \
#   --outdir "./alex_pbe_1d_all_elemental"

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms


def parse_formula_elements(formula: str) -> list[str]:
    """Fallback parser for chemical formula strings like Si2 or FeO."""
    if not formula:
        return []
    return sorted(set(re.findall(r"[A-Z][a-z]?", formula)))


def get_unique_elements(entry: dict) -> list[str]:
    """
    Try several routes to infer the unique chemical species in a JARVIS entry.
    Priority:
      1. atoms dict -> Atoms.from_dict(...).composition
      2. explicit elements field
      3. formula-like fields
    """
    # Best route: use the structure object directly
    atoms_dict = entry.get("atoms")
    if atoms_dict is not None:
        try:
            atoms = Atoms.from_dict(atoms_dict)
            return sorted(atoms.composition.keys())
        except Exception:
            pass

    # Sometimes a flat elements list may exist
    elements = entry.get("elements")
    if elements is not None:
        if isinstance(elements, (list, tuple)):
            return sorted(set(str(x) for x in elements))
        if isinstance(elements, str):
            # handle strings like "Si" or "Si,Si,Si"
            tmp = re.split(r"[\s,]+", elements.strip())
            tmp = [x for x in tmp if x]
            if tmp:
                return sorted(set(tmp))

    # Formula fallbacks
    for key in ("formula", "full_formula", "reduced_formula", "composition"):
        value = entry.get(key)
        if isinstance(value, str):
            elems = parse_formula_elements(value)
            if elems:
                return elems

    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter elemental structures from JARVIS alex_pbe_1d_all."
    )
    parser.add_argument(
        "--dataset",
        default="alex_pbe_1d_all",
        help="JARVIS dataset name (default: alex_pbe_1d_all)",
    )
    parser.add_argument(
        "--store-dir",
        default=None,
        help="Directory where JARVIS caches the downloaded dataset",
    )
    parser.add_argument(
        "--outdir",
        default="alex_pbe_1d_all_elemental",
        help="Output directory",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    
    if args.store_dir is not None:
        store_dir = Path(args.store_dir).resolve()
        store_dir.mkdir(parents=True, exist_ok=True)
        store_dir_str = str(store_dir)
    else:
        store_dir_str = None
    
    print(f"Downloading/loading dataset: {args.dataset}")
    records = data(args.dataset, store_dir=store_dir_str)
    print(f"Total records loaded: {len(records)}")

    elemental_records = []
    counts_by_element = {}

    for entry in records:
        uniq = get_unique_elements(entry)
        if len(uniq) == 1:
            el = uniq[0]
            elemental_records.append(entry)
            counts_by_element[el] = counts_by_element.get(el, 0) + 1

    elemental_json = outdir / "elemental_alex_pbe_1d_all.json"
    elemental_jids = outdir / "elemental_jids.txt"
    summary_json = outdir / "summary.json"

    with elemental_json.open("w") as f:
        json.dump(elemental_records, f)

    with elemental_jids.open("w") as f:
        for entry in elemental_records:
            jid = entry.get("jid", entry.get("id", "UNKNOWN_ID"))
            f.write(f"{jid}\n")

    summary = {
        "dataset": args.dataset,
        "total_records": len(records),
        "elemental_records": len(elemental_records),
        "counts_by_element": dict(sorted(counts_by_element.items())),
        "output_files": {
            "elemental_json": str(elemental_json),
            "elemental_jids": str(elemental_jids),
            "summary_json": str(summary_json),
        },
    }

    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Elemental records found: {len(elemental_records)}")
    print(f"Wrote: {elemental_json}")
    print(f"Wrote: {elemental_jids}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
