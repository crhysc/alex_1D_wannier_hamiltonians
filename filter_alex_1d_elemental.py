#!/usr/bin/env python3

# python filter_alex_1d_elemental.py \
#   --dataset alex_pbe_1d_all \
#   --store-dir "$SCRATCH/alex_pbe_1d_all_cache" \
#   --outdir "$SCRATCH/alex_pbe_1d_all_elemental"

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms


FORMULA_KEYS = ("formula", "full_formula", "reduced_formula", "composition")
PRIMARY_METHOD = "atoms.uniq_species + atoms.composition.nspecies"


def record_id(entry: dict[str, Any], idx: int) -> str:
    """Return the most informative record identifier available."""
    return str(entry.get("jid", entry.get("id", f"INDEX_{idx}")))


def parse_formula_elements(formula: str) -> list[str]:
    """Fallback parser for chemical formula strings like Si2 or FeO."""
    if not formula:
        return []
    return sorted(set(re.findall(r"[A-Z][a-z]?", formula)))


def infer_species(entry: dict[str, Any], idx: int) -> dict[str, Any]:
    """
    Infer species using the exact JARVIS-native path first.

    Priority:
      1. atoms dict -> Atoms.from_dict(...).uniq_species and composition.nspecies
      2. explicit elements field
      3. formula-like string fields
    """
    rid = record_id(entry, idx)
    result: dict[str, Any] = {
        "record_id": rid,
        "method": None,
        "uniq_species": [],
        "nspecies": None,
        "elemental": False,
        "warnings": [],
    }

    atoms_dict = entry.get("atoms")
    if atoms_dict is not None:
        try:
            atoms = Atoms.from_dict(atoms_dict)
            uniq_species = [str(x) for x in atoms.uniq_species]
            nspecies = int(atoms.composition.nspecies)

            result["method"] = PRIMARY_METHOD
            result["uniq_species"] = uniq_species
            result["nspecies"] = nspecies
            result["elemental"] = nspecies == 1

            uniq_count = len(set(uniq_species))
            if uniq_count != nspecies:
                result["warnings"].append(
                    "Mismatch between len(set(atoms.uniq_species)) "
                    f"({uniq_count}) and atoms.composition.nspecies ({nspecies})."
                )

            if nspecies == 0:
                result["warnings"].append(
                    "Parsed Atoms object but composition.nspecies == 0."
                )

            return result

        except Exception as exc:
            result["warnings"].append(
                "Failed JARVIS atoms parse via "
                "Atoms.from_dict(entry['atoms']): "
                f"{type(exc).__name__}: {exc}"
            )
    else:
        result["warnings"].append("Missing 'atoms' field.")

    elements = entry.get("elements")
    if elements is not None:
        if isinstance(elements, (list, tuple)):
            uniq_species = sorted(set(str(x) for x in elements))
            result["method"] = "elements field"
            result["uniq_species"] = uniq_species
            result["nspecies"] = len(uniq_species)
            result["elemental"] = len(uniq_species) == 1
            result["warnings"].append(
                "Used fallback 'elements' field because JARVIS atoms parsing "
                "was unavailable or failed."
            )
            return result

        if isinstance(elements, str):
            tmp = re.split(r"[\s,]+", elements.strip())
            tmp = [x for x in tmp if x]
            uniq_species = sorted(set(tmp))
            result["method"] = "elements string field"
            result["uniq_species"] = uniq_species
            result["nspecies"] = len(uniq_species)
            result["elemental"] = len(uniq_species) == 1
            result["warnings"].append(
                "Used fallback string 'elements' field because JARVIS atoms "
                "parsing was unavailable or failed."
            )
            return result

        result["warnings"].append(
            f"Field 'elements' exists but has unsupported type {type(elements).__name__}."
        )

    for key in FORMULA_KEYS:
        value = entry.get(key)
        if isinstance(value, str):
            uniq_species = parse_formula_elements(value)
            if uniq_species:
                result["method"] = f"{key} formula parse"
                result["uniq_species"] = uniq_species
                result["nspecies"] = len(uniq_species)
                result["elemental"] = len(uniq_species) == 1
                result["warnings"].append(
                    f"Used fallback formula parsing from '{key}' because JARVIS "
                    "atoms parsing was unavailable or failed."
                )
                return result

    result["method"] = "unresolved"
    result["warnings"].append(
        "Could not infer species from atoms/elements/formula fields."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter elemental structures from a JARVIS dataset."
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

    if not isinstance(records, list):
        raise TypeError(
            f"Expected JARVIS dataset '{args.dataset}' to load as a list of records, "
            f"but got {type(records).__name__}."
        )

    print(f"Total records loaded: {len(records)}")

    elemental_records: list[dict[str, Any]] = []
    counts_by_element: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()

    issue_log: list[dict[str, Any]] = []
    unresolved_log: list[dict[str, Any]] = []
    fallback_log: list[dict[str, Any]] = []

    sample_warning_limit = 20
    sample_warning_count = 0

    for idx, entry in enumerate(records):
        if not isinstance(entry, dict):
            raise TypeError(
                f"Record at index {idx} is not a dict; got {type(entry).__name__}."
            )

        info = infer_species(entry, idx)
        rid = info["record_id"]
        method = str(info["method"])
        uniq_species = list(info["uniq_species"])
        nspecies = info["nspecies"]
        warnings = list(info["warnings"])

        method_counts[method] += 1

        if warnings:
            issue_item = {
                "record_id": rid,
                "method": method,
                "nspecies": nspecies,
                "uniq_species": uniq_species,
                "warnings": warnings,
                "available_keys": sorted(entry.keys()),
            }
            issue_log.append(issue_item)

            if sample_warning_count < sample_warning_limit:
                print(
                    f"WARNING [{rid}] method={method} warnings={warnings}",
                    file=sys.stderr,
                )
                sample_warning_count += 1

        if method != PRIMARY_METHOD:
            fallback_log.append(
                {
                    "record_id": rid,
                    "method": method,
                    "nspecies": nspecies,
                    "uniq_species": uniq_species,
                }
            )

        if method == "unresolved":
            unresolved_log.append(
                {
                    "record_id": rid,
                    "available_keys": sorted(entry.keys()),
                    "warnings": warnings,
                }
            )

        if nspecies == 1 and uniq_species:
            el = uniq_species[0]
            elemental_records.append(entry)
            counts_by_element[el] += 1

    elemental_json = outdir / "elemental_alex_pbe_1d_all.json"
    elemental_jids = outdir / "elemental_jids.txt"
    summary_json = outdir / "summary.json"
    issue_log_json = outdir / "species_inference_issues.json"
    fallback_log_json = outdir / "species_inference_fallbacks.json"
    unresolved_log_json = outdir / "species_inference_unresolved.json"

    with elemental_json.open("w") as f:
        json.dump(elemental_records, f)

    with elemental_jids.open("w") as f:
        for entry in elemental_records:
            jid = entry.get("jid", entry.get("id", "UNKNOWN_ID"))
            f.write(f"{jid}\n")

    with issue_log_json.open("w") as f:
        json.dump(issue_log, f, indent=2)

    with fallback_log_json.open("w") as f:
        json.dump(fallback_log, f, indent=2)

    with unresolved_log_json.open("w") as f:
        json.dump(unresolved_log, f, indent=2)

    summary = {
        "dataset": args.dataset,
        "total_records": len(records),
        "elemental_records": len(elemental_records),
        "counts_by_element": dict(sorted(counts_by_element.items())),
        "species_inference_method_counts": dict(method_counts),
        "n_records_with_warnings": len(issue_log),
        "n_records_using_fallback_or_unresolved": len(fallback_log),
        "n_unresolved_records": len(unresolved_log),
        "output_files": {
            "elemental_json": str(elemental_json),
            "elemental_jids": str(elemental_jids),
            "summary_json": str(summary_json),
            "issue_log_json": str(issue_log_json),
            "fallback_log_json": str(fallback_log_json),
            "unresolved_log_json": str(unresolved_log_json),
        },
    }

    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Elemental records found: {len(elemental_records)}")
    print(f"Species inference method counts: {dict(method_counts)}")
    print(f"Records with warnings: {len(issue_log)}")
    print(f"Records using fallback/unresolved paths: {len(fallback_log)}")
    print(f"Unresolved records: {len(unresolved_log)}")
    print(f"Wrote: {elemental_json}")
    print(f"Wrote: {elemental_jids}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {issue_log_json}")
    print(f"Wrote: {fallback_log_json}")
    print(f"Wrote: {unresolved_log_json}")


if __name__ == "__main__":
    main()
