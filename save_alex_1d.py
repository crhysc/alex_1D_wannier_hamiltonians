#!/usr/bin/env python3

# python filter_alex_1d_elemental.py \
#   --dataset alex_pbe_1d_all \
#   --store-dir "$SCRATCH/alex_pbe_1d_all_cache" \
#   --outdir "$SCRATCH/alex_pbe_1d_all_all_structures"

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms


def record_id(entry: dict[str, Any], idx: int) -> str:
    """Return the most informative record identifier available."""
    return str(entry.get("jid", entry.get("id", f"INDEX_{idx}")))


def validate_record(entry: dict[str, Any], idx: int) -> dict[str, Any]:
    """
    Validate that the record is a dict and that its atoms payload
    can be reconstructed with JARVIS.
    """
    if not isinstance(entry, dict):
        raise TypeError(
            f"Record at index {idx} is not a dict; got {type(entry).__name__}."
        )

    rid = record_id(entry, idx)

    atoms_dict = entry.get("atoms")
    if atoms_dict is None:
        raise KeyError(f"Record {rid} is missing the 'atoms' field.")

    try:
        atoms = Atoms.from_dict(atoms_dict)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse atoms for record {rid} with "
            f"Atoms.from_dict(entry['atoms']): {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "record_id": rid,
        "num_atoms": int(atoms.num_atoms),
        "uniq_species": [str(x) for x in atoms.uniq_species],
        "nspecies": int(atoms.composition.nspecies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and store all structures from a JARVIS dataset."
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
        default="alex_pbe_1d_all_all_structures",
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

    validation_log: list[dict[str, Any]] = []
    for idx, entry in enumerate(records):
        info = validate_record(entry, idx)
        validation_log.append(info)

    all_structures_json = outdir / f"{args.dataset}_all_structures.json"
    all_jids_txt = outdir / "all_jids.txt"
    validation_json = outdir / "validation_summary.json"
    summary_json = outdir / "summary.json"

    with all_structures_json.open("w") as f:
        json.dump(records, f)

    with all_jids_txt.open("w") as f:
        for idx, entry in enumerate(records):
            f.write(f"{record_id(entry, idx)}\n")

    with validation_json.open("w") as f:
        json.dump(validation_log, f, indent=2)

    species_histogram: dict[str, int] = {}
    for item in validation_log:
        key = str(item["nspecies"])
        species_histogram[key] = species_histogram.get(key, 0) + 1

    summary = {
        "dataset": args.dataset,
        "total_records": len(records),
        "validated_records": len(validation_log),
        "species_count_histogram": dict(
            sorted(species_histogram.items(), key=lambda kv: int(kv[0]))
        ),
        "output_files": {
            "all_structures_json": str(all_structures_json),
            "all_jids_txt": str(all_jids_txt),
            "validation_json": str(validation_json),
            "summary_json": str(summary_json),
        },
    }

    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Stored all records: {len(records)}")
    print(f"Wrote: {all_structures_json}")
    print(f"Wrote: {all_jids_txt}")
    print(f"Wrote: {validation_json}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
