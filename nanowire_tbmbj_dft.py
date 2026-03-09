#!/usr/bin/env python3
"""
Build status metadata:
python nanowire_tbmbj_dft.py refresh-status \
  --jsonl alex_pbe_1d_all_all_structures.jsonl \
  --superdir $SCRATCH/nanowire_tbmbj_dataset

Submit DFT jobs to cluster:
python nanowire_tbmbj_dft.py submit \
  --jsonl alex_pbe_1d_all_all_structures.jsonl \
  --superdir $SCRATCH/nanowire_tbmbj_dataset \
  --vasp-cmd "srun vasp_std" \
  --queue normal \
  --account my_account \
  --cores 32 \
  --memory 128G \
  --walltime 24:00:00 \
  --pre-job-lines $'module load vasp\nexport VASP_PSP_DIR=/potcars' \
  --submit

Collect final dataset:
python nanowire_tbmbj_resume.py collect \
  --jsonl alex_pbe_1d_all_all_structures.jsonl \
  --superdir $SCRATCH/nanowire_tbmbj_dataset \
  --out-jsonl $SCRATCH/nanowire_tbmbj_dataset/tbmbj_dataset.jsonl

This script manages a large set of single-shot TBmBJ VASP calculations for
1D structures stored in a JSONL file. The main() function does not run one
fixed workflow by itself. Instead, it reads a command from the terminal and
then sends the script into one of four modes: submit, refresh-status,
run-case, or collect.

If main() receives the submit command, it reads the input JSONL file, assigns
each structure a stable case ID, checks the central status cache in the
superdirectory, and decides which structures still need work. It skips cases
that are already complete, usually skips cases that are still running, and can
optionally retry failed ones. For every structure that should be processed, it
writes a small case record and prepares a SLURM submission script that will run
a single VASP calculation in that structure's subdirectory.

If main() receives the refresh-status command, it scans the superdirectory and
updates the central metadata file that records whether each case is complete,
running, partial, failed, or still unattempted. This is useful after crashes,
job interruptions, or long cluster runs, because it lets the script avoid
wasting time on repeated filesystem checks the next time jobs are submitted.

If main() receives the run-case command, it runs one actual VASP calculation
for one structure. It first checks whether that case is already complete. If
so, it stops immediately. If the case looks partial or broken, it archives the
old calculation directory so that the new run starts cleanly. It then builds
the POSCAR, INCAR, and KPOINTS objects through JARVIS, launches the TBmBJ VASP
job, and writes a run summary in the case directory.

If main() receives the collect command, it loops over all structures again and
builds the final dataset summary. For completed cases it parses vasprun.xml,
extracts quantities such as the direct and indirect band gaps, Fermi level, and
density of states, writes the DOS to a compressed file, and records the paths
to important raw outputs such as WAVECAR, CHGCAR, OUTCAR, and CONTCAR. For
incomplete cases it still records their status, so the output JSONL can act as
a full ledger of the dataset rather than only a list of successes.

In short, main() acts as the dispatcher for the whole workflow. It does not do
the physics calculations itself. Its job is to decide whether the script should
prepare jobs, update progress tracking, run one case, or gather finished
results into a structured dataset.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from jarvis.core.atoms import Atoms
from jarvis.core.kpoints import Kpoints3D as Kpoints
from jarvis.io.vasp.inputs import Incar, Poscar
from jarvis.io.vasp.outputs import Vasprun
from jarvis.tasks.queue_jobs import Queue
from jarvis.tasks.vasp.vasp import VaspJob


STATUS_SCHEMA_VERSION = 1


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sanitize_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "case"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def source_uid(rec: dict) -> str:
    for key in ("mat_id", "id", "jid", "material_id", "name", "formula"):
        val = rec.get(key)
        if val not in (None, ""):
            return str(val)
    return "unknown_id"


def get_case_id(rec: dict, idx: int) -> str:
    return f"{idx:06d}_{sanitize_name(source_uid(rec))}"


def record_to_atoms(rec: dict) -> Atoms:
    atoms_dict = rec.get("atoms")
    if atoms_dict is None:
        raise ValueError("Each JSONL record must contain an 'atoms' dictionary.")
    return Atoms.from_dict(atoms_dict)


def status_dir(superdir: Path) -> Path:
    d = superdir / "_status"
    d.mkdir(parents=True, exist_ok=True)
    return d


def status_cache_path(superdir: Path) -> Path:
    return status_dir(superdir) / "calc_status.json"


def records_dir(superdir: Path) -> Path:
    d = superdir / "_records"
    d.mkdir(parents=True, exist_ok=True)
    return d


def submit_dir(superdir: Path) -> Path:
    d = superdir / "_submit"
    d.mkdir(parents=True, exist_ok=True)
    return d


def case_dir(superdir: Path, case_id: str) -> Path:
    return superdir / case_id


def calc_dir(superdir: Path, case_id: str) -> Path:
    return case_dir(superdir, case_id) / f"MAIN-MBJ-{case_id}"


def expected_case_paths(superdir: Path, case_id: str) -> dict[str, Path]:
    cdir = case_dir(superdir, case_id)
    ddir = calc_dir(superdir, case_id)
    return {
        "case_dir": cdir,
        "case_json": cdir / "case.json",
        "run_summary": cdir / "run_summary.json",
        "calc_dir": ddir,
        "vasprun_xml": ddir / "vasprun.xml",
        "outcar": ddir / "OUTCAR",
        "wavecar": ddir / "WAVECAR",
        "chgcar": ddir / "CHGCAR",
        "contcar": ddir / "CONTCAR",
        "std_err": cdir / "std_err.txt",
        "vasp_out": cdir / "vasp.out",
        "dos_npz": cdir / "tbmbj_dos.npz",
        "parsed_vasprun_json_gz": cdir / "vasprun_parsed.json.gz",
    }


def maybe_abs(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.resolve()) if path.exists() else None


def maybe_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_status_cache(superdir: Path) -> dict:
    path = status_cache_path(superdir)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "generated_at": None,
        "superdir": str(superdir),
        "cases": {},
    }


def save_status_cache(superdir: Path, cache: dict) -> None:
    cache["schema_version"] = STATUS_SCHEMA_VERSION
    cache["generated_at"] = now_ts()
    cache["superdir"] = str(superdir)
    path = status_cache_path(superdir)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(path)


def base_tbmbj_incar(encut: float, nedos: int) -> dict:
    return {
        "PREC": "Accurate",
        "ENCUT": float(encut),
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "EDIFF": "1E-7",
        "NELM": 500,
        "NEDOS": int(nedos),
        "LORBIT": 11,
        "ISPIN": 2,
        "METAGGA": "MBJ",
        "ISYM": 0,
        "NSW": 0,
        "IBRION": -1,
        "LCHARG": ".TRUE.",
        "LWAVE": ".TRUE.",
    }


def write_case_record(case_dir_path: Path, case_id: str, idx: int, rec: dict) -> Path:
    case_json = case_dir_path / "case.json"
    payload = {
        "case_id": case_id,
        "record_index": idx,
        "record": rec,
    }
    case_json.write_text(json.dumps(payload, indent=2))
    return case_json


def archive_incomplete_calc_dir(ddir: Path) -> Path | None:
    if not ddir.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archived = ddir.parent / f"{ddir.name}.incomplete_{stamp}"
    shutil.move(str(ddir), str(archived))
    return archived


def classify_case_from_fs(
    superdir: Path,
    case_id: str,
    verify_convergence: bool = False,
) -> dict:
    paths = expected_case_paths(superdir, case_id)

    if not paths["case_dir"].exists():
        return {
            "status": "unattempted",
            "reason": "case_directory_missing",
            "case_id": case_id,
            "checked_at": now_ts(),
            "paths": {k: str(v) for k, v in paths.items()},
            "file_flags": {k: False for k in paths},
            "converged": None,
            "converged_electronic": None,
            "converged_ionic": None,
        }

    summary_status = None
    summary_error = None
    if paths["run_summary"].exists():
        try:
            summary = json.loads(paths["run_summary"].read_text())
            summary_status = summary.get("status")
            summary_error = summary.get("error")
        except Exception as exc:
            summary_status = "corrupt_summary"
            summary_error = repr(exc)

    flags = {k: v.exists() for k, v in paths.items()}
    required_complete = ["vasprun_xml", "outcar", "wavecar", "chgcar", "contcar"]
    all_required = all(flags[k] for k in required_complete)
    any_output = any(
        flags[k]
        for k in ["vasprun_xml", "outcar", "wavecar", "chgcar", "contcar", "std_err", "vasp_out"]
    )

    converged = None
    converged_electronic = None
    converged_ionic = None
    parse_error = None

    if all_required and verify_convergence:
        try:
            vrun = Vasprun(str(paths["vasprun_xml"]))
            converged = bool(vrun.converged)
            converged_electronic = bool(vrun.converged_electronic)
            converged_ionic = bool(vrun.converged_ionic)
        except Exception as exc:
            parse_error = repr(exc)

    if all_required:
        if verify_convergence:
            if parse_error is not None:
                status = "partial"
                reason = "required_files_exist_but_vasprun_parse_failed"
            elif converged_electronic:
                status = "complete"
                reason = "required_files_exist_and_electronically_converged"
            else:
                status = "partial"
                reason = "required_files_exist_but_not_electronically_converged"
        else:
            if summary_status == "failed":
                status = "failed"
                reason = "summary_marked_failed_even_though_files_exist"
            else:
                status = "complete"
                reason = "required_files_exist"
    else:
        if summary_status in {"started", "running"}:
            status = "running"
            reason = "summary_marked_started"
        elif summary_status == "failed":
            status = "failed"
            reason = "summary_marked_failed"
        elif any_output or flags["run_summary"] or flags["case_json"]:
            status = "partial"
            reason = "case_directory_exists_but_required_outputs_missing"
        else:
            status = "unattempted"
            reason = "empty_case_directory"

    return {
        "status": status,
        "reason": reason,
        "case_id": case_id,
        "checked_at": now_ts(),
        "summary_status": summary_status,
        "summary_error": summary_error,
        "paths": {k: str(v) for k, v in paths.items()},
        "file_flags": flags,
        "file_sizes_bytes": {k: maybe_size(v) for k, v in paths.items()},
        "converged": converged,
        "converged_electronic": converged_electronic,
        "converged_ionic": converged_ionic,
        "vasprun_parse_error": parse_error,
    }


def build_or_refresh_status_cache(
    jsonl_path: Path,
    superdir: Path,
    verify_convergence: bool = False,
    trust_cached_complete: bool = False,
) -> dict:
    cache = load_status_cache(superdir)
    cases = cache.get("cases", {})
    new_cases = {}

    for idx, rec in enumerate(iter_jsonl(jsonl_path)):
        cid = get_case_id(rec, idx)
        old = cases.get(cid)

        if (
            trust_cached_complete
            and old is not None
            and old.get("status") == "complete"
            and case_dir(superdir, cid).exists()
        ):
            entry = old
            entry["checked_at"] = now_ts()
            entry["reason"] = "trusted_cached_complete"
        else:
            entry = classify_case_from_fs(
                superdir=superdir,
                case_id=cid,
                verify_convergence=verify_convergence,
            )

        entry["record_index"] = idx
        entry["source_uid"] = source_uid(rec)
        entry["formula"] = rec.get("formula")
        entry["elements"] = rec.get("elements")
        entry["mat_id"] = rec.get("mat_id")
        entry["id"] = rec.get("id")
        new_cases[cid] = entry

    cache["cases"] = new_cases
    save_status_cache(superdir, cache)
    return cache


def submit_cases(args: argparse.Namespace) -> None:
    superdir = Path(args.superdir).resolve()
    superdir.mkdir(parents=True, exist_ok=True)

    jsonl_path = Path(args.jsonl).resolve()
    records_directory = records_dir(superdir)
    submit_directory = submit_dir(superdir)

    cache = build_or_refresh_status_cache(
        jsonl_path=jsonl_path,
        superdir=superdir,
        verify_convergence=False,
        trust_cached_complete=(not args.refresh_status),
    )

    script_path = Path(__file__).resolve()
    python_exec = shlex.quote(args.python_exec or sys.executable)

    prepared = 0
    skipped_complete = 0
    skipped_running = 0
    skipped_failed = 0

    for idx, rec in enumerate(iter_jsonl(jsonl_path)):
        cid = get_case_id(rec, idx)
        status = cache["cases"].get(cid, {}).get("status", "unattempted")

        if status == "complete":
            skipped_complete += 1
            continue
        if status == "running" and not args.retry_running:
            skipped_running += 1
            continue
        if status == "failed" and args.skip_failed:
            skipped_failed += 1
            continue

        cdir = case_dir(superdir, cid)
        cdir.mkdir(parents=True, exist_ok=True)

        case_json = write_case_record(cdir, cid, idx, rec)
        record_copy = records_directory / f"{cid}.json"
        record_copy.write_text(case_json.read_text())

        job_line = " ".join(
            [
                python_exec,
                shlex.quote(str(script_path)),
                "run-case",
                "--case-json",
                shlex.quote(str(record_copy)),
                "--superdir",
                shlex.quote(str(superdir)),
                "--vasp-cmd",
                shlex.quote(args.vasp_cmd),
                "--encut",
                str(args.encut),
                "--kpleng",
                str(args.kpleng),
                "--nedos",
                str(args.nedos),
                "--pot-type",
                shlex.quote(args.pot_type),
                "--attempts",
                str(args.attempts),
            ]
        )

        submit_script = submit_directory / f"{cid}.slurm"

        Queue.slurm(
            filename=str(submit_script),
            nnodes=args.nnodes,
            cores=args.cores,
            walltime=args.walltime,
            queue=args.queue,
            account=args.account,
            memory=args.memory,
            jobname=cid,
            jobout=str(cdir / "slurm.out"),
            joberr=str(cdir / "slurm.err"),
            pre_job_lines=args.pre_job_lines,
            directory=str(superdir),
            job_line=job_line,
            submit_cmd=["sbatch", str(submit_script)] if args.submit else None,
        )

        prepared += 1
        print(f"Prepared {cid}  [prior status: {status}]")

    print("\nSummary")
    print(f"  prepared/submitted : {prepared}")
    print(f"  skipped complete   : {skipped_complete}")
    print(f"  skipped running    : {skipped_running}")
    print(f"  skipped failed     : {skipped_failed}")
    print(f"  status cache       : {status_cache_path(superdir)}")


def run_case(args: argparse.Namespace) -> None:
    case_json = Path(args.case_json).resolve()
    payload = json.loads(case_json.read_text())

    cid = payload["case_id"]
    rec = payload["record"]
    superdir = Path(args.superdir).resolve()

    cdir = case_dir(superdir, cid)
    ddir = calc_dir(superdir, cid)
    cdir.mkdir(parents=True, exist_ok=True)

    pre = classify_case_from_fs(
        superdir=superdir,
        case_id=cid,
        verify_convergence=False,
    )

    if pre["status"] == "complete":
        summary = {
            "case_id": cid,
            "record_index": payload["record_index"],
            "status": "skipped_complete",
            "started_at": now_ts(),
            "finished_at": now_ts(),
            "input_metadata": {k: v for k, v in rec.items() if k != "atoms"},
            "precheck": pre,
        }
        (cdir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{cid}: already complete, skipping.")
        return

    archived_old_calc = None
    if pre["status"] in {"partial", "failed"} and ddir.exists():
        archived_old_calc = archive_incomplete_calc_dir(ddir)

    atoms = record_to_atoms(rec)
    poscar = Poscar(atoms)
    poscar.comment = cid

    incar = Incar(base_tbmbj_incar(encut=args.encut, nedos=args.nedos))
    kpoints = Kpoints().automatic_length_mesh(
        lattice_mat=atoms.lattice_mat,
        length=args.kpleng,
    )

    summary = {
        "case_id": cid,
        "record_index": payload["record_index"],
        "status": "started",
        "started_at": now_ts(),
        "input_metadata": {k: v for k, v in rec.items() if k != "atoms"},
        "settings": {
            "encut": args.encut,
            "kpleng": args.kpleng,
            "nedos": args.nedos,
            "pot_type": args.pot_type,
            "attempts": args.attempts,
        },
        "precheck": pre,
        "archived_old_calc_dir": str(archived_old_calc) if archived_old_calc else None,
        "paths": {
            "case_dir": str(cdir),
            "calc_dir": str(ddir),
        },
    }
    (cdir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    cwd = Path.cwd()
    os.chdir(cdir)
    try:
        try:
            job = VaspJob(
                poscar=poscar,
                incar=incar,
                kpoints=kpoints,
                vasp_cmd=args.vasp_cmd,
                output_file="vasp.out",
                stderr_file="std_err.txt",
                copy_files=[],
                jobname=f"MAIN-MBJ-{cid}",
                pot_type=args.pot_type,
                attempts=args.attempts,
            )
            energy, contcar = job.runjob()

            summary.update(
                {
                    "status": "finished",
                    "finished_at": now_ts(),
                    "final_energy_returned": energy,
                    "contcar_returned": contcar,
                    "paths": {
                        "case_dir": str(cdir),
                        "calc_dir": str(ddir),
                        "vasprun_xml": str(ddir / "vasprun.xml"),
                        "outcar": str(ddir / "OUTCAR"),
                        "wavecar": str(ddir / "WAVECAR"),
                        "chgcar": str(ddir / "CHGCAR"),
                        "contcar": str(ddir / "CONTCAR"),
                    },
                }
            )
        except Exception as exc:
            summary.update(
                {
                    "status": "failed",
                    "finished_at": now_ts(),
                    "error": repr(exc),
                }
            )
            raise
        finally:
            (cdir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        os.chdir(cwd)


def refresh_status(args: argparse.Namespace) -> None:
    superdir = Path(args.superdir).resolve()
    jsonl_path = Path(args.jsonl).resolve()

    cache = build_or_refresh_status_cache(
        jsonl_path=jsonl_path,
        superdir=superdir,
        verify_convergence=args.verify_convergence,
        trust_cached_complete=False,
    )

    counts = {}
    for entry in cache["cases"].values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1

    print(f"Wrote status cache to: {status_cache_path(superdir)}")
    print("Counts by status:")
    for k in ["complete", "running", "partial", "failed", "unattempted"]:
        print(f"  {k:11s} : {counts.get(k, 0)}")


def collect_cases(args: argparse.Namespace) -> None:
    superdir = Path(args.superdir).resolve()
    jsonl_path = Path(args.jsonl).resolve()
    out_jsonl = Path(args.out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    cache = build_or_refresh_status_cache(
        jsonl_path=jsonl_path,
        superdir=superdir,
        verify_convergence=args.verify_convergence,
        trust_cached_complete=False,
    )

    rows = []
    for idx, rec in enumerate(iter_jsonl(jsonl_path)):
        cid = get_case_id(rec, idx)
        centry = cache["cases"].get(cid, {})
        status = centry.get("status", "unattempted")

        row = {
            "case_id": cid,
            "record_index": idx,
            "source_uid": source_uid(rec),
            "status": status,
            "formula": rec.get("formula"),
            "elements": rec.get("elements"),
            "input_metadata": {k: v for k, v in rec.items() if k != "atoms"},
        }

        if status != "complete":
            row["status_reason"] = centry.get("reason")
            row["paths"] = centry.get("paths")
            row["converged"] = centry.get("converged")
            row["converged_electronic"] = centry.get("converged_electronic")
            row["converged_ionic"] = centry.get("converged_ionic")
            rows.append(row)
            continue

        paths = expected_case_paths(superdir, cid)
        vrun = Vasprun(str(paths["vasprun_xml"]))

        indir_gap, cbm, vbm = vrun.get_indir_gap
        dir_gap = vrun.get_dir_gap
        energies, spin_up, spin_dn = vrun.total_dos

        dos_npz = paths["dos_npz"]
        np.savez_compressed(
            dos_npz,
            energies=np.asarray(energies),
            spin_up=np.asarray(spin_up),
            spin_dn=np.asarray(spin_dn),
        )

        parsed_json_gz = None
        if args.dump_full_vasprun_json:
            parsed_json_gz = paths["parsed_vasprun_json_gz"]
            with gzip.open(parsed_json_gz, "wt") as f:
                json.dump(vrun.to_dict(), f)

        final_atoms = None
        try:
            final_atoms = vrun.all_structures[-1].to_dict()
        except Exception:
            pass

        row.update(
            {
                "status": "complete",
                "converged": bool(vrun.converged),
                "converged_electronic": bool(vrun.converged_electronic),
                "converged_ionic": bool(vrun.converged_ionic),
                "final_energy": float(vrun.final_energy),
                "fermi_ev": float(vrun.efermi),
                "indir_gap_ev": float(indir_gap),
                "direct_gap_ev": float(dir_gap),
                "cbm_ev": float(cbm),
                "vbm_ev": float(vbm),
                "is_spin_polarized": bool(vrun.is_spin_polarized),
                "num_atoms": int(vrun.num_atoms),
                "vrun_elements": list(vrun.elements),
                "initial_atoms": rec.get("atoms"),
                "final_atoms": final_atoms,
                "all_input_parameters": vrun.all_input_parameters,
                "paths": {
                    "case_dir": maybe_abs(paths["case_dir"]),
                    "calc_dir": maybe_abs(paths["calc_dir"]),
                    "dos_npz": maybe_abs(dos_npz),
                    "vasprun_xml": maybe_abs(paths["vasprun_xml"]),
                    "outcar": maybe_abs(paths["outcar"]),
                    "wavecar": maybe_abs(paths["wavecar"]),
                    "chgcar": maybe_abs(paths["chgcar"]),
                    "contcar": maybe_abs(paths["contcar"]),
                    "vasprun_parsed_json_gz": maybe_abs(parsed_json_gz) if parsed_json_gz else None,
                },
                "file_sizes_bytes": {
                    "vasprun_xml": maybe_size(paths["vasprun_xml"]),
                    "outcar": maybe_size(paths["outcar"]),
                    "wavecar": maybe_size(paths["wavecar"]),
                    "chgcar": maybe_size(paths["chgcar"]),
                    "contcar": maybe_size(paths["contcar"]),
                    "dos_npz": maybe_size(dos_npz),
                },
                "sha256": {
                    "wavecar": file_sha256(paths["wavecar"]) if args.hash_wavecar else None,
                    "vasprun_xml": file_sha256(paths["vasprun_xml"]) if args.hash_vasprun else None,
                },
            }
        )

        rows.append(row)

    with out_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} records to {out_jsonl}")
    print(f"Status cache is at {status_cache_path(superdir)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--jsonl", required=True)
    p_submit.add_argument("--superdir", required=True)
    p_submit.add_argument("--vasp-cmd", required=True)
    p_submit.add_argument("--python-exec", default=sys.executable)
    p_submit.add_argument("--encut", type=float, default=520.0)
    p_submit.add_argument("--kpleng", type=float, default=25.0)
    p_submit.add_argument("--nedos", type=int, default=5000)
    p_submit.add_argument("--pot-type", default="POT_GGA_PAW_PBE")
    p_submit.add_argument("--attempts", type=int, default=3)
    p_submit.add_argument("--submit", action="store_true")
    p_submit.add_argument("--refresh-status", action="store_true")
    p_submit.add_argument("--retry-running", action="store_true")
    p_submit.add_argument("--skip-failed", action="store_true")
    p_submit.add_argument("--nnodes", type=int, default=1)
    p_submit.add_argument("--cores", type=int, default=32)
    p_submit.add_argument("--walltime", default="24:00:00")
    p_submit.add_argument("--queue", default=None)
    p_submit.add_argument("--account", default=None)
    p_submit.add_argument("--memory", default=None)
    p_submit.add_argument(
        "--pre-job-lines",
        default=None,
        help="Multiline shell text inserted before the job command.",
    )

    p_refresh = sub.add_parser("refresh-status")
    p_refresh.add_argument("--jsonl", required=True)
    p_refresh.add_argument("--superdir", required=True)
    p_refresh.add_argument("--verify-convergence", action="store_true")

    p_run = sub.add_parser("run-case")
    p_run.add_argument("--case-json", required=True)
    p_run.add_argument("--superdir", required=True)
    p_run.add_argument("--vasp-cmd", required=True)
    p_run.add_argument("--encut", type=float, default=520.0)
    p_run.add_argument("--kpleng", type=float, default=25.0)
    p_run.add_argument("--nedos", type=int, default=5000)
    p_run.add_argument("--pot-type", default="POT_GGA_PAW_PBE")
    p_run.add_argument("--attempts", type=int, default=3)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--jsonl", required=True)
    p_collect.add_argument("--superdir", required=True)
    p_collect.add_argument("--out-jsonl", required=True)
    p_collect.add_argument("--verify-convergence", action="store_true")
    p_collect.add_argument("--dump-full-vasprun-json", action="store_true")
    p_collect.add_argument("--hash-wavecar", action="store_true")
    p_collect.add_argument("--hash-vasprun", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "submit":
        submit_cases(args)
    elif args.cmd == "refresh-status":
        refresh_status(args)
    elif args.cmd == "run-case":
        run_case(args)
    elif args.cmd == "collect":
        collect_cases(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
