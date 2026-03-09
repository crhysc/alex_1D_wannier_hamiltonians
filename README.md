# alex_1D_wannier_hamiltonians_tbmbj


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
