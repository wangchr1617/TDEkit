# Recoil workflows

## GPUMD

Provide a GPUMD binary on PATH or pass `executable`. With `potential nep.txt`
in `run_in`, pass the source potential as `nep_path`. TDEkit creates a fresh
case directory, stages the potential as `nep.txt`, writes `model.xyz` and
`run.in`, and saves the process output in `gpumd.out`. Use a time step and
output interval appropriate for the recoil energy. For a restart, provide
`restart.xyz` from a completed run with the same potential.

```python
from tdekit import TDESearch

search = TDESearch("runs")
count, case = search.run_cascade_simulation(
    energy=5.0,
    input_file="restart.xyz",
    index=0,
    mode="impulse",
    clean=False,
    nep_path="my_potential.txt",
    run_in=[
        "potential nep.txt",
        "ensemble nve",
        "time_step 0",
        "dump_exyz 1",
        "run 1",
        "time_step 0.1",
        "dump_exyz 10",
        "run 10",
    ],
)
```

This example uses 1 fs of dynamics and illustrates file flow. Choose a
production observation window and verify the initial and final frames for
the system of interest. The current Wigner–Seitz workflow uses the first
trajectory frame as its lattice reference. A defect count is relative to
that reference. For repeated trials, define independent thermal states and
record the direction, species, index, time step and sampling protocol.

## Initial velocities

`atoms.info["velocities"]` stores Cartesian velocities in Å/fs.
`mode="replace"` retains the legacy protocol of replacing the selected
atom's velocity and compensating the net momentum change across other atoms.
`mode="impulse"` adds a Cartesian velocity increment to one atom. Its energy
argument is the kinetic energy associated with that increment, and the
change in total kinetic energy also depends on the initial PKA velocity.
With zero energy and the default `center_pka=False`, the impulse leaves
the initial velocities and positions intact. Recoil energy is in eV.

## OVITO

Run TDEkit and native OVITO from the same Python environment where supported.
The engine-only workflow can be run with `defects_analysis=False, clean=False`.
For analysis on another machine, copy the trajectory case directory and run
the following in the destination Python environment after installing the
analysis extra. Create the analysis directory first.

```bash
python -m tdekit.defects_analyzer --work_dir results --cascade_dir path/to/case --energy 5
```

On a site that already uses a container or a separate analysis interpreter,
pass its command prefix as a list. The interpreter must be able to access
the installed analyzer script and trajectory.

```python
search = TDESearch("runs", ovito_command=["/path/to/ovito-python"])
# Or a user-provided container:
search = TDESearch("runs", ovito_command=["apptainer", "exec", "/path/to/ovito.sif", "python"])
```

For system compatibility, see the [OVITO installation guide](https://docs.ovito.org/python/introduction/getting_started/installation.html).
Keep the case files until the run and analysis are accepted. A missing or
invalid defect summary is an error.

## LAMMPS

Use `tdekit.lammps.run_lammps` with an ASE structure and the LAMMPS commands
that follow `read_data`. The adapter uses `metal` units and `atomic` atom
style. Atom IDs follow the ASE order, starting at one. Element types are
assigned alphabetically. ASE converts the cell, positions and velocities
to the LAMMPS box basis. Use the same basis for any direction-dependent
commands and when comparing vectors. Supply any potential parameters or files
in the commands and use paths visible from the case directory.

```bash
python examples/lammps_recoil.py /path/to/fresh/smoke-output --executable lmp
```

The example runs two 32-atom Ar Lennard-Jones NVE cases, four steps each,
with initial recoil energies of 0 and 1 eV. It checks atom IDs, finite
coordinates and velocities, and the actual initial recoil energy in the
dump. Verified on macOS with LAMMPS 30 Sep 2026 and ASE 3.27.0.

## CP2K

Use `tdekit.cp2k.run_cp2k` with an ASE structure carrying explicit velocities
and a CP2K MD input string. Add `@INCLUDE "tdekit-state.inc"` under
`FORCE_EVAL / SUBSYS` in place of the CELL, COORD and VELOCITY sections. The
adapter writes positions and cell vectors in Å and converts velocities to
bohr per atomic unit of time. Atoms are written in ASE order, with chemical
symbols as CP2K kind names. Set each kind's mass to the mass used by ASE to
generate the recoil. Choose compatible boundary conditions in the cell and
Poisson settings. Supply the basis, pseudopotential, integrator and output
settings in the CP2K input. Avoid restart or MD settings that reset or
rescale the supplied velocities when verifying an impulse.

```bash
python examples/cp2k_recoil.py /path/to/fresh/smoke-output --executable cp2k.psmp
```

The example uses a water molecule, Quickstep PBE, and two 0.1 fs NVE steps.
It verifies SCF convergence, trajectory and velocity frames, and the actual
initial kinetic energy for both zero-energy and 1 eV recoil. CP2K must be
able to locate `BASIS_MOLOPT` and `GTH_POTENTIALS` in its data directory.

To reproduce the two-case GPUMD check from a source checkout, run the
following with GPUMD and native OVITO available. On clusters that use a
separate analysis environment, append `--ovito-command` followed by the
interpreter command and arguments.

```bash
python examples/gpumd_recoil.py /path/to/fresh/smoke-output --executable gpumd
```

The default inputs are the repository's original GeTe model and potential.
The example checks a zero-time frame, a final frame after 0.2 fs of NVE,
a restart, finite thermo output, the prescribed initial velocity, and a
Wigner–Seitz summary. Its observation window is for software verification.

## Bounded TDE search

Use `TDESearch.find_tde` to search for a crossing of a chosen observed
Frenkel-pair probability. The search tests the exact `max_energy` if the
exponential sequence would step past it, and bisects only after observing
probabilities on both sides of the threshold. The midpoint, or the upper
bracket if needed, must pass a new verification sample. An unsuccessful
search or verification raises an exception. The returned tuple contains
the reported energy in eV and the measured verification probability.

Pass `sample_inputs` and `verification_inputs` as disjoint lists of
restart files with lengths equal to `num_samples` and
`verification_samples`. The library rejects byte-identical files within
or across the two sets. The caller must prepare statistically independent
thermal states within each set. The same set is used at each energy so
comparisons across energies are paired. The older single-`input_file`
interface remains available for fixed-state or PKA-site sampling.

```python
energy, probability = search.find_tde(
    init_energy=5.0, max_energy=40.0, precision=0.5,
    num_samples=len(search_restarts), verification_samples=len(held_out_restarts),
    sample_inputs=search_restarts, verification_inputs=held_out_restarts,
    index=0, mode="impulse", clean=False, nep_path="my_potential.txt",
    run_in=recoil_commands,
)
```

Each successfully processed case writes its own JSON receipt with the
energy, direction, input hashes and Frenkel-pair count. The search reads
the defect summary from the case directory. The work directory's legacy
summary and history are conveniences for serial use. The per-case JSON
receipts survive the legacy default cleanup. Pass `clean=False` to retain
the original engine inputs, outputs and trajectory.

Bisection is intended for protocols with an approximately monotone
damage probability. `precision` bounds the energy bracket width and
does not quantify statistical uncertainty. For a nonmonotone response,
inspect an explicit energy scan and the individual sample outcomes.
