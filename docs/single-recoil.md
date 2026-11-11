# Single-recoil API

Use the same event and trajectory acceptance interface with GPUMD, LAMMPS
or CP2K. Each backend reuses the existing engine runner and accepts native
engine input. The caller chooses the potential, timestep, integration
method and sampling intervals.

```python
from tdekit.core import read_restart
from tdekit.recoil import GPUMDBackend, RecoilEvent, run_recoil

atoms = read_restart("restart.xyz")
backend = GPUMDBackend(
    commands=[
        "potential nep.txt", "ensemble nve", "time_step 0",
        "dump_exyz 1 1 0", "run 1",
        "time_step 0.1", "dump_exyz 2 1 0", "run 2",
    ],
    nep_path="my_potential.txt",
)
result = run_recoil(
    atoms, RecoilEvent(atom_id=1, energy_ev=1.0, direction=[1, 0, 0]),
    backend, "runs/one_recoil", sample_times_fs=[0, 0.2],
)
for frame in result.frames:
    print(frame.time_fs, frame.atom_ids, frame.velocities_angstrom_fs)
```

The API uses 1-based IDs in ASE input order and a Cartesian direction,
normalized before applying the event. Default `impulse` assigns the
requested energy to the velocity increment. Its change in system kinetic
energy also includes the initial velocity cross term. A zero-energy
impulse preserves the initial state. The optional `replace` mode uses
the historical velocity replacement and momentum compensation. The
receipt records the increment energy and total kinetic energy change
calculated from the prepared state. Explicit `atoms.info["velocities"]`
in Å/fs take precedence over ASE momenta.

## Backend output requirements

| Backend | Output read by the unified interface |
| --- | --- |
| `GPUMDBackend` | `dump.xyz` with `Time` in fs, positions, velocities and potential energy. The validated GPUMD 5.3 example uses `dump_exyz` and a zero-time output stage. |
| `LAMMPSBackend` | `trajectory.dump` from `dump ... custom ... id type x y z vx vy vz` and `dump_modify ... time yes units yes`. Uses `metal` units and reads the reported time in ps. Atom rows are aligned by ID, including when written out of order. ASE maps a restricted triclinic dump back to the input Cartesian basis. |
| `CP2KBackend` | Quickstep fixed-cell NVE or NVT input, with the existing `tdekit-state.inc` include. Writes `<project>-pos-1.xyz`, `<project>-vel-1.xyz` and `<project>-1.ener`. The `project` argument must match the output prefix. Use the default position and velocity units, and KIND masses consistent with the ASE state. The reader synchronizes position, velocity and energy records and checks reported kinetic energy against the sampled velocities and input masses. |

CP2K XYZ omits the cell and the backend uses the input cell for each frame.
CP2K and GPUMD use species and output order to carry identity. The returned
`atom_identity` field distinguishes this from explicit LAMMPS IDs.
Same-species permutations cannot be detected in an order-only trajectory.

The interface requires normal engine termination, a finite cell and
finite positions and velocities in each frame, matching species and
periodicity, and a strictly increasing engine clock matching
`sample_times_fs`. The input-equivalent frame at zero is mandatory.
Default absolute tolerances are `1e-5 fs` for time, `2e-6 Å` for initial
positions and cell, and `2e-7 Å/fs` for initial velocities. Initial
positions are compared modulo periodic images. The result holds frames in
memory and leaves the raw outputs in the case directory. An invalid run
raises and retains its files. The `recoil.json` success receipt is only
written after acceptance.

Run the existing small engine examples with `--unified` to exercise this
interface. The original runner and TDE search entry points remain available.

```bash
python examples/lammps_recoil.py /path/to/lammps-check --unified
python examples/cp2k_recoil.py /path/to/cp2k-check --unified
python examples/gpumd_recoil.py /path/to/gpumd-check --unified
```

See the [GPUMD trajectory format](https://gpumd.org/gpumd/output_files/dump_xyz.html)
and [CP2K velocity units](https://manual.cp2k.org/cp2k-2024_2-branch/CP2K_INPUT/MOTION/PRINT/VELOCITIES.html).
