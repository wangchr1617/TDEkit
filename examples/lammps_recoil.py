"""Run two small LAMMPS recoil cases and verify atom IDs and velocities."""

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.units import fs

from tdekit.core import set_pka
from tdekit.lammps import run_lammps
from tdekit.recoil import LAMMPSBackend, RecoilEvent, run_recoil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", default="lmp")
    parser.add_argument("--unified", action="store_true")
    args = parser.parse_args()

    for name, energy in (("control", 0.0), ("recoil", 1.0)):
        atoms = bulk("Ar", "fcc", a=5.3, cubic=True).repeat((2, 2, 2))
        atoms.info["velocities"] = np.zeros((len(atoms), 3))
        case = args.output / name
        commands = [
            "pair_style lj/cut 4.5",
            "pair_coeff * * 0.0104 3.4",
            "timestep 0.0001",
            "fix integrate all nve",
            "dump atoms all custom 4 trajectory.dump id type x y z vx vy vz",
            "dump_modify atoms sort id format float %.16g"
            + (" time yes units yes" if args.unified else ""),
            "thermo 4",
            "run 4",
        ]
        if args.unified:
            result = run_recoil(atoms, RecoilEvent(1, energy, [1, 0, 0]),
                                LAMMPSBackend(commands, args.executable), case,
                                sample_times_fs=[0, 0.4])
            observed = result.frames[0].velocities_angstrom_fs[0]
            actual_energy = 0.5 * atoms[0].mass * np.sum((observed / fs)**2)
            np.testing.assert_allclose(actual_energy, energy, atol=1e-10)
            print(f"{name}: unified LAMMPS accepted {len(result.frames)} frames, {actual_energy:g} eV")
            continue
        set_pka(atoms, energy, [1, 0, 0], index=0, mode="impulse")
        run_lammps(atoms, case, commands, executable=args.executable)

        frames = (case / "trajectory.dump").read_text().split("ITEM: TIMESTEP\n")[1:]
        if len(frames) != 2:
            raise ValueError(f"Expected exactly two frames in {case}")
        initial, final = [np.loadtxt(frame.splitlines()[8:]) for frame in frames]
        for frame in (initial, final):
            np.testing.assert_array_equal(frame[:, 0], np.arange(1, len(atoms) + 1))
            if not np.isfinite(frame).all():
                raise ValueError(f"Nonfinite LAMMPS output in {case}")
        np.testing.assert_allclose(initial[:, 5:], atoms.info["velocities"] * 1000,
                                   rtol=1e-12, atol=1e-12)
        energy_observed = 0.5 * atoms[0].mass * np.sum((initial[0, 5:] / (1000 * fs))**2)
        np.testing.assert_allclose(energy_observed, energy, atol=1e-12)
        print(f"{name}: verified {len(atoms)} atom IDs, two frames, initial PKA {energy_observed:g} eV")


if __name__ == "__main__":
    main()
