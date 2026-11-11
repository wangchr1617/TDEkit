"""Verify CP2K Quickstep NVE with zero-energy and 1 eV recoil inputs."""

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read

from tdekit.core import set_pka
from tdekit.cp2k import run_cp2k
from tdekit.recoil import CP2KBackend, RecoilEvent, run_recoil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", default="cp2k.psmp")
    parser.add_argument("--unified", action="store_true")
    args = parser.parse_args()
    template = Path(__file__).with_suffix(".inp").read_text(encoding="utf-8")

    for name, energy in (("control", 0.0), ("recoil", 1.0)):
        atoms = Atoms("OH2", positions=[[5, 5, 5], [5.757, 5.586, 5],
                                      [4.243, 5.586, 5]], cell=[10] * 3)
        atoms.info["velocities"] = np.zeros((len(atoms), 3))
        case = args.output / name
        if args.unified:
            result = run_recoil(atoms, RecoilEvent(2, energy, [1, 0, 0]),
                                CP2KBackend(template, executable=args.executable), case,
                                sample_times_fs=[0, 0.1, 0.2])
            expected = result.frames[0].velocities_angstrom_fs
        else:
            set_pka(atoms, energy, [1, 0, 0], index=1, mode="impulse")
            run_cp2k(atoms, case, template, executable=args.executable)
            expected = atoms.info["velocities"]

        log = (case / "cp2k.out").read_text()
        if log.count("SCF run converged") != 3:
            raise ValueError(f"Expected three converged SCF calculations in {case}")
        table = np.loadtxt(case / "recoil-1.ener")
        np.testing.assert_array_equal(table[:, 0], [0, 1, 2])
        if not np.isfinite(table).all():
            raise ValueError(f"Nonfinite energy in {case}")
        np.testing.assert_allclose(table[0, 2] * units.Hartree, energy, atol=1e-6)

        positions = read(case / "recoil-pos-1.xyz", index=":", format="xyz")
        velocities = read(case / "recoil-vel-1.xyz", index=":", format="xyz")
        for frames in (positions, velocities):
            if len(frames) != 3 or any(
                frame.get_chemical_symbols() != atoms.get_chemical_symbols() or not np.isfinite(frame.positions).all()
                for frame in frames
            ):
                raise ValueError(f"Incomplete or invalid CP2K trajectory in {case}")
        initial_velocities = velocities[0].positions * units.Bohr * units.fs / units.AUT
        np.testing.assert_allclose(initial_velocities, expected, atol=1e-8)
        print(f"{name}: three converged SCF steps, initial kinetic energy {table[0, 2] * units.Hartree:.6f} eV")


if __name__ == "__main__":
    main()
