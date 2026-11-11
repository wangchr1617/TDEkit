"""Compare an exported ZBL change with GPUMD on fixed two-atom structures."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from ase import Atoms

from tdekit.core import run_gpumd
from tdekit.zbl import read_flexible_zbl, write_candidate

ROOT = Path(__file__).resolve().parents[1]


def single_point(atoms, directory, model, executable):
    commands = [
        "potential nep.txt", "ensemble nve", "time_step 0",
        "dump_force 1", "dump_thermo 1", "run 1",
    ]
    run_gpumd(atoms, directory, commands, nep_path=model, executable=executable)
    if "Finished running GPUMD." not in (directory / "gpumd.out").read_text():
        raise ValueError(f"Missing GPUMD normal termination marker in {directory}")
    force = np.loadtxt(directory / "force.out", ndmin=2)
    thermo = np.loadtxt(directory / "thermo.out", ndmin=2)
    if (force.shape != (2, 3) or thermo.shape != (1, 18)
            or not np.isfinite(force).all() or not np.isfinite(thermo).all()):
        raise ValueError(f"Incomplete or nonfinite single point in {directory}")
    return thermo[0, 2], force


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", default="gpumd")
    parser.add_argument("--model", type=Path, default=ROOT / "test/potentials/0.txt")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    parameters = read_flexible_zbl(args.model)
    pair = ("Te", "Ge")
    baseline = parameters[pair]
    candidate = replace(baseline, amplitudes=(0.2, 0.5, 0.3, 0.04))
    parameters[pair] = candidate
    exported = args.output / "nep_candidate.txt"
    write_candidate(args.model, exported, parameters)
    for index, distance in enumerate((0.8, 1.5, 2.2)):
        atoms = Atoms(pair, positions=[[10, 10, 10], [10 + distance, 10, 10]],
                      cell=[24] * 3, pbc=True)
        atoms.info["velocities"] = np.zeros((2, 3))
        energy_a, force_a = single_point(atoms, args.output / f"base_{index}",
                                         args.model, args.executable)
        energy_b, force_b = single_point(atoms, args.output / f"candidate_{index}",
                                         exported, args.executable)
        expected_energy, expected_force = (
            np.asarray(candidate.evaluate(pair, distance))
            - np.asarray(baseline.evaluate(pair, distance))
        )
        np.testing.assert_allclose(energy_b - energy_a, expected_energy, atol=3e-3)
        np.testing.assert_allclose(force_b - force_a,
                                   [[-expected_force, 0, 0], [expected_force, 0, 0]], atol=3e-3)
        print(f"{distance:g} Å: delta energy {energy_b - energy_a:.8f} eV, "
              f"delta radial force {force_b[1, 0] - force_a[1, 0]:.8f} eV/Å")


if __name__ == "__main__":
    main()
