"""Run and verify two small GPUMD recoil cases and their defect summaries."""

import argparse
import json
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read

from tdekit import TDESearch
from tdekit.core import dump_xyz, read_restart, read_xyz
from tdekit.recoil import GPUMDBackend, RecoilEvent, run_recoil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", default="gpumd")
    parser.add_argument("--model", type=Path, default=ROOT / "test/xyzs/model.xyz")
    parser.add_argument("--potential", type=Path, default=ROOT / "test/potentials/0.txt")
    parser.add_argument("--ovito-command", nargs="+")
    parser.add_argument("--check-search", action="store_true", help="Also check bounded TDE search")
    parser.add_argument("--unified", action="store_true", help="Run the unified trajectory interface")
    args = parser.parse_args()

    atoms = read(args.model, format="extxyz").repeat((2, 2, 2))
    atoms.info["velocities"] = np.zeros((len(atoms), 3))
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "initial.xyz"
    with source.open("w", encoding="utf-8") as stream:
        dump_xyz(stream, atoms)
    commands = [
        "potential nep.txt", "ensemble nve", "time_step 0",
        "dump_exyz 1 1 0", "dump_thermo 1", "run 1",
        "time_step 0.1", "dump_exyz 2 1 0", "dump_thermo 2",
        "dump_restart 2", "run 2",
    ]
    for name, energy in (("control", 0.0), ("recoil", 1.0)):
        if args.unified:
            case = args.output / name
            backend = GPUMDBackend(commands, args.potential, args.executable)
            result = run_recoil(atoms, RecoilEvent(1, energy, [1, 0, 0]), backend,
                                case, sample_times_fs=[0, 0.2])
            initial_velocity = result.frames[0].velocities_angstrom_fs[0]
            actual_energy = 0.5 * atoms[0].mass * np.sum((initial_velocity / units.fs)**2)
            np.testing.assert_allclose(actual_energy, energy, atol=1e-6)
            print(f"{name}: unified GPUMD accepted {len(result.frames)} frames, {actual_energy:g} eV")
            continue
        search = TDESearch(args.output / name, direction=[1, 0, 0],
                           ovito_command=args.ovito_command)
        count, case_id = search.run_cascade_simulation(
            energy, clean=False, input_file=source, index=0, mode="impulse",
            run_in=commands, nep_path=args.potential, executable=args.executable,
        )
        case = search.work_dir / case_id
        if "Finished running GPUMD." not in (case / "gpumd.out").read_text():
            raise ValueError(f"Missing GPUMD normal termination marker in {case}")
        frames = read_xyz(case / "dump.xyz")
        if len(frames) != 2:
            raise ValueError(f"Expected a zero-time and a final frame in {case}")
        final = read_restart(case / "restart.xyz")
        for frame in [*frames, final]:
            np.testing.assert_array_equal(frame.numbers, atoms.numbers)
            if not (np.isfinite(frame.positions).all() and
                    np.isfinite(frame.info["velocities"]).all()):
                raise ValueError(f"Invalid trajectory or restart in {case}")
        np.testing.assert_allclose(frames[0].positions, atoms.positions, atol=2e-6)
        initial = np.asarray(frames[0].info["velocities"])
        expected = np.zeros_like(initial)
        expected[0, 0] = units.fs * np.sqrt(2 * energy / atoms[0].mass)
        np.testing.assert_allclose(initial, expected, atol=2e-7)
        thermo = np.loadtxt(case / "thermo.out", ndmin=2)
        if len(thermo) != 2 or not np.isfinite(thermo).all():
            raise ValueError(f"Missing or nonfinite thermo data in {case}")
        if count != 0:
            raise ValueError(f"Unexpected defect count after 0.2 fs in {case}: {count}")
        receipt = json.loads((search.work_dir / f"{case_id}.json").read_text())
        if receipt["frenkel_pairs"] != count or receipt["energy_ev"] != energy:
            raise ValueError(f"Invalid case receipt: {case_id}")
        print(f"{name}: {len(atoms)} atoms, 2 frames, {energy:g} eV initial recoil, {count} Frenkel pairs")
    if args.check_search:
        check_bounded_search(args, atoms, commands)


def check_bounded_search(args, atoms, commands):
    sources = []
    for seed in range(4):
        path = args.output / f"sample_{seed}.xyz"
        atoms.info["velocities"] = np.random.default_rng(seed).normal(0, 0.0001, (len(atoms), 3))
        with path.open("w", encoding="utf-8") as stream:
            dump_xyz(stream, atoms)
        sources.append(path)
    search = TDESearch(args.output / "search", ovito_command=args.ovito_command)
    try:
        search.find_tde(
            init_energy=0.5, max_energy=0.75, num_samples=2, verification_samples=2,
            sample_inputs=sources[:2], verification_inputs=sources[2:],
            index=0, mode="impulse", clean=False, run_in=commands,
            nep_path=args.potential, executable=args.executable,
        )
    except RuntimeError as error:
        if str(error) != "No observed threshold crossing through 0.75 eV":
            raise
    else:
        raise ValueError("Expected a bounded no-crossing result")
    receipts = [json.loads(path.read_text()) for path in search.work_dir.glob("cascade_id*.json")]
    expected = {(energy, str(path.resolve())) for energy in (0.5, 0.75) for path in sources[:2]}
    observed = {(row["energy_ev"], row["input_file"]) for row in receipts}
    if len(receipts) != len(expected) or observed != expected:
        raise ValueError("Unexpected energies or initial states in the search receipts")
    if any(row["frenkel_pairs"] != 0 for row in receipts):
        raise ValueError("Unexpected defects in the short search cases")
    print("Search: both explicit initial states evaluated at 0.5 and 0.75 eV, no crossing")


if __name__ == "__main__":
    main()
