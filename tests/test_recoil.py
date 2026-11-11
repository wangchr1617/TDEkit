import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms, units

from tdekit._recoil_io import read_cp2k_frames, read_lammps_frames
from tdekit.recoil import RecoilEvent, RecoilFrame, run_recoil


class TestBackend:
    name = "test"
    identity = "output_order"

    def __init__(self, fault=None):
        self.fault = fault

    def run(self, atoms, directory):
        directory.mkdir()
        frames = []
        for t in (0.0, 0.1):
            frame = atoms.copy()
            velocities = frame.info["velocities"].copy()
            ids = np.array([1, 2])
            if self.fault == "velocity":
                velocities[0] = 0
            if self.fault == "ids" and t > 0:
                ids[1] = 1
            if self.fault == "species" and t > 0:
                frame.numbers[:] = frame.numbers[::-1]
            if self.fault == "nan" and t > 0:
                frame.positions[1, 0] = np.nan
            if self.fault == "clock" and t > 0:
                t += 0.01
            frames.append(RecoilFrame(t, frame, ids, velocities))
        return frames[:1] if self.fault == "truncated" else frames


class RecoilTests(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms("GeTe", positions=[[1, 1, 1], [3, 3, 3]], cell=[8] * 3, pbc=True)
        self.atoms.info["velocities"] = np.array([[0.001, 0, 0], [0, 0.002, 0]])

    def test_impulse_energy_receipt_and_input_immutability(self):
        initial = self.atoms.info["velocities"].copy()
        with tempfile.TemporaryDirectory() as folder:
            case = Path(folder) / "run"
            result = run_recoil(self.atoms, RecoilEvent(1, 1.0, [2, 0, 0]), TestBackend(),
                                case, sample_times_fs=[0, 0.1])
            np.testing.assert_array_equal(self.atoms.info["velocities"], initial)
            expected = initial.copy()
            expected[0, 0] += units.fs * np.sqrt(2 / self.atoms[0].mass)
            np.testing.assert_allclose(result.frames[0].velocities_angstrom_fs, expected)
            self.assertAlmostEqual(result.velocity_increment_energy_ev, 1.0)
            self.assertGreater(result.kinetic_energy_change_ev, 1.0)
            receipt = json.loads((case / "recoil.json").read_text())
            self.assertEqual(receipt["status"], "validated")
            self.assertEqual(receipt["atom_identity"], "output_order")
            self.assertEqual(receipt["direction_cartesian"], [1, 0, 0])
            with self.assertRaises(FileExistsError):
                run_recoil(self.atoms, RecoilEvent(1, 0, [1, 0, 0]), TestBackend(),
                           case, sample_times_fs=[0, 0.1])

    def test_lammps_uses_explicit_ids_and_dump_time(self):
        dump = """ITEM: UNITS
metal
ITEM: TIME
0.001
ITEM: TIMESTEP
3
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 8
0 8
0 8
ITEM: ATOMS id type x y z vx vy vz
2 2 3 3 3 0 2 0
1 1 1 1 1 1 0 0
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectory.dump"
            path.write_text(dump)
            frame, = read_lammps_frames(path, self.atoms)
            self.assertEqual(frame.time_fs, 1)
            np.testing.assert_array_equal(frame.atom_ids, [1, 2])
            np.testing.assert_allclose(frame.atoms.positions, self.atoms.positions)
            np.testing.assert_allclose(frame.velocities_angstrom_fs, self.atoms.info["velocities"])
            path.write_text(dump.replace("2 2 3 3 3", "1 2 3 3 3"))
            with self.assertRaisesRegex(ValueError, "atom IDs"):
                list(read_lammps_frames(path, self.atoms))

    def test_cp2k_rejects_inconsistent_reported_kinetic_energy(self):
        with tempfile.TemporaryDirectory() as folder:
            case = Path(folder)
            (case / "recoil-pos-1.xyz").write_text(
                "2\ni = 0, time = 0.000\nGe 1 1 1\nTe 3 3 3\n"
            )
            (case / "recoil-vel-1.xyz").write_text(
                "2\ni = 0, time = 0.000\nGe 0 0 0\nTe 0 0 0\n"
            )
            (case / "recoil-1.ener").write_text("0 0 1 0 0 1 0\n")
            with self.assertRaisesRegex(ValueError, "kinetic energy"):
                list(read_cp2k_frames(case, "recoil", self.atoms))

    def test_control_preserves_ase_momenta(self):
        del self.atoms.info["velocities"]
        self.atoms.set_velocities([[0.05, 0, 0], [0, 0.01, 0]])
        with tempfile.TemporaryDirectory() as folder:
            result = run_recoil(self.atoms, RecoilEvent(1, 0, [1, 0, 0]), TestBackend(),
                                Path(folder) / "run", sample_times_fs=[0, 0.1])
            np.testing.assert_allclose(result.frames[0].velocities_angstrom_fs,
                                       self.atoms.get_velocities() * units.fs)

    def test_invalid_trajectory_leaves_no_success_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            for fault in ("velocity", "ids", "species", "nan", "truncated", "clock"):
                with self.subTest(fault=fault):
                    case = Path(folder) / fault
                    with self.assertRaises(ValueError):
                        run_recoil(self.atoms, RecoilEvent(1, 1, [1, 0, 0]), TestBackend(fault),
                                   case, sample_times_fs=[0, 0.1])
                    self.assertTrue(case.is_dir())
                    self.assertFalse((case / "recoil.json").exists())


if __name__ == "__main__":
    unittest.main()
