import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from tdekit.cp2k_labels import read_cp2k_label
from tdekit.nep_dataset import DatasetFrame, audit_partitions, geometry_fingerprint, write_nep_dataset

LOG = """ GLOBAL| Run type ENERGY_FORCE
 CELL| Volume [angstrom^3]: 64.000000
 CELL| Vector a [angstrom]: 4.000 0.000 0.000 |a| = 4.000000
 CELL| Vector b [angstrom]: 0.000 4.000 0.000 |b| = 4.000000
 CELL| Vector c [angstrom]: 0.000 0.000 4.000 |c| = 4.000000
 MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM
   1  1 Ge 32 4.500000 0.500000 0.500000 4.0000 72.6400
   2  2 Te 52 1.000000 1.000000 1.000000 6.0000 127.6000

 *** SCF run converged in 18 steps ***
 ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -2.0D+00
 FORCES| Atomic forces [hartree/bohr]
 FORCES| 1 0.1 0.0 0.0 0.1
 FORCES| 2 -0.1 0.0 0.0 0.1
 FORCES| Sum 0.0 0.0 0.0
 STRESS| Analytical stress tensor [GPa]
 STRESS| x 2 0.1 0
 STRESS| y 0.1 3 0
 STRESS| z 0 0 4
 STRESS| Eigenvectors and eigenvalues of the analytical stress tensor [GPa]
 STRESS| x 1 0 0
 STRESS| y 0 1 0
 STRESS| z 0 0 1
 PROGRAM ENDED AT
"""


class CP2KDatasetTests(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms("GeTe", positions=[[0.5] * 3, [1.0] * 3], cell=[4] * 3, pbc=True)

    def test_cp2k_2025_label_and_virial(self):
        label = read_cp2k_label(LOG, self.atoms, exit_code=0)
        self.assertAlmostEqual(label.energy_ev, -54.422772491976)
        native_heading = LOG.replace("[a.u.]:", "[hartree]")
        self.assertEqual(read_cp2k_label(native_heading, self.atoms, exit_code=0).energy_ev,
                         label.energy_ev)
        np.testing.assert_allclose(label.forces_ev_angstrom[:, 0], [5.14220675, -5.14220675])
        self.assertAlmostEqual(label.stress_gpa[0, 1], 0.1)
        self.assertAlmostEqual(label.virial_ev[0, 1], 6.4 / 160.2176634)
        for old, new in ((" 2 -0.1", " 1 -0.1"), ("1.000000 1.000000", "1.100000 1.000000"),
                         ("SCF run converged", "SCF run NOT converged"),
                         ("64.000000", "65.000000"), ("[a.u.]:", "[eV]:")):
            with self.subTest(old=old), self.assertRaises(ValueError):
                read_cp2k_label(LOG.replace(old, new), self.atoms, exit_code=0)
        with self.assertRaises(ValueError):
            read_cp2k_label(LOG, self.atoms, exit_code=1)

    def test_group_and_geometry_are_independent_checks(self):
        label = read_cp2k_label(LOG, self.atoms, exit_code=0)
        train = DatasetFrame("train-one", "train", "source-a", self.atoms, label)
        shifted = self.atoms.copy()
        shifted.positions[0, 0] += 0.2
        same_group = DatasetFrame("test-one", "test", "source-a", shifted)
        audit = audit_partitions([train, same_group])
        self.assertEqual(audit["shared_source_groups"], ["source-a"])
        self.assertEqual(audit["shared_geometries"], [])
        reordered = self.atoms[[1, 0]]
        reordered.positions[1, 0] += 4
        self.assertEqual(geometry_fingerprint(reordered), geometry_fingerprint(self.atoms))
        duplicate = DatasetFrame("test-two", "test", "source-b", reordered)
        self.assertEqual(len(audit_partitions([train, duplicate])["shared_geometries"]), 1)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "export"
            with self.assertRaisesRegex(ValueError, "overlap"):
                write_nep_dataset(target, [train, same_group])
            self.assertFalse(target.exists())

    def test_candidate_export_round_trip(self):
        label = read_cp2k_label(LOG, self.atoms, exit_code=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset"
            frame = DatasetFrame("one", "candidate", "source-a", self.atoms, label)
            write_nep_dataset(path, [frame])
            out = read(path / "candidate.xyz")
            np.testing.assert_allclose(out.get_forces(), label.forces_ev_angstrom)
            np.testing.assert_allclose(out.info["virial"], label.virial_ev)
            self.assertAlmostEqual(out.get_potential_energy(), label.energy_ev)
            self.assertFalse((path / "train.xyz").exists())


if __name__ == "__main__":
    unittest.main()
