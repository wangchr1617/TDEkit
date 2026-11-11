import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms

from tdekit import TDESearch
from tdekit.core import run_gpumd, set_pka


class WorkflowTests(unittest.TestCase):
    def test_missing_or_invalid_defect_summary_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            search = TDESearch(workdir)
            with self.assertRaises(FileNotFoundError):
                search.read_frenkel_pairs(workdir)
            summary = workdir / "final_defect_summary.txt"
            for value in ("", "bad", "-1", "0"):
                summary.write_text(f"Timestep\tFrenkel_pairs\n1\t{value}\n")
                if value == "0":
                    self.assertEqual(search.read_frenkel_pairs(workdir), 0)
                else:
                    with self.assertRaises(ValueError):
                        search.read_frenkel_pairs(workdir)

    def test_disabled_analysis_cannot_supply_probability(self):
        with patch.object(TDESearch, "run_cascade_simulation", return_value=(-1, "case")):
            with self.assertRaises(ValueError):
                TDESearch().get_fp_probability(5.0, 1, defects_analysis=False)

    def test_gpumd_runs_in_case_directory_with_explicit_executable(self):
        atoms = Atoms("Ar2", positions=[[1, 1, 1], [4, 4, 4]], cell=[8] * 3, pbc=True)
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            potential = workdir / "original.txt"
            potential.write_text("test potential\n")
            engine = workdir / "engine"
            engine.write_text(
                f"#!{sys.executable}\n"
                "from pathlib import Path\n"
                "assert Path('model.xyz').is_file()\n"
                "assert Path('run.in').read_text() == 'potential nep.txt\\n'\n"
                "assert Path('nep.txt').read_text() == 'test potential\\n'\n"
                "print('executed')\n"
            )
            engine.chmod(0o700)
            cwd = Path.cwd()
            case = workdir / "case"
            run_gpumd(atoms, case, ["potential nep.txt"], potential, executable=engine)
            self.assertEqual(Path.cwd(), cwd)
            self.assertEqual((case / "gpumd.out").read_text(), "executed\n")
            with self.assertRaises(FileExistsError):
                run_gpumd(atoms, case, [], potential, executable=engine)
            engine.write_text(f"#!{sys.executable}\nraise SystemExit(3)\n")
            with self.assertRaises(subprocess.CalledProcessError):
                run_gpumd(atoms, workdir / "failed", [], potential, executable=engine)
            self.assertTrue((workdir / "failed" / "gpumd.out").exists())

    def test_zero_energy_control_preserves_velocities(self):
        atoms = Atoms("GeTe", positions=[[0, 0, 0], [2, 2, 2]], cell=[8] * 3, pbc=True)
        initial = [[0.001, 0.0, 0.0], [0.0, -0.002, 0.0]]
        atoms.info["velocities"] = initial
        set_pka(atoms, 0, [1, 0, 0], index=0, mode="impulse")
        np.testing.assert_array_equal(atoms.info["velocities"], initial)


if __name__ == "__main__":
    unittest.main()
