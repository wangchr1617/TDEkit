import tempfile
import unittest
from pathlib import Path

import numpy as np

from tdekit.zbl import FlexibleZBL, fit_amplitudes, read_flexible_zbl, write_candidate

ROOT = Path(__file__).resolve().parents[1]


class ZBLTests(unittest.TestCase):
    def test_radial_force_and_switch(self):
        model = FlexibleZBL(1.0, 2.0, (0.18, 0.51, 0.28, 0.03),
                            (3.2, 0.94, 0.40, 0.20))
        distances = np.array([0.6, 1.0, 1.5, 2.0, 2.3])
        energy, force = model.evaluate(("Ge", "Te"), distances)
        h = 1e-6
        above, _ = model.evaluate(("Ge", "Te"), distances + h)
        below, _ = model.evaluate(("Ge", "Te"), distances - h)
        np.testing.assert_allclose(force, (below - above) / (2 * h), rtol=4e-7, atol=2e-5)
        np.testing.assert_array_equal(energy[-2:], 0)
        np.testing.assert_array_equal(force[-2:], 0)

    def test_fit_and_held_out_prediction(self):
        pair = ("Ge", "Te")
        prior = FlexibleZBL(1.0, 2.0, (0.18, 0.51, 0.28, 0.03), (3.2, 0.94, 0.4, 0.2))
        truth = FlexibleZBL(1.0, 2.0, (0.21, 0.54, 0.32, 0.05), prior.decays)
        train = np.linspace(0.35, 1.9, 30)
        test = np.array([0.41, 0.88, 1.21, 1.72])
        energy, force = truth.evaluate(pair, train)
        candidate = fit_amplitudes(prior, pair, train, energy, force,
                                   energy_scale=10.0, force_scale=100.0)
        np.testing.assert_allclose(candidate.evaluate(pair, test), truth.evaluate(pair, test),
                                   atol=1e-8)
        with self.assertRaises(ValueError):
            fit_amplitudes(prior, pair, [0.5], [1.0], [1.0],
                           energy_scale=1.0, force_scale=1.0)

    def test_legacy_pair_order_and_non_destructive_export(self):
        model = ROOT / "test/potentials/0.txt"
        original = model.read_bytes()
        pairs = read_flexible_zbl(model)
        self.assertEqual(list(pairs), [("Te", "Te"), ("Te", "Ge"), ("Ge", "Ge")])
        self.assertLess(pairs["Te", "Ge"].amplitudes[-1], 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.txt"
            write_candidate(model, output, pairs)
            self.assertEqual(read_flexible_zbl(output), pairs)
            self.assertEqual(output.read_bytes().splitlines()[:-30], original.splitlines()[:-30])
            with self.assertRaises(FileExistsError):
                write_candidate(model, output, pairs)
            truncated = Path(directory) / "truncated.txt"
            lines = original.splitlines(keepends=True)
            truncated.write_bytes(b"".join(lines[:7] + lines[8:]))
            with self.assertRaises(ValueError):
                read_flexible_zbl(truncated)
        self.assertEqual(model.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
