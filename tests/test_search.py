import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tdekit import TDESearch


class SearchTests(unittest.TestCase):
    def test_growth_checks_exact_upper_bound(self):
        search = TDESearch()
        with patch.object(search, "run_cascade_simulation",
                          side_effect=lambda energy: (int(energy >= 9), "test")) as run:
            self.assertEqual(search.exponential_growth_search(2, 9), (8, 9))
            self.assertEqual([call.args[0] for call in run.call_args_list], [2, 4, 8, 9])

    def test_find_never_exceeds_limit_or_verifies_using_search_result(self):
        search = TDESearch()
        with patch.object(search, "get_fp_probability", side_effect=[0, 1, 0, 0]) as run:
            with self.assertRaisesRegex(RuntimeError, "verification"):
                search.find_tde(init_energy=2, max_energy=3, precision=1)
            self.assertEqual([call.args[0] for call in run.call_args_list], [2, 3, 2.5, 3])

        with patch.object(search, "get_fp_probability", return_value=0) as run:
            with self.assertRaisesRegex(RuntimeError, "3"):
                search.find_tde(init_energy=2, max_energy=3)
            self.assertEqual([call.args[0] for call in run.call_args_list], [2, 3])

    def test_midpoint_failure_requires_a_new_success_at_upper_bound(self):
        search = TDESearch()
        with patch.object(search, "get_fp_probability", side_effect=[0, 1, 0, 0.75]) as run:
            self.assertEqual(search.find_tde(2, 3, precision=1, num_samples=2,
                                             verification_samples=4), (3, 0.75))
            self.assertEqual([call.args for call in run.call_args_list],
                             [(2, 2), (3, 2), (2.5, 4), (3, 4)])

    def test_held_out_restarts_are_disjoint_and_used_for_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / "search.xyz"
            held_out = Path(directory) / "held_out.xyz"
            initial.write_text("initial")
            held_out.write_text("held out")
            search = TDESearch(directory)
            with patch.object(search, "get_fp_probability", side_effect=[0, 1, 1]) as run:
                with self.assertRaisesRegex(ValueError, "disjoint"):
                    search.find_tde(2, 3, num_samples=1, verification_samples=1,
                                    sample_inputs=[initial], verification_inputs=[initial])
                self.assertEqual(search.find_tde(2, 3, num_samples=1, verification_samples=1,
                                                 sample_inputs=[initial], verification_inputs=[held_out]),
                                 (2.5, 1))
                self.assertEqual([call.kwargs["sample_inputs"] for call in run.call_args_list],
                                 [(initial,), (initial,), (held_out,)])

    def test_sample_inputs_are_used_and_duplicates_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = [Path(directory) / name for name in ("a.xyz", "b.xyz")]
            a.write_text("state a")
            b.write_text("state b")
            search = TDESearch(directory)
            with patch.object(search, "run_cascade_simulation",
                              side_effect=[(0, "a"), (1, "b")]) as run:
                self.assertEqual(search.get_fp_probability(5, 2, sample_inputs=[a, b]), 0.5)
                self.assertEqual([call.kwargs["input_file"] for call in run.call_args_list], [a, b])
                b.write_bytes(a.read_bytes())
                with self.assertRaisesRegex(ValueError, "distinct"):
                    search.get_fp_probability(5, 2, sample_inputs=[a, b])
                self.assertEqual(run.call_count, 2)

    def test_case_receipt_survives_cleanup_and_ignores_stale_summary(self):
        def run_case(*, dirname, **kwargs):
            dirname.mkdir(parents=True)
        def analyze(*args, output_dir, **kwargs):
            (output_dir / "final_defect_summary.txt").write_text(
                "Timestep\tFrenkel_pairs\n1\t2\n"
            )
        with tempfile.TemporaryDirectory() as directory:
            search = TDESearch(directory)
            search.work_dir.mkdir()
            (search.work_dir / "final_defect_summary.txt").write_text(
                "Timestep\tFrenkel_pairs\n1\t0\n"
            )
            with patch("tdekit.tde.run_cascade", side_effect=run_case):
                with patch.object(search, "run_defects_analyzer", side_effect=analyze):
                    count, case = search.run_cascade_simulation(3)
            self.assertEqual(count, 2)
            self.assertFalse((search.work_dir / case).exists())
            receipt = json.loads((search.work_dir / f"{case}.json").read_text())
            self.assertEqual((receipt["energy_ev"], receipt["frenkel_pairs"]), (3, 2))


if __name__ == "__main__":
    unittest.main()
