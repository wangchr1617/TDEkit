import unittest

from tdekit.occupancy import DefectSiteCluster, OccupancyFrame
from tdekit.occupancy_summary import OccupancyTrial, summarize_trials


def endpoint(vacancies, clusters=None, time=5):
    return OccupancyFrame(2, time, vacancies, 0, 0, 0, (), clusters)


class OccupancySummaryTests(unittest.TestCase):
    def test_valid_zero_and_missing_cluster_or_endpoint(self):
        rows = [
            OccupancyTrial(1, "+x", "a", endpoint(2, (DefectSiteCluster("vacancy", (0, 1)),))),
            OccupancyTrial(1, "+x", "b", endpoint(0, ())),
            OccupancyTrial(1, "+x", "c", endpoint(1)),
            OccupancyTrial(1, "+x", "d", None),
            OccupancyTrial(1, "-x", "a", None),
        ]
        forward, reverse = summarize_trials(rows, observation_time_fs=5)
        self.assertEqual((forward.n_valid, forward.n_clustered), (3, 2))
        self.assertEqual(forward.missing_repeats, ("d",))
        self.assertEqual(forward.vacancies.mean, 1)
        self.assertEqual(forward.vacancies.sample_sd, 1)
        self.assertEqual(forward.cluster_histogram, {"vacancy": {2: 1}})
        self.assertIsNone(reverse.vacancies.mean)
        self.assertIsNone(reverse.cluster_histogram)

    def test_duplicate_repeats_and_inconsistent_time_fail(self):
        row = OccupancyTrial(1, "+x", "a", endpoint(0))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_trials([row, row], observation_time_fs=5)
        for time in (None, 6, float("nan")):
            with self.subTest(time=time), self.assertRaisesRegex(ValueError, "observation time"):
                summarize_trials([OccupancyTrial(1, "+x", "a", endpoint(0, time=time))],
                                 observation_time_fs=5)

    def test_single_valid_cluster_free_trial(self):
        summary, = summarize_trials([OccupancyTrial(2, "+z", "a", endpoint(0, ()))],
                                    observation_time_fs=5)
        self.assertEqual(summary.vacancies.mean, 0)
        self.assertIsNone(summary.vacancies.sample_sd)
        self.assertEqual(summary.cluster_histogram, {})


class DamageScanTests(unittest.TestCase):
    def test_counts_and_preserves_all_observed_crossings(self):
        from tdekit.occupancy_summary import scan_damage_probability

        trials = [
            OccupancyTrial(e, "+x", "a", endpoint(n) if n is not None else None)
            for e, n in [(8, 0), (1, 0), (3, 1), (4, None), (5, 1), (7, 1), (9, 1)]
        ]
        trials.extend([OccupancyTrial(3, "+x", "b", endpoint(0)),
                       OccupancyTrial(3, "+x", "c", None)])
        scan, = scan_damage_probability(trials, observation_time_fs=5,
                                        damaged=lambda frame: frame.vacancy_count >= 1)
        self.assertEqual([p.energy_ev for p in scan.points], [1, 3, 4, 5, 7, 8, 9])
        self.assertEqual(scan.upward_brackets_ev, ((1, 3), (8, 9)))
        self.assertTrue(scan.observed_decrease)
        point = scan.points[1]
        self.assertEqual((point.n_damaged, point.n_valid, point.missing_repeats), (1, 2, ("c",)))
        self.assertAlmostEqual(point.wilson_95[0], 0.0945312)
        self.assertIsNone(scan.points[2].probability)

    def test_missing_energy_is_not_bridged_and_predicate_is_explicit(self):
        from tdekit.occupancy_summary import scan_damage_probability

        trials = [OccupancyTrial(1, "-x", "a", endpoint(0)),
                  OccupancyTrial(2, "-x", "a", None),
                  OccupancyTrial(3, "-x", "a", endpoint(1))]
        scan, = scan_damage_probability(trials, observation_time_fs=5,
                                        damaged=lambda frame: frame.vacancy_count > 0)
        self.assertEqual(scan.upward_brackets_ev, ())
        with self.assertRaisesRegex(ValueError, "boolean"):
            scan_damage_probability(trials, observation_time_fs=5, damaged=lambda frame: 1)
