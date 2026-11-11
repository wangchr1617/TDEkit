import importlib.util
import tempfile
import unittest
from pathlib import Path

from tdekit.occupancy import analyze_occupancy


@unittest.skipUnless(importlib.util.find_spec("ovito"), "requires OVITO")
class OccupancyTests(unittest.TestCase):
    def test_external_reference_species_time_ids_and_affine_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, trajectory = (Path(tmp) / name for name in ("ref.xyz", "traj.xyz"))
            def frame(rows, scale=1, time=0):
                return (f'{len(rows)}\nLattice="{8*scale} 0 0 0 {8*scale} 0 0 0 {8*scale}" '
                        f'pbc="T T T" Time={time} Properties=species:S:1:pos:R:3:id:I:1\n'
                        + "".join(f"{sp} {x*scale} 1 1 {i}\n" for sp, x, i in rows))
            pristine = [("Ge", 1, 11), ("Te", 5, 12)]
            ref.write_text(frame(pristine))
            trajectory.write_text(frame(pristine, scale=2)
                                  + frame([("Te", 1, 12), ("Ge", 5, 11)], time=2)
                                  + frame([("Ge", 1, 11), ("Te", 1.1, 12)], time=4))
            control, swap, damage = list(analyze_occupancy(trajectory, reference=ref,
                                                           map_to_reference=True))
            self.assertEqual(control.sites, ())
            self.assertEqual(swap.time_fs, 2)
            self.assertEqual(swap.antisite_count, 2)
            self.assertEqual(swap.sites[0].site_id, 11)
            self.assertEqual(swap.sites[0].atom_ids, (12,))
            self.assertEqual(damage.vacancy_count, 1)
            self.assertEqual(damage.excess_atom_count, 1)
            self.assertEqual(damage.sites[0].species_counts, {"Ge": 1, "Te": 1})
            self.assertEqual(damage.antisite_count, 0)

    def test_no_ids_or_clock_and_unequal_atom_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, traj = (Path(tmp) / name for name in ("ref.xyz", "traj.xyz"))
            header = 'Lattice="8 0 0 0 8 0 0 0 8" pbc="T T T" Properties=species:S:1:pos:R:3\n'
            ref.write_text("2\n" + header + "Ge 1 1 1\nGe 5 1 1\n")
            traj.write_text("1\n" + header + "Ge 9 1 1\n")
            result, = analyze_occupancy(traj, reference=ref)
            self.assertIsNone(result.time_fs)
            self.assertEqual((result.vacancy_count, result.excess_atom_count), (1, 0))
            self.assertIsNone(result.sites[0].site_id)
            self.assertIsNone(result.sites[0].atom_ids)

    def test_clusters_keep_defect_kinds_separate_and_cross_periodic_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, traj = (Path(tmp) / name for name in ("ref.xyz", "traj.xyz"))
            header = 'Lattice="10 0 0 0 10 0 0 0 10" pbc="T T T" Properties=species:S:1:pos:R:3\n'
            ref.write_text("5\n" + header +
                           "Ge 0.2 1 1\nGe 9.8 1 1\nGe 1.2 1 1\nGe 5 1 1\nTe 5.4 1 1\n")
            traj.write_text("4\n" + header +
                            "Ge 1.2 1 1\nGe 1.3 1 1\nTe 5 1 1\nGe 5.4 1 1\n")
            unclustered, = analyze_occupancy(traj, reference=ref)
            self.assertIsNone(unclustered.clusters)
            result, = analyze_occupancy(traj, reference=ref, cluster_cutoff_angstrom=1.1)
            self.assertEqual({(c.kind, c.site_indices) for c in result.clusters}, {
                ("vacancy", (0, 1)), ("overoccupied", (2,)), ("antisite", (3, 4)),
            })
            self.assertEqual(result.cluster_size_histogram(),
                             {"vacancy": {2: 1}, "overoccupied": {1: 1}, "antisite": {2: 1}})
            with self.assertRaisesRegex(ValueError, "cutoff"):
                list(analyze_occupancy(traj, reference=ref, cluster_cutoff_angstrom=0))

    def test_cluster_geometry_finite_and_periodically_connected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, traj = (Path(tmp) / name for name in ("ref.xyz", "traj.xyz"))
            header = 'Lattice="10 0 0 0 10 0 0 0 10" pbc="T T T" Properties=species:S:1:pos:R:3\n'
            traj.write_text("1\n" + header + "Ge 5 7 1\n")
            ref.write_text("3\n" + header + "Ge 0.2 1 1\nGe 9.8 1 1\nGe 5 7 1\n")
            result, = analyze_occupancy(traj, reference=ref, cluster_cutoff_angstrom=0.5,
                                         describe_clusters=True)
            cluster, = result.clusters
            self.assertEqual(cluster.site_indices, (0, 1))
            self.assertEqual(cluster.geometry.periodic_axes, (False, False, False))
            self.assertAlmostEqual(cluster.geometry.radius_of_gyration_angstrom, 0.2)
            self.assertEqual(cluster.geometry.principal_axis, (1.0, 0.0, 0.0))
            self.assertAlmostEqual(cluster.geometry.relative_shape_anisotropy, 1)

            ref.write_text("4\n" + header +
                           "Ge 0.2 1 1\nGe 3.53 1 1\nGe 6.87 1 1\nGe 5 7 1\n")
            result, = analyze_occupancy(traj, reference=ref, cluster_cutoff_angstrom=3.5,
                                         describe_clusters=True)
            cluster, = result.clusters
            self.assertEqual(cluster.geometry.periodic_axes, (True, False, False))
            self.assertIsNone(cluster.geometry.radius_of_gyration_angstrom)
            self.assertIsNone(cluster.geometry.principal_axis)
