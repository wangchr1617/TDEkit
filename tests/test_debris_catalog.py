import json
import tempfile
import unittest
from pathlib import Path

from tdekit.debris_catalog import load_debris_catalog


class DebrisCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.metadata = {
            "recoil": "W", "target": {"W": 1.0}, "lattice": "bcc",
            "interatomic_potentials": ["potential-reference"], "electronic_interactions": "SRIM",
            "doi": "dataset-reference", "contributors": ["Research group"],
        }
        (self.root / "meta.json").write_text(json.dumps(self.metadata))

    def write_case(self, energy, name):
        folder = self.root / energy
        folder.mkdir(exist_ok=True)
        path = folder / name
        path.write_text("0\nEmpty synthetic debris fixture\n")
        return path

    def test_indexes_sorted_bins_and_converts_kev(self):
        a = self.write_case("0.5", "a.xyz")
        b = self.write_case("0.5", "b.xyz")
        c = self.write_case("10", "a.xyz")
        self.write_case("notes", "ignored.xyz")
        catalog = load_debris_catalog(self.root)
        self.assertEqual(catalog.metadata, self.metadata)
        self.assertEqual(list(catalog.files_by_energy_ev.items()),
                         [(500, (a, b)), (10000, (c,))])

    def test_rejects_ambiguous_or_nonfinite_bins(self):
        self.write_case("1", "a.xyz")
        self.write_case("1.0", "b.xyz")
        with self.assertRaisesRegex(ValueError, "Duplicate energy"):
            load_debris_catalog(self.root)
        (self.root / "1.0" / "b.xyz").unlink()
        self.write_case("nan", "a.xyz")
        with self.assertRaisesRegex(ValueError, "positive finite"):
            load_debris_catalog(self.root)

    def test_missing_metadata_is_rejected(self):
        del self.metadata["electronic_interactions"]
        (self.root / "meta.json").write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, "electronic_interactions"):
            load_debris_catalog(self.root)
