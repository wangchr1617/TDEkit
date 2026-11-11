"""Read the metadata and energy bins of an irradiapy-style debris dataset."""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

_REQUIRED_FIELDS = {
    "recoil", "target", "lattice", "interatomic_potentials",
    "electronic_interactions", "doi", "contributors",
}


@dataclass(frozen=True)
class DebrisCatalog:
    directory: Path
    metadata: dict
    files_by_energy_ev: dict[float, tuple[Path, ...]]


def load_debris_catalog(directory) -> DebrisCatalog:
    """Index one dataset directory without modifying or parsing its XYZ files.

    The source layout uses keV directory names. Returned keys use eV.
    """
    root = Path(directory).resolve(strict=True)
    metadata_file = root / "meta.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected an object in {metadata_file}")
    missing = _REQUIRED_FIELDS.difference(metadata)
    if missing:
        raise ValueError(f"Missing metadata fields in {metadata_file}: {', '.join(sorted(missing))}")

    bins = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        try:
            energy_ev = float(folder.name) * 1000
        except ValueError:
            continue
        files = tuple(sorted(path for path in folder.glob("*.xyz") if path.is_file()))
        if not files:
            continue
        if not isfinite(energy_ev) or energy_ev <= 0:
            raise ValueError(f"Expected a positive finite energy bin: {folder}")
        if energy_ev in bins:
            raise ValueError(f"Duplicate energy bin for {energy_ev:g} eV: {folder}")
        bins[energy_ev] = files
    return DebrisCatalog(root, metadata, dict(sorted(bins.items())))
