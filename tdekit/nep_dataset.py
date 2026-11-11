"""Build NEP extended XYZ datasets with explicit provenance and split checks."""

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from ase import Atoms
from ase.io import write

from .cp2k_labels import CP2KLabel

PARTITIONS = {"train", "test", "development", "candidate", "bridge"}


@dataclass(frozen=True)
class PartitionEntry:
    case_id: str
    partition: str
    source_group: str
    geometry_sha256: str


@dataclass(frozen=True)
class DatasetFrame:
    case_id: str
    partition: str
    source_group: str
    atoms: Atoms
    label: Optional[CP2KLabel] = None
    provenance: dict = field(default_factory=dict)

    def partition_entry(self):
        return PartitionEntry(self.case_id, self.partition, self.source_group,
                              geometry_fingerprint(self.atoms))


def geometry_fingerprint(atoms: Atoms) -> str:
    """Fingerprint the same cell and geometry, modulo wrapping and atom order.

    Uses cell components rounded to 1e-6 Å and fractional coordinates to 1e-6.
    Cell rotation, origin shifts and near-duplicate structures are separate checks.
    """
    if not len(atoms) or atoms.cell.rank != 3:
        raise ValueError("Fingerprint requires a nonempty structure and a full cell")
    cell = atoms.cell.array
    frac = atoms.get_scaled_positions(wrap=False)
    if not np.isfinite(cell).all() or not np.isfinite(frac).all():
        raise ValueError("Fingerprint requires finite positions and cell")
    frac[:, atoms.pbc] %= 1.0
    frac = np.round(frac, 6)
    frac[:, atoms.pbc] %= 1.0
    frac[frac == 0] = 0
    cell = np.round(cell, 6)
    cell[cell == 0] = 0
    numbers = atoms.numbers
    order = np.lexsort((frac[:, 2], frac[:, 1], frac[:, 0], numbers))
    digest = hashlib.sha256(cell.astype("<f8").tobytes())
    digest.update(atoms.pbc.tobytes())
    digest.update(numbers[order].astype("<i4").tobytes())
    digest.update(frac[order].astype("<f8").tobytes())
    return digest.hexdigest()


def audit_partitions(entries: Iterable) -> dict:
    """Audit source-group and exact geometry overlap with a declared test set."""
    entries = [entry.partition_entry() if isinstance(entry, DatasetFrame) else entry for entry in entries]
    if any(entry.partition not in PARTITIONS or not entry.source_group.strip()
           or not entry.case_id.strip() or not entry.geometry_sha256 for entry in entries):
        raise ValueError("Each entry requires a known partition, case ID, source group and geometry")
    if len({entry.case_id for entry in entries}) != len(entries):
        raise ValueError("Dataset case IDs must be unique")
    exposed = [entry for entry in entries if entry.partition in {"train", "development"}]
    test = [entry for entry in entries if entry.partition == "test"]
    groups = sorted({entry.source_group for entry in exposed} & {entry.source_group for entry in test})
    geometries = sorted({entry.geometry_sha256 for entry in exposed} & {entry.geometry_sha256 for entry in test})
    return {
        "partition_counts": dict(Counter(entry.partition for entry in entries)),
        "test_compared": bool(exposed and test),
        "shared_source_groups": groups,
        "shared_geometries": geometries,
    }


def _nep_frame(frame):
    label = frame.label
    if label is None:
        raise ValueError(f"Missing labels: {frame.case_id}")
    atoms = Atoms(numbers=frame.atoms.numbers, positions=frame.atoms.positions,
                  cell=frame.atoms.cell, pbc=frame.atoms.pbc)
    forces, virial, stress = map(np.asarray, (label.forces_ev_angstrom, label.virial_ev, label.stress_gpa))
    if (forces.shape != (len(atoms), 3) or virial.shape != (3, 3) or stress.shape != (3, 3)
            or not all(np.isfinite(value).all() for value in (forces, virial, stress, label.energy_ev))):
        raise ValueError(f"Invalid labels: {frame.case_id}")
    atoms.set_array("forces", forces.copy())
    atoms.info.update(energy=float(label.energy_ev), virial=virial.copy())
    return atoms


def write_nep_dataset(directory, frames: Iterable[DatasetFrame]) -> dict:
    """Write explicit partitions and a provenance manifest to a new directory.

    Bridge frames are omitted from the extended XYZ files. Candidate files
    remain separate from train, development and test exports.
    """
    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    frames = list(frames)
    entries = [frame.partition_entry() for frame in frames]
    report = audit_partitions(entries)
    if report["shared_source_groups"] or report["shared_geometries"]:
        raise ValueError("Source-group or geometry overlap with the test set")
    keys = Counter((entry.partition, entry.geometry_sha256) for entry in entries)
    if any(count > 1 for count in keys.values()):
        raise ValueError("Duplicate geometry within a partition")
    data = {}
    rows = []
    for frame, entry in zip(frames, entries):
        atoms = _nep_frame(frame)
        if entry.partition != "bridge":
            data.setdefault(entry.partition, []).append(atoms)
        rows.append(dict(vars(entry), natoms=len(atoms), provenance=frame.provenance))
    if not data:
        raise ValueError("No labelled frames to export")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        files = {}
        for partition, structures in data.items():
            file = temp / f"{partition}.xyz"
            write(file, structures, format="extxyz", write_results=False)
            files[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        report.update(files=files, frames=rows)
        (temp / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
        destination.mkdir()
        try:
            for file in temp.iterdir():
                file.rename(destination / file.name)
        except OSError:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(temp)
    return report
