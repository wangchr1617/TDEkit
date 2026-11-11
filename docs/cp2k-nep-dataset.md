# CP2K labels and NEP datasets

The label reader accepts one completed Quickstep `ENERGY_FORCE` output
with the W4 indexed `FORCES|` Hartree/bohr table, a GPa stress tensor, and
printed cell and Cartesian coordinates. The caller supplies the process
exit code and the frozen ASE structure. The reader checks normal
termination, SCF convergence, cell and atom ordering, and coordinate
agreement modulo periodic images. It converts total energy to eV, forces
to eV/Å, and CP2K's pressure-positive stress to virial in eV using
`virial = stress_GPa * volume_A3 / 160.2176634`. The parser and geometry
checks are adapted from W4's `verify_label.py` and `audit_production.py`.

```python
from pathlib import Path

from ase.io import read
from tdekit.cp2k_labels import read_cp2k_label
from tdekit.nep_dataset import DatasetFrame, write_nep_dataset

case = Path("existing-single-point")
atoms = read(case / "frozen_structure.xyz")
label = read_cp2k_label(
    (case / "output.log").read_text(),
    atoms,
    exit_code=int((case / "exit_code").read_text()),
)
frame = DatasetFrame(
    "case-001", "candidate", "thermal-trajectory-005", atoms, label,
    provenance={"source_frame": 42, "source_path": "trajectory-005.xyz"},
)
write_nep_dataset("new-nep-dataset", [frame])
```

Use the same source-group ID for related frames across all input files.
A trajectory, simulation seed or shared parent structure is often a better
group ID than the filename. Supply externally fixed `train`,
`development` and `test` roles to build a split. The exporter refuses a
shared source group or geometry between test and either train or
development, and rejects duplicate geometries within each partition.
`audit_partitions` also accepts precomputed `PartitionEntry` records for
existing manifests. Its overlap check covers the supplied group IDs and
same-cell geometries up to atom ordering, periodic wrapping and the stated
rounding precision. It does not establish trajectory independence when
parent information is incomplete.

Use `candidate` to stage a dataset before approving training membership.
It writes `candidate.xyz` with total energy, forces and virial, plus a
JSON manifest with frame order, source groups, geometry fingerprints and
dataset SHA-256. `bridge` entries are excluded from XYZ exports.

## Reproduce the W4 audit

From an installed TDEkit source checkout:

```bash
python examples/w4_cp2k_dataset.py \
  /path/to/w4/workdir/nep_train/public_cp2k_v1_20261104/cp2k_relabel_168_20261110 \
  /path/to/extracted/public/Database \
  /path/to/fresh/output
```

The script reads archived output only. It rechecks the frozen ASI candidate
list, CP2K input and output hashes, original structures, six public source
frames and their reference labels, the original numerical bridge criteria,
and a round trip of all exported labels. It also audits the historical
Iter7 and public splits. Results are written to `w4_audit.json`. Public
bridge frames are development references and are not copied into the
candidate dataset.
