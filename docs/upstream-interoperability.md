# Upstream interoperability

## Irradiation debris catalogs

`tdekit.debris_catalog.load_debris_catalog` reads the directory layout
documented by [irradiapy](https://irradiapy.readthedocs.io/en/latest/debris_datasets.html).
Each dataset contains `meta.json` and directories named by the recoil
energy in keV. Metadata includes recoil species, target composition,
lattice, interatomic potentials, electronic interactions, DOI, and
contributors. TDEkit retains this metadata and converts the energy
keys to eV.

```python
from tdekit.debris_catalog import load_debris_catalog

catalog = load_debris_catalog("/path/to/debris/dataset")
for energy_ev, files in catalog.files_by_energy_ev.items():
    print(energy_ev, len(files))
```

This read-only index leaves the XYZ payload and coordinate conventions
with the source dataset. Sparse debris coordinates cannot be used
directly as complete atomic configurations for Wigner–Seitz occupancy.

## GPUMD-Wizard radiation trajectories

The [primary radiation tutorial](https://github.com/Jonsnow-willow/GPUMD-Wizard/blob/d41bd8f922358b08e7d49f9175eb2f11c25404fa/tutorials/Molecular_Dynamics/run_radiation_damage/run_primary_radiaion.py)
uses `dump_exyz` and writes the run under `radiation/cascade`. After an
external run finishes, supply its `dump.xyz` and a separate single-frame
reference with the intended crystal sites and species.

```bash
python examples/analyze_gpumd_trajectory.py \
  radiation/cascade/dump.xyz reference.xyz \
  --cluster-cutoff-angstrom 3.0 --describe-clusters > occupancy.jsonl
```

Each line contains one frame's defect counts, site records, optional
cluster geometry and size histogram. Physical time is read from `Time`
when supplied. The same command accepts GPUMD extended-XYZ trajectories
from other workflows. It uses TDEkit's optional OVITO backend.

## Scope of upstream references

| Project | License | Relevant capability |
| --- | --- | --- |
| [ezcascades](https://github.com/mb4512/ezcascades/blob/cee69feafea9504952de28caff9f708fe01cb95b/geted.py) | CC0-1.0 | Directional displacement threshold statistics for alloys. TDEkit reports probability and sampled crossing intervals from independent endpoints. |
| [CSaransh](https://github.com/haptork/csaransh/blob/3480ae8fb2f386950e4717dc6ebc4f1ef9bd2311/csaransh_pp/src/cluster2features.cpp) | GPL-3.0 | Geometric descriptors and histograms for cascade defect clusters. TDEkit computes reference-site gyration and anisotropy with periodic winding detection. |
| [irradiapy](https://github.com/acgc99/irradiapy/blob/8eb5721604e8f1818a0f08fccc4da17e0c9d44ef/irradiapy/debris_dataset.py) | MIT | Energy-indexed debris datasets with provenance. TDEkit implements its own catalog reader for this layout. |
| [GPUMD-Wizard](https://github.com/Jonsnow-willow/GPUMD-Wizard/blob/d41bd8f922358b08e7d49f9175eb2f11c25404fa/README.md) | AGPL-3.0 | ASE-based GPUMD irradiation workflows. TDEkit analyzes the existing GPUMD extended-XYZ output directly. |

Upstream projects are references for methods and file formats. TDEkit
implements the integration in its own modules.
