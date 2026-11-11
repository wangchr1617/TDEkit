# Frame-resolved lattice occupancy

Install the optional OVITO backend with `pip install ".[analysis]"`.

```python
from tdekit.occupancy import analyze_occupancy

for frame in analyze_occupancy("dump.xyz", reference="reference.xyz",
                               map_to_reference=True):
    print(frame.time_fs, frame.vacancy_count, frame.excess_atom_count,
          frame.antisite_count)
```

The reference file contains one configuration with the intended site
species. For fully periodic cells, the optional affine mapping removes
homogeneous cell deformation before site assignment. Use consistent
species names across inputs, or consistent numeric type IDs for unnamed
types. The iterator processes frames individually and records the
reference index and available input identifiers for defective sites.
A missing identifier remains unset. The optional `Time` attribute is
interpreted as femtoseconds.

The output distinguishes vacant sites, overoccupied sites, excess atoms
and singly occupied sites with a species mismatch. These are occupancy
observations relative to the chosen reference. Vacancy and excess-atom
counts do not identify which atoms and sites form physical Frenkel pairs.
The legacy endpoint analysis retains its existing output schema.

See the [OVITO Wigner–Seitz analysis documentation](https://ovito.org/docs/current/reference/pipelines/modifiers/wigner_seitz_analysis.html).

## Defect-site clusters

Set `cluster_cutoff_angstrom` to a positive cutoff to run OVITO distance
clustering separately for vacancies, overoccupied sites, and antisites.
Distances and periodic boundaries use the reference cell. A cluster's
size counts reference sites, including single-site clusters. Member
indices correspond to the `site_index` fields in the same frame.

```python
for frame in analyze_occupancy("dump.xyz", reference="reference.xyz",
                               cluster_cutoff_angstrom=3.0):
    print(frame.cluster_size_histogram())
```

The histogram maps each defect kind to cluster counts by size.
An empty histogram indicates no clusters in a processed frame.
Clustering describes reference-site connectivity. It does not classify
interstitial atomic geometry or link clusters across frames. See the
[OVITO cluster analysis documentation](https://www.ovito.org/docs/current/reference/pipelines/modifiers/cluster_analysis.html).

## Comparing runs at a fixed observation time

Use `summarize_trials` with one endpoint per independent repeat. Supply
comparable cases with the same protocol, reference definition and cluster
cutoff, and use consistent direction labels and energy values.

```python
from tdekit.occupancy_summary import OccupancyTrial, summarize_trials

frames = analyze_occupancy("dump.xyz", reference="reference.xyz",
                           cluster_cutoff_angstrom=3.0)
endpoint = next(frame for frame in frames if frame.time_fs == 0.2)
summary, = summarize_trials(
    [OccupancyTrial(1.0, "+x", "seed-01", endpoint),
     OccupancyTrial(1.0, "+x", "seed-02", None)],
    observation_time_fs=0.2,
)
print(summary.n_valid, summary.missing_repeats, summary.vacancies.mean)
```

An unavailable endpoint is represented by `None` and excluded from the
mean and sample standard deviation. A valid zero count is included.
Duplicate repeat IDs within an energy/direction group and incompatible
endpoint times are rejected. The pooled cluster histogram counts
clusters by kind and size across the `n_clustered` processed repeats.
A run with clustering enabled and zero clusters contributes to this
denominator. Sample standard deviations require two valid repeats.

## Damage probability and sampled threshold intervals

`scan_damage_probability` uses the same independent-repeat and observation
time checks. The damage rule is an explicit Boolean function of one endpoint.

```python
from tdekit.occupancy_summary import scan_damage_probability

scans = scan_damage_probability(
    trials,
    observation_time_fs=0.2,
    damaged=lambda frame: frame.vacancy_count > 0 and frame.excess_atom_count > 0,
    threshold_probability=0.5,
)
```

This example measures coexistence of vacant sites and excess atoms.
Each energy reports the number of positive and valid repeats, missing
repeat IDs, and a pointwise 95% Wilson interval for the observed fraction.
A threshold bracket is reported for each adjacent sampled energy pair
that crosses the chosen probability upwards. List planned energies with
unavailable outcomes explicitly to prevent bracketing across them.
Observed decreases between adjacent measured probabilities are flagged.
Brackets describe sampled crossings of the point estimates. They are
not confidence intervals on the physical TDE.

### Cluster geometry

With a cutoff set, `describe_clusters=True` computes the equal-site-weight
radius of gyration, principal axis, and relative shape anisotropy. The
algorithm unwraps each cluster using its periodic neighbor connections.
A cycle with a nonzero cell translation indicates periodic winding.
`periodic_axes` records the affected cell directions, and geometry
descriptors are unset for that cluster. A finite cluster can cross the
box boundary without winding. The principal axis is unset when the
largest two gyration eigenvalues are degenerate. A single site has zero
radius and undefined anisotropy.

These descriptors characterize the reference-site positions belonging
to each cluster. They do not assign a dislocation or interstitial motif.
