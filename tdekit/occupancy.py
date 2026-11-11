"""Frame-resolved Wigner–Seitz occupancy using the optional OVITO backend."""

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class SiteOccupancy:
    site_index: int
    site_id: Optional[int]
    reference_species: str
    species_counts: dict[str, int]
    atom_ids: Optional[tuple[int, ...]]
    kind: str


@dataclass(frozen=True)
class ClusterGeometry:
    periodic_axes: tuple[bool, bool, bool]
    radius_of_gyration_angstrom: Optional[float]
    principal_axis: Optional[tuple[float, float, float]]
    relative_shape_anisotropy: Optional[float]


@dataclass(frozen=True)
class DefectSiteCluster:
    kind: str
    site_indices: tuple[int, ...]
    geometry: Optional[ClusterGeometry] = None


@dataclass(frozen=True)
class OccupancyFrame:
    frame: int
    time_fs: Optional[float]
    vacancy_count: int
    overoccupied_site_count: int
    excess_atom_count: int
    antisite_count: int
    sites: tuple[SiteOccupancy, ...]
    clusters: Optional[tuple[DefectSiteCluster, ...]] = None

    def cluster_size_histogram(self) -> Optional[dict[str, dict[int, int]]]:
        if self.clusters is None:
            return None
        counts = {}
        for cluster in self.clusters:
            counts.setdefault(cluster.kind, Counter())[len(cluster.site_indices)] += 1
        return {kind: dict(histogram) for kind, histogram in counts.items()}


def _species(particles):
    prop = particles["Particle Type"]
    names = {kind.id: kind.name or str(kind.id) for kind in prop.types}
    return np.asarray([names[int(value)] for value in prop])


def _identifiers(particles):
    if "Particle Identifier" not in particles:
        return None
    ids = np.asarray(particles["Particle Identifier"])
    if np.unique(ids).size != ids.size:
        raise ValueError("Particle identifiers must be unique within a frame")
    return ids


def _cluster_geometry(group, finder, data):
    images = {group[0]: np.zeros(3, dtype=int)}
    pending = deque([group[0]])
    members = set(group)
    periodic_axes = np.zeros(3, dtype=bool)
    while pending:
        site = pending.popleft()
        for neighbor in finder.find(site):
            if neighbor.index not in members:
                continue
            image = images[site] + neighbor.pbc_shift
            if neighbor.index in images:
                periodic_axes |= image != images[neighbor.index]
            else:
                images[neighbor.index] = image
                pending.append(neighbor.index)

    flags = tuple(map(bool, periodic_axes))
    if periodic_axes.any():
        return ClusterGeometry(flags, None, None, None)
    if len(images) != len(group):
        raise RuntimeError("Cannot unwrap a disconnected cluster")
    shifts = np.array([images[site] for site in group])
    positions = np.asarray(data.particles.positions)[group] + shifts @ np.asarray(data.cell)[:, :3].T
    centered = positions - positions.mean(axis=0)
    eigenvalues, axes = np.linalg.eigh(centered.T @ centered / len(group))
    eigenvalues = eigenvalues.clip(min=0)
    total = eigenvalues.sum()
    if total == 0:
        return ClusterGeometry(flags, 0.0, None, None)
    axis = None
    if not np.isclose(eigenvalues[-1], eigenvalues[-2], rtol=1e-8, atol=0):
        principal = axes[:, -1]
        principal *= np.sign(principal[np.argmax(abs(principal))])
        axis = tuple(map(float, principal))
    anisotropy = 1.5 * np.sum((eigenvalues / total)**2) - 0.5
    return ClusterGeometry(flags, float(np.sqrt(total)), axis, float(np.clip(anisotropy, 0, 1)))


def _cluster_sites(sites, reference, cutoff, describe):
    from ovito.data import CutoffNeighborFinder, DataCollection
    from ovito.modifiers import ClusterAnalysisModifier

    by_kind = {}
    for site in sites:
        by_kind.setdefault(site.kind, []).append(site.site_index)
    clusters = []
    for kind, indices in by_kind.items():
        data = DataCollection()
        data.create_cell(np.asarray(reference.cell), pbc=reference.cell.pbc)
        data.create_particles().create_property(
            "Position", data=np.asarray(reference.particles.positions)[indices]
        )
        data.apply(ClusterAnalysisModifier(cutoff=cutoff))
        groups = {}
        for local_index, label in enumerate(data.particles["Cluster"]):
            groups.setdefault(int(label), []).append(local_index)
        finder = CutoffNeighborFinder(cutoff, data) if describe else None
        for group in groups.values():
            geometry = _cluster_geometry(group, finder, data) if finder is not None else None
            clusters.append(DefectSiteCluster(kind, tuple(indices[i] for i in group), geometry))
    return tuple(clusters)


def analyze_occupancy(trajectory, *, reference, map_to_reference=False,
                      cluster_cutoff_angstrom=None, describe_clusters=False) -> Iterator[OccupancyFrame]:
    """Yield one occupancy summary per frame using a single reference structure.

    Antisites are singly occupied sites with a species mismatch. Reported
    identifiers come from the inputs. The reader does not synthesize IDs.
    Time is read from the GPUMD-compatible Time attribute in fs, when present.
    Optional clustering uses reference-site distances within each defect kind.
    """
    try:
        from ovito.io import import_file
        from ovito.modifiers import WignerSeitzAnalysisModifier
        from ovito.pipeline import FileSource, ReferenceConfigurationModifier
    except ImportError as error:
        raise RuntimeError("Install TDEkit with the analysis extra to use OVITO") from error

    if describe_clusters and cluster_cutoff_angstrom is None:
        raise ValueError("Cluster geometry requires a cluster cutoff")
    if cluster_cutoff_angstrom is not None:
        if not np.isfinite(cluster_cutoff_angstrom) or cluster_cutoff_angstrom <= 0:
            raise ValueError("Expected a positive finite cluster cutoff in Å")

    pipeline = import_file(str(Path(trajectory)))
    modifier = WignerSeitzAnalysisModifier(output_displaced=True)
    modifier.reference = FileSource()
    modifier.reference.load(str(Path(reference)))
    if modifier.reference.num_frames != 1:
        raise ValueError("Expected exactly one frame in the reference file")
    ref = modifier.reference.compute(0)
    site_species = _species(ref.particles)
    site_ids = _identifiers(ref.particles)
    if map_to_reference:
        if not all(ref.cell.pbc):
            raise ValueError("Affine mapping requires a fully periodic reference")
        modifier.affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference
    pipeline.modifiers.append(modifier)

    for frame in range(pipeline.source.num_frames):
        if map_to_reference and not all(pipeline.source.compute(frame).cell.pbc):
            raise ValueError("Affine mapping requires fully periodic trajectory frames")
        data = pipeline.compute(frame)
        particles = data.particles
        indices = np.asarray(particles["Site Index"])
        species = _species(particles)
        ids = _identifiers(particles)
        counts = np.bincount(indices, minlength=len(site_species))
        antisite = np.zeros(len(counts), dtype=bool)
        single_mismatch = (counts[indices] == 1) & (species != site_species[indices])
        antisite[indices[single_mismatch]] = True
        order = np.argsort(indices, kind="stable")
        offsets = np.concatenate(([0], np.cumsum(counts)))
        sites = []
        for site in np.flatnonzero((counts != 1) | antisite):
            occupants = order[offsets[site]:offsets[site + 1]]
            kind = "vacancy" if counts[site] == 0 else "antisite" if antisite[site] else "overoccupied"
            sites.append(SiteOccupancy(
                int(site), None if site_ids is None else int(site_ids[site]),
                str(site_species[site]), dict(Counter(map(str, species[occupants]))),
                None if ids is None else tuple(map(int, ids[occupants])), kind,
            ))
        clusters = None if cluster_cutoff_angstrom is None else _cluster_sites(
            sites, ref, cluster_cutoff_angstrom, describe_clusters
        )
        time = data.attributes.get("Time")
        yield OccupancyFrame(
            frame, None if time is None else float(time),
            int(np.count_nonzero(counts == 0)), int(np.count_nonzero(counts > 1)),
            int(np.maximum(counts - 1, 0).sum()), int(antisite.sum()), tuple(sites), clusters,
        )
