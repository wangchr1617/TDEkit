"""Evaluate, fit and export the flexible ZBL term of legacy NEP models."""

from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
import re

import numpy as np
from ase.data import atomic_numbers

from .core import PathLike


@dataclass(frozen=True)
class FlexibleZBL:
    """Cosine-switched four-exponential ZBL term for one element pair."""

    inner: float
    outer: float
    amplitudes: tuple[float, float, float, float]
    decays: tuple[float, float, float, float]

    def __post_init__(self):
        if not np.isfinite([self.inner, self.outer]).all() or not 0 <= self.inner < self.outer:
            raise ValueError("Expected finite cutoffs with 0 <= inner < outer")
        if len(self.amplitudes) != 4 or not np.isfinite(self.amplitudes).all():
            raise ValueError("Expected four finite amplitudes")
        if len(self.decays) != 4 or not np.isfinite(self.decays).all() or min(self.decays) <= 0:
            raise ValueError("Expected four positive finite decay coefficients")

    def _basis(self, pair, distances):
        z1, z2 = (atomic_numbers[symbol] for symbol in pair)
        if min(z1, z2) <= 0:
            raise ValueError("ZBL requires real chemical elements")
        r = np.asarray(distances, dtype=float)
        if not np.isfinite(r).all() or np.any(r <= 0):
            raise ValueError("Distances must be finite and positive")
        inverse_screening = 2.134563 * (z1**0.23 + z2**0.23)
        slope = np.asarray(self.decays) * inverse_screening
        phase_scale = np.pi / (self.outer - self.inner)
        phase = np.clip((r - self.inner) * phase_scale, 0, np.pi)
        switch = np.where(r >= self.outer, 0.0, 0.5 * (1 + np.cos(phase)))
        minus_derivative = np.where(
            (r > self.inner) & (r < self.outer), 0.5 * phase_scale * np.sin(phase), 0.0
        )
        raw = (14.399645 * z1 * z2 / r[..., None]) * np.exp(-r[..., None] * slope)
        energy = raw * switch[..., None]
        force = energy * (slope + 1 / r[..., None]) + raw * minus_derivative[..., None]
        return energy, force

    def evaluate(self, pair, distances):
        """Return pair energy (eV) and outward radial force (eV/Å)."""
        energy, force = self._basis(pair, distances)
        return energy @ self.amplitudes, force @ self.amplitudes

    def parameters(self):
        return (self.inner, self.outer, *[value for term in zip(self.amplitudes, self.decays)
                                         for value in term])


def fit_amplitudes(initial, pair, distances, target_energy, target_force, *,
                   energy_scale, force_scale):
    """Fit four nonnegative amplitudes at fixed cutoffs and decays.

    Targets must be the ZBL contribution. Subtract the fixed NEP contribution
    from total reference energies and forces before fitting.
    """
    from scipy.optimize import nnls

    scales = np.asarray([energy_scale, force_scale], dtype=float)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Energy and force scales must be positive and finite")
    distances, target_energy, target_force = (
        np.asarray(values, dtype=float) for values in (distances, target_energy, target_force)
    )
    if distances.ndim != 1 or target_energy.shape != distances.shape or target_force.shape != distances.shape:
        raise ValueError("Distances, energy and force targets must be equally sized vectors")
    if not (np.isfinite(target_energy).all() and np.isfinite(target_force).all()):
        raise ValueError("Fit targets must be finite")

    energy, force = initial._basis(pair, distances)
    design = np.vstack((energy / energy_scale, force / force_scale))
    targets = np.concatenate((target_energy / energy_scale, target_force / force_scale))
    if not len(design) or np.linalg.matrix_rank(design) < 4:
        raise ValueError("The supplied distances cannot resolve four amplitudes")
    amplitudes, _ = nnls(design, targets)
    return FlexibleZBL(initial.inner, initial.outer, tuple(amplitudes), initial.decays)


def _validate_payload(lines, end, version, species_count):
    cutoff, orders, basis, angular, network = [line.split() for line in lines[2:7]]
    expected = (b"cutoff", b"n_max", b"basis_size", b"l_max", b"ANN")
    if any(not tokens or tokens[0] != key
           for tokens, key in zip((cutoff, orders, basis, angular, network), expected)):
        raise ValueError("Unrecognized legacy NEP model layout")
    radial_order, angular_order = map(int, orders[1:])
    radial_basis, angular_basis = map(int, basis[1:])
    three_body, four_body, five_body = map(int, angular[1:])
    neurons, network_flag = map(int, network[1:])
    if network_flag != 0 or min(radial_order, angular_order, radial_basis,
                                angular_basis, three_body, neurons) < 0:
        raise ValueError("Invalid NEP dimensions")
    channels = three_body + (four_body == 2) + (five_body == 1)
    dimensions = radial_order + 1 + (angular_order + 1) * channels
    single_network = (dimensions + 2) * neurons
    network_size = {
        3: single_network + 1,
        4: single_network * species_count + 1,
        5: (single_network + 1) * species_count + 1,
    }[version]
    descriptor_size = species_count**2 * (
        (radial_order + 1) * (radial_basis + 1)
        + (angular_order + 1) * (angular_basis + 1)
    )
    if end != 7 + network_size + descriptor_size + dimensions:
        raise ValueError("NEP scalar count does not match the legacy model header")
    if not np.isfinite([float(line.strip()) for line in lines[7:end]]).all():
        raise ValueError("NEP payload contains nonfinite parameters")


def _read_model(filename):
    lines = Path(filename).read_bytes().splitlines(keepends=True)
    if len(lines) < 7:
        raise ValueError("Incomplete NEP file")
    header = lines[0].decode("ascii").split()
    if len(header) < 3 or not re.fullmatch(r"nep[345]_zbl", header[0]):
        raise ValueError("Expected a legacy nep3_zbl, nep4_zbl or nep5_zbl model")
    species = header[2:]
    if int(header[1]) != len(species) or len(set(species)) != len(species):
        raise ValueError("Invalid NEP species header")
    if any(atomic_numbers.get(symbol, 0) < 1 for symbol in species):
        raise ValueError("Unrecognized element in NEP header")
    zbl = lines[1].split()
    if len(zbl) != 3 or zbl[0] != b"zbl" or any(float(value) != 0 for value in zbl[1:]):
        raise ValueError("Expected a flexible ZBL header (zbl 0 0)")
    pairs = tuple(combinations_with_replacement(species, 2))
    start = len(lines) - 10 * len(pairs)
    _validate_payload(lines, start, int(header[0][3]), len(species))
    parameters = {}
    for index, pair in enumerate(pairs):
        offset = start + 10 * index
        values = [float(line.strip()) for line in lines[offset:offset + 10]]
        parameters[pair] = FlexibleZBL(*values[:2], tuple(values[2::2]), tuple(values[3::2]))
    return lines[:start], parameters


def read_flexible_zbl(filename: PathLike):
    """Read flexible pair parameters in the species order of the NEP header."""
    _, parameters = _read_model(filename)
    return parameters


def write_candidate(source: PathLike, destination: PathLike, parameters):
    """Write a new model preserving every byte before the flexible ZBL tail."""
    body, original = _read_model(source)
    if parameters.keys() != original.keys():
        raise ValueError("Parameters must match all ordered pair keys from the source model")
    tail = "".join(
        f"{value:.17g}\n" for pair in original for value in parameters[pair].parameters()
    ).encode("ascii")
    with Path(destination).open("xb") as stream:
        stream.write(b"".join(body) + tail)
