"""Validate CP2K Quickstep single-point labels against a frozen ASE geometry.

The FORCE_EVAL, force table, and stress parsing and unit constants are
adapted from W4's verify_label.py. Geometry checks follow audit_production.py.
"""

import re
from dataclasses import dataclass

import numpy as np
from ase import Atoms

HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903
GPA_ANGSTROM3_PER_EV = 160.2176634
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
_VECTOR = rf"({_NUMBER})[ \t]+({_NUMBER})[ \t]+({_NUMBER})"


@dataclass(frozen=True)
class CP2KLabel:
    energy_ev: float
    forces_ev_angstrom: np.ndarray
    stress_gpa: np.ndarray
    virial_ev: np.ndarray


def _numbers(rows):
    return np.asarray(
        [[float(value.replace("D", "E").replace("d", "e")) for value in row] for row in rows]
    )


def _one_block(log, heading, name):
    matches = list(re.finditer(heading, log, re.MULTILINE | re.IGNORECASE))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} block, found {len(matches)}")
    return log[matches[0].end():]


def _check_geometry(log, atoms):
    cell = atoms.cell.array
    positions = atoms.get_positions()
    volume = atoms.get_volume()
    if not len(atoms) or not np.isfinite(cell).all() or not np.isfinite(positions).all() or volume <= 0:
        raise ValueError("Expected a finite nonempty reference geometry with positive volume")

    vectors = re.findall(
        rf"^\s*CELL\| Vector ([abc]) \[angstrom\]:\s+{_VECTOR}\s+\|\1\|\s+=\s+({_NUMBER})[ \t]*$",
        log, re.MULTILINE,
    )
    if [row[0] for row in vectors] != list("abc"):
        raise ValueError("Missing or repeated CP2K cell vectors")
    printed = _numbers([row[1:] for row in vectors])
    volumes = re.findall(rf"^\s*CELL\| Volume \[angstrom\^3\]:\s+({_NUMBER})[ \t]*$",
                         log, re.MULTILINE)
    if (len(volumes) != 1 or abs(_numbers([volumes])[0, 0] - volume) > 1e-5
            or not np.allclose(printed[:, :3], cell, rtol=0, atol=5.01e-4)
            or not np.allclose(printed[:, 3], np.linalg.norm(cell, axis=1), rtol=0, atol=1e-6)):
        raise ValueError("CP2K cell disagrees with the reference")

    block = _one_block(log, r"MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM", "coordinate")
    rows = re.findall(
        rf"^[ \t]*(\d+)[ \t]+\d+[ \t]+([A-Z][a-z]?)[ \t]+\d+[ \t]+"
        rf"{_VECTOR}[ \t]+{_NUMBER}[ \t]+{_NUMBER}[ \t]*$", block, re.MULTILINE,
    )
    if ([int(row[0]) for row in rows] != list(range(1, len(atoms) + 1))
            or [row[1] for row in rows] != atoms.get_chemical_symbols()):
        raise ValueError("CP2K coordinate indices or species disagree with the reference")
    delta = np.linalg.solve(cell.T, (_numbers([row[2:] for row in rows]) - positions).T).T
    delta[:, atoms.pbc] -= np.rint(delta[:, atoms.pbc])
    if not np.allclose(delta @ cell, 0, rtol=0, atol=2e-6):
        raise ValueError("CP2K coordinates disagree with the reference")
    return volume


def read_cp2k_label(log: str, atoms: Atoms, *, exit_code: int) -> CP2KLabel:
    """Read one normally terminated ENERGY_FORCE result with W4-style tables.

    Forces follow input atom order. CP2K's printed GPa tensor uses the
    pressure-positive convention, so NEP virial is stress times volume.
    """
    run_types = re.findall(r"^\s*GLOBAL\| Run type[ \t]+(\S+)[ \t]*$", log, re.MULTILINE)
    if run_types != ["ENERGY_FORCE"]:
        raise ValueError("Expected a single ENERGY_FORCE run")
    if (exit_code != 0 or log.count("PROGRAM ENDED AT") != 1
            or "SCF run converged in" not in log
            or re.search(r"SCF run (?:NOT converged|did not converge)|\[ABORT\]", log, re.IGNORECASE)):
        raise ValueError("CP2K did not exit normally with converged SCF")

    volume = _check_geometry(log, atoms)
    energies = re.findall(
        rf"^\s*ENERGY\| Total FORCE_EVAL .*?energy \[(?:a\.u\.|hartree)\]:?[ \t]+({_NUMBER})[ \t]*$",
        log, re.MULTILINE,
    )
    if len(energies) != 1:
        raise ValueError("Expected one total FORCE_EVAL energy in Hartree")
    energy = float(_numbers([energies])[0, 0] * HARTREE_EV)

    block = _one_block(log, r"FORCES\| Atomic forces \[hartree/bohr\]", "force")
    force_table, end, _ = block.partition("FORCES| Sum")
    rows = re.findall(
        rf"^[ \t]*FORCES\|[ \t]+(\d+)[ \t]+{_VECTOR}[ \t]+{_NUMBER}[ \t]*$",
        force_table, re.MULTILINE,
    )
    if not end or [int(row[0]) for row in rows] != list(range(1, len(atoms) + 1)):
        raise ValueError("Missing, truncated or out-of-order CP2K force table")
    force = _numbers([row[1:] for row in rows]) * (HARTREE_EV / BOHR_ANGSTROM)

    block = _one_block(
        log, r"^[ \t]*(?:STRESS\|[ \t]*)?(?:Analytical|Numerical) stress tensor[ \t]*\[GPa\]",
        "stress",
    )
    block = block.split("Eigenvectors and eigenvalues", 1)[0]
    rows = re.findall(rf"^[ \t]*(?:STRESS\|[ \t]*)?([xyz])[ \t]+{_VECTOR}[ \t]*$",
                      block, re.MULTILINE)
    if [row[0] for row in rows] != list("xyz"):
        raise ValueError("Missing or out-of-order CP2K stress rows")
    stress = _numbers([row[1:] for row in rows])
    virial = stress * (volume / GPA_ANGSTROM3_PER_EV)
    if not np.allclose(stress, stress.T, rtol=0, atol=1e-3):
        raise ValueError("CP2K stress tensor is asymmetric")
    if not all(np.isfinite(values).all() for values in (energy, force, stress, virial)):
        raise ValueError("CP2K result contains nonfinite labels")
    return CP2KLabel(energy, force, stress, virial)
