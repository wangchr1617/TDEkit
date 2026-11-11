"""Prepare and run CP2K molecular dynamics from an ASE recoil state."""

import re
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms, units

from .core import PathLike, resolve_executable, write_run

STATE_INCLUDE = '@INCLUDE "tdekit-state.inc"'


def _format_vector(vector) -> str:
    return " ".join(f"{value:.16e}" for value in vector)


def write_cp2k_state(atoms: Atoms, filename: PathLike) -> None:
    """Write CELL, COORD and VELOCITY sections for inclusion under SUBSYS."""
    velocities = np.asarray(atoms.info.get("velocities"), dtype=float)
    if not len(atoms) or velocities.shape != (len(atoms), 3) or not np.isfinite(velocities).all():
        raise ValueError("Expected finite velocities in atoms.info['velocities'] (Å/fs)")
    if atoms.cell.rank != 3 or not np.isfinite(atoms.cell.array).all():
        raise ValueError("CP2K requires a finite three-dimensional cell")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("CP2K requires finite atomic coordinates")

    periodic = "".join(axis for axis, flag in zip("XYZ", atoms.pbc) if flag) or "NONE"
    lines = ["&CELL", f"  PERIODIC {periodic}"]
    for axis, vector in zip("ABC", atoms.cell):
        lines.append(f"  {axis} [angstrom] {_format_vector(vector)}")
    lines.extend(["&END CELL", "&COORD", "  UNIT angstrom"])
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(f"  {symbol} {_format_vector(position)}")
    lines.extend(["&END COORD", "&VELOCITY"])
    velocities_au = velocities * (units.AUT / units.fs) / units.Bohr
    lines.extend(f"  {_format_vector(velocity)}" for velocity in velocities_au)
    lines.append("&END VELOCITY")
    write_run(lines, filename)


def run_cp2k(
    atoms: Atoms,
    dirname: PathLike,
    input_text: str,
    *,
    executable: PathLike = "cp2k.psmp",
) -> None:
    """Run a user-supplied CP2K input containing the TDEkit SUBSYS include."""
    if [line.strip() for line in input_text.splitlines()].count(STATE_INCLUDE) != 1:
        raise ValueError(f"Include {STATE_INCLUDE} once inside FORCE_EVAL / SUBSYS")
    engine = resolve_executable(executable)
    directory = Path(dirname)
    directory.mkdir(parents=True, exist_ok=False)
    write_cp2k_state(atoms, directory / "tdekit-state.inc")
    (directory / "recoil.inp").write_text(input_text, encoding="utf-8")
    log_path = directory / "cp2k.out"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run([engine, "-i", "recoil.inp"], cwd=directory, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    with log_path.open(encoding="utf-8", errors="replace") as log:
        completed = False
        for line in log:
            if re.search(r"SCF run NOT converged|SCF run did not converge", line, re.IGNORECASE):
                raise RuntimeError(f"CP2K SCF failed to converge: {log_path}")
            if "PROGRAM ENDED AT" in line:
                completed = True
    if not completed:
        raise RuntimeError(f"Missing CP2K termination marker: {log_path}")
