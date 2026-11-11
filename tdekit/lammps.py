"""Input and execution for atomic LAMMPS cases in metal units."""

import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms, units
from ase.io import write

from .core import PathLike, resolve_executable, write_run


def run_lammps(
    atoms: Atoms,
    dirname: PathLike,
    commands: Iterable[str],
    *,
    executable: PathLike = "lmp",
) -> None:
    """Write an ASE structure and run commands after reading the data file.

    Cartesian velocities in atoms.info["velocities"] (Å/fs), if present,
    take precedence over ASE momenta. LAMMPS atom IDs start at one and
    follow the ASE atom order. Element types follow alphabetical order.
    """
    engine = resolve_executable(executable)
    if not len(atoms) or atoms.cell.rank != 3:
        raise ValueError("LAMMPS requires atoms and a three-dimensional cell")

    model = atoms.copy()
    atomic_masses = model.get_masses()
    if "velocities" in atoms.info:
        velocities = np.asarray(atoms.info["velocities"], dtype=float)
        if velocities.shape != (len(atoms), 3) or not np.isfinite(velocities).all():
            raise ValueError("Expected finite velocities with shape (n_atoms, 3)")
        model.set_momenta(velocities * atomic_masses[:, None] / units.fs, apply_constraint=False)

    if not np.isfinite(atomic_masses).all() or np.any(atomic_masses <= 0):
        raise ValueError("LAMMPS requires positive finite masses")
    for symbol in set(model.get_chemical_symbols()):
        masses = atomic_masses[model.symbols == symbol]
        if not np.all(masses == masses[0]):
            raise ValueError(f"LAMMPS atomic style requires one mass per type: {symbol}")

    directory = Path(dirname)
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "model.data", model, format="lammps-data", units="metal",
          atom_style="atomic", masses=True, velocities=model.has("momenta"),
          force_skew=True)
    boundary = " ".join("p" if periodic else "f" for periodic in model.pbc)
    write_run(["units metal", "atom_style atomic", f"boundary {boundary}",
               "read_data model.data", *commands], directory / "in.lammps")
    with (directory / "lammps.out").open("w", encoding="utf-8") as log:
        subprocess.run([engine, "-in", "in.lammps"], cwd=directory, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
