"""Trajectory readers for the single-recoil backend contracts."""

from dataclasses import dataclass
from io import StringIO
from itertools import zip_longest
from pathlib import Path
import re

import numpy as np
from ase import Atoms, units
from ase.calculators.lammps import Prism
from ase.io import iread, read


@dataclass(frozen=True)
class RecoilFrame:
    """An ASE structure and explicit atom IDs, velocities and physical time."""

    time_fs: float
    atoms: Atoms
    atom_ids: np.ndarray
    velocities_angstrom_fs: np.ndarray


def read_gpumd_frames(path: Path):
    for atoms in iread(path, index=":", format="extxyz"):
        try:
            time = float(atoms.info["Time"])
            velocities = atoms.arrays["vel"]
            energy = atoms.get_potential_energy()
        except (KeyError, RuntimeError) as error:
            raise ValueError("GPUMD trajectory requires Time, velocities and potential energy") from error
        if not np.isfinite(energy):
            raise ValueError("Nonfinite GPUMD potential energy")
        yield RecoilFrame(time, atoms, np.arange(1, len(atoms) + 1), velocities)


def _expect(stream, header):
    if stream.readline().strip() != header:
        raise ValueError(f"Expected {header!r} in LAMMPS dump")


def read_lammps_frames(path: Path, reference: Atoms):
    species = sorted(set(reference.get_chemical_symbols()))
    prism = Prism(reference.cell.array, pbc=reference.pbc)
    with path.open(encoding="utf-8") as stream:
        _expect(stream, "ITEM: UNITS")
        _expect(stream, "metal")
        previous_step = -1
        for marker in stream:
            if marker.strip() != "ITEM: TIME":
                raise ValueError("LAMMPS dump requires time yes on every frame")
            time_fs = 1000 * float(stream.readline())
            _expect(stream, "ITEM: TIMESTEP")
            step = int(stream.readline())
            if step <= previous_step:
                raise ValueError("LAMMPS steps must be strictly increasing")
            previous_step = step
            _expect(stream, "ITEM: NUMBER OF ATOMS")
            count = int(stream.readline())
            if count != len(reference):
                raise ValueError("LAMMPS atom count differs from the input")
            box = [stream.readline() for _ in range(4)]
            if not box[0].startswith("ITEM: BOX BOUNDS") or not all(box[1:]):
                raise ValueError("Missing LAMMPS box bounds")
            _expect(stream, "ITEM: ATOMS id type x y z vx vy vz")
            rows = [stream.readline() for _ in range(count)]
            columns = [row.split() for row in rows]
            if any(len(row) != 8 for row in columns):
                raise ValueError("Truncated or malformed LAMMPS atom rows")
            ids = np.array([int(row[0]) for row in columns])
            if sorted(ids) != list(range(1, count + 1)):
                raise ValueError("LAMMPS atom IDs are missing or duplicated")
            if any(not 1 <= int(row[1]) <= len(species) for row in columns):
                raise ValueError("LAMMPS atom type is outside the input species map")

            # ASE handles LAMMPS bounds, tilts and rotation back to the input
            # basis. Its standard dump reader sorts by ID and discards the IDs.
            dump = StringIO(
                f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{count}\n"
                + "".join(box) + "ITEM: ATOMS id type x y z vx vy vz\n" + "".join(rows)
            )
            atoms = read(dump, format="lammps-dump-text", specorder=species, prismobj=prism)
            yield RecoilFrame(time_fs, atoms, np.sort(ids), atoms.get_velocities() * units.fs)


def read_cp2k_frames(directory: Path, project: str, reference: Atoms):
    # CP2K's XYZ output omits the cell. The backend only accepts fixed-cell MD.
    position_file = directory / f"{project}-pos-1.xyz"
    velocity_file = directory / f"{project}-vel-1.xyz"
    thermo = np.loadtxt(directory / f"{project}-1.ener", ndmin=2)
    if (thermo.shape[1] != 7 or not np.isfinite(thermo).all()
            or np.any(np.diff(thermo[:, :2], axis=0) <= 0)
            or np.any(thermo[:, 0] != thermo[:, 0].astype(int))):
        raise ValueError("Invalid CP2K energy/time table")
    thermo_by_step = {row[0]: row for row in thermo}
    masses = reference.get_masses()[:, None]
    pos_frames = iread(position_file, index=":", format="extxyz")
    vel_frames = iread(velocity_file, index=":", format="extxyz")
    final_step = None
    for pos, vel in zip_longest(pos_frames, vel_frames):
        if pos is None or vel is None:
            raise ValueError("CP2K position and velocity frame counts differ")
        step, time = pos.info.get("i"), pos.info.get("time")
        if (step not in thermo_by_step or time is None or not np.isfinite(time)
                or abs(thermo_by_step[step][1] - time) > 5.01e-4
                or step != vel.info.get("i") or time != vel.info.get("time")
                or not np.array_equal(pos.numbers, vel.numbers)):
            raise ValueError("CP2K trajectory and energy/velocity records disagree")
        if final_step is not None and step <= final_step:
            raise ValueError("CP2K trajectory steps must be strictly increasing")
        final_step = step
        pos.set_cell(reference.cell)
        pos.set_pbc(reference.pbc)
        velocity = vel.positions * units.Bohr * units.fs / units.AUT
        if velocity.shape != (len(reference), 3):
            raise ValueError("CP2K atom count differs from the input")
        kinetic = 0.5 * np.sum(masses * (velocity / units.fs)**2)
        reported = thermo_by_step[step][2] * units.Hartree
        if not np.isclose(kinetic, reported, rtol=2e-6, atol=1e-6):
            raise ValueError("CP2K kinetic energy disagrees with velocities and input masses")
        yield RecoilFrame(float(thermo_by_step[step][1]), pos, np.arange(1, len(pos) + 1), velocity)
    if final_step != thermo[-1, 0]:
        raise ValueError("CP2K trajectory does not include the final MD step")


def check_cp2k_md_input(input_text):
    # These checks scope the reader to the validated Quickstep fixed-cell XYZ
    # contract. CP2K remains responsible for validating the full input syntax.
    lines = [re.split(r"[!#]", line, maxsplit=1)[0].split() for line in input_text.splitlines()]
    for keyword, allowed in (("RUN_TYPE", {"MD"}), ("ENSEMBLE", {"NVE", "NVT"})):
        settings = [line[1:] for line in lines if line and line[0].upper() == keyword]
        if len(settings) != 1 or len(settings[0]) != 1 or settings[0][0].upper() not in allowed:
            raise ValueError("CP2K recoil requires explicit RUN_TYPE MD and fixed-cell NVE or NVT")
    # User-specified units in PRINT/VELOCITIES can otherwise silently change a
    # zero-energy trajectory, for which the initial velocity check is blind.
    velocity_print = False
    for line in lines:
        if not line:
            continue
        key = line[0].upper()
        if key == "&VELOCITIES":
            velocity_print = True
        elif key == "&END" and len(line) > 1 and line[1].upper() == "VELOCITIES":
            velocity_print = False
        elif velocity_print and key == "UNIT":
            raise ValueError("CP2K velocity output must use default atomic units")
