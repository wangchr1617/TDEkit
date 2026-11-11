"""Engine-independent single-recoil execution and trajectory acceptance."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np
from ase import Atoms, units

from ._recoil_io import (RecoilFrame, check_cp2k_md_input, read_cp2k_frames,
                         read_gpumd_frames, read_lammps_frames)
from .core import PathLike, run_gpumd, set_pka
from .cp2k import run_cp2k
from .lammps import run_lammps

__all__ = ["RecoilEvent", "RecoilFrame", "RecoilResult", "RecoilBackend",
           "GPUMDBackend", "LAMMPSBackend", "CP2KBackend", "run_recoil"]


@dataclass(frozen=True)
class RecoilEvent:
    """Input atom ID (1-based), energy (eV), and Cartesian direction."""

    atom_id: int
    energy_ev: float
    direction: Sequence[float]
    mode: str = "impulse"


@dataclass(frozen=True)
class RecoilResult:
    """A validated trajectory, energy accounting, and its case directory."""

    directory: Path
    engine: str
    atom_identity: str
    frames: tuple[RecoilFrame, ...]
    velocity_increment_energy_ev: float
    kinetic_energy_change_ev: float


class RecoilBackend(Protocol):
    """Backend supplying engine execution and normalized trajectory frames."""

    name: str
    identity: str

    def run(self, atoms: Atoms, directory: Path) -> Iterable[RecoilFrame]:
        ...


def _require_footer(directory, name, marker):
    log = (directory / name).read_text(encoding="utf-8", errors="replace")
    if marker not in log:
        raise ValueError(f"Missing normal termination marker in {directory / name}")
    return log


@dataclass(frozen=True)
class GPUMDBackend:
    commands: Sequence[str]
    nep_path: Optional[PathLike] = None
    executable: PathLike = "gpumd"
    name: str = field(default="gpumd", init=False)
    identity: str = field(default="output_order", init=False)

    def run(self, atoms, directory):
        run_gpumd(atoms, directory, list(self.commands), self.nep_path, executable=self.executable)
        _require_footer(directory, "gpumd.out", "Finished running GPUMD.")
        return read_gpumd_frames(directory / "dump.xyz")


@dataclass(frozen=True)
class LAMMPSBackend:
    commands: Sequence[str]
    executable: PathLike = "lmp"
    name: str = field(default="lammps", init=False)
    identity: str = field(default="explicit_id", init=False)

    def run(self, atoms, directory):
        run_lammps(atoms, directory, self.commands, executable=self.executable)
        _require_footer(directory, "lammps.out", "Total wall time:")
        return read_lammps_frames(directory / "trajectory.dump", atoms)


@dataclass(frozen=True)
class CP2KBackend:
    input_text: str
    project: str = "recoil"
    executable: PathLike = "cp2k.psmp"
    name: str = field(default="cp2k", init=False)
    identity: str = field(default="output_order", init=False)

    def run(self, atoms, directory):
        if not self.project or Path(self.project).name != self.project or self.project == "..":
            raise ValueError("Expected a CP2K project name without path components")
        check_cp2k_md_input(self.input_text)
        run_cp2k(atoms, directory, self.input_text, executable=self.executable)
        log = _require_footer(directory, "cp2k.out", "PROGRAM ENDED AT")
        if "SCF run converged" not in log:
            raise ValueError("No converged CP2K SCF record")
        return read_cp2k_frames(directory, self.project, atoms)


def _accept_frames(frames, expected_times, state, expected_velocities):
    ids = np.arange(1, len(state) + 1)
    times = np.asarray([frame.time_fs for frame in frames])
    if (times.shape != expected_times.shape or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or not np.allclose(times, expected_times, rtol=0, atol=1e-5)):
        raise ValueError(f"Trajectory clock or frame count differs: expected {expected_times}, got {times}")
    for index, frame in enumerate(frames):
        atoms = frame.atoms
        velocities = np.asarray(frame.velocities_angstrom_fs)
        if (not np.array_equal(frame.atom_ids, ids)
                or not np.array_equal(atoms.numbers, state.numbers)
                or not np.array_equal(atoms.pbc, state.pbc)):
            raise ValueError(f"Atom IDs, species or boundary flags disagree in frame {index}")
        if (velocities.shape != (len(state), 3) or not np.isfinite(velocities).all()
                or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all()
                or np.linalg.det(atoms.cell.array) <= 0):
            raise ValueError(f"Invalid positions, velocities or cell in frame {index}")
    initial = frames[0]
    if not np.allclose(initial.atoms.cell, state.cell, rtol=0, atol=2e-6):
        raise ValueError("Initial trajectory cell differs from the input")
    delta = np.linalg.solve(state.cell.array.T, (initial.atoms.positions - state.positions).T).T
    delta[:, state.pbc] -= np.rint(delta[:, state.pbc])
    if not np.allclose(delta @ state.cell.array, 0, rtol=0, atol=2e-6):
        raise ValueError("Initial trajectory positions differ from the input")
    if not np.allclose(initial.velocities_angstrom_fs, expected_velocities, rtol=0, atol=2e-7):
        raise ValueError("Initial trajectory velocities differ from the prescribed recoil")


def run_recoil(atoms: Atoms, event: RecoilEvent, backend: RecoilBackend,
               directory: PathLike, *, sample_times_fs: Sequence[float]) -> RecoilResult:
    """Apply one recoil, run an engine and accept its sampled trajectory.

    The declared time grid starts at zero and is compared to engine output.
    Use ASE momenta or explicitly provide velocities in Å/fs through
    atoms.info["velocities"]. Input structures are left untouched.
    """
    target = Path(directory).resolve()
    if target.exists():
        raise FileExistsError(target)
    times = np.asarray(sample_times_fs, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Expected strictly increasing sample times starting at zero")
    if (not isinstance(event.atom_id, (int, np.integer)) or isinstance(event.atom_id, bool)
            or not 1 <= event.atom_id <= len(atoms)):
        raise ValueError("atom_id must be a 1-based input atom ID")
    if (not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all()
            or np.linalg.det(atoms.cell.array) <= 0):
        raise ValueError("Expected finite positions and a right-handed three-dimensional cell")

    state = atoms.copy()
    initial = state.info.get("velocities")
    if initial is None and state.has("momenta"):
        initial = state.get_velocities() * units.fs
    state.info["velocities"] = np.asarray(initial, dtype=float).copy()
    direction = np.asarray(event.direction, dtype=float)
    set_pka(state, event.energy_ev, direction, index=int(event.atom_id) - 1, mode=event.mode)
    expected_velocities = state.info["velocities"].copy()
    before = np.asarray(initial) / units.fs
    after = expected_velocities / units.fs
    masses = state.get_masses()[:, None]
    index = event.atom_id - 1
    increment = float(0.5 * masses[index, 0] * np.sum((after[index] - before[index])**2))
    change = float(0.5 * np.sum(masses * (after - before) * (after + before)))

    frames = tuple(backend.run(state.copy(), target))
    _accept_frames(frames, times, state, expected_velocities)
    receipt = {
        "status": "validated", "engine": backend.name, "atom_identity": backend.identity,
        "atom_id": int(event.atom_id), "element": state[index].symbol,
        "mode": event.mode, "requested_energy_ev": float(event.energy_ev),
        "direction_cartesian": (direction / np.linalg.norm(direction)).tolist(),
        "velocity_increment_energy_ev": increment, "kinetic_energy_change_ev": change,
        "sample_times_fs": [float(frame.time_fs) for frame in frames],
        "natoms": len(state),
    }
    payload = json.dumps(receipt, allow_nan=False, indent=2) + "\n"
    with (target / "recoil.json").open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return RecoilResult(target, backend.name, backend.identity, frames, increment, change)
