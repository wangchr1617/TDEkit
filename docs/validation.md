# Engine validation

## 2026-11-11

| Engine | Actual execution | Checks |
| --- | --- | --- |
| LAMMPS 30 Sep 2026, macOS | 32 Ar atoms, four NVE steps, 0 and 1 eV | Two frames per case, sorted atom IDs, finite positions and velocities, initial PKA energy recovered from the dump |
| CP2K 2026.2 (`67b5da8`), macOS | Quickstep PBE water molecule, two 0.1 fs NVE steps, 0 and 1 eV | Three converged SCF calculations and three position and velocity frames per case, finite energy table, correct initial velocities and kinetic energy |
| GPUMD 5.3, cluster-alkemie | PBS `386042.mu01`, `gpu05`, 96 GeTe atoms, 0 and 1 eV | Normal GPUMD completion marker, reference and final frames, finite thermo and restart, expected initial recoil velocity, explicit OVITO container execution and a zero-pair summary for each case |

GPUMD executable SHA256
`15605ebbd78a2910269a992b0e1d7779634a5179fa8009e37c565346294592e6`.
The 25 output and receipt files and 13 staged input files matched their
SHA-256 manifests.

Native OVITO 3.17.1 on macOS independently produced byte-identical defect
summaries from both downloaded trajectories via the default TDEkit
interpreter. A two-frame fixture with a known single vacancy and
interstitial returned one Frenkel pair.

The flexible-ZBL example was exercised against a synthetic Te–Ge reference
using 30 training distances and four disjoint holdout distances. Held-out
RMSE was `4.58e-13 eV` for energy and `2.89e-14 eV/Å` for force, verifying
the numerical fit and export workflow on a known model.

These checks establish the documented software behavior under the stated
inputs. They do not establish convergence of a material's TDE or the
transferability of an interatomic potential.

## GPUMD check of the exported ZBL parameters

PBS job `386044.mu01` ran six fixed-geometry single points on `gpu05`
using the same GPUMD executable. The code compared the difference between
the original and candidate full potentials against the analytical ZBL
difference with the NEP backbone held fixed.

| Te–Ge distance (Å) | GPUMD energy difference (eV) | GPUMD outward force difference (eV/Å) |
| ---: | ---: | ---: |
| 0.8 | 199.19708252 | 694.40869141 |
| 1.5 | 10.26911068 | 63.92317200 |
| 2.2 | 0 | 0 |

Maximum absolute differences against the Python evaluator were
`2.39e-5 eV` and `4.54e-4 eV/Å`, below the respective test thresholds of
`0.003 eV` and `0.003 eV/Å`.

## Bounded TDE search on GPUMD

PBS job `386055.mu01` ran on `gpu05` with the same GPUMD executable and
the configured OVITO container. It reran the 0 and 1 eV smoke cases and
exercised `find_tde` through its exact 0.75 eV limit. Both explicit initial
states were evaluated at 0.5 and 0.75 eV. All four search runs had zero
Frenkel pairs and the search reported no crossing in that range.
The run used generated initial velocity fixtures and a 0.2 fs observation
window. Positive crossings and verification failure are covered by
regression tests with controlled defect probabilities.

The four case receipts, initial-state hashes, staged input hashes,
normal engine termination, two-frame trajectories and finite
thermodynamic output were independently checked after transfer.

## Unified single-recoil API

The same `run_recoil` entry point ran both zero-energy and 1 eV impulse
cases with the existing example inputs on 2026-11-11.

| Engine and environment | Validated trajectory |
| --- | --- |
| LAMMPS 30 Sep 2026 on macOS | 32 Ar atoms, 0 and 0.4 fs, explicit atom IDs, initial velocity and PKA kinetic energy. A separate rotated triclinic 27-atom case verified coordinate and velocity conversion back to the original basis. |
| CP2K 2026.2 on macOS | Quickstep PBE water, 0, 0.1 and 0.2 fs, three converged SCF calculations per case. Position and velocity frames agreed with the energy table and reproduced the initial 0 and 1 eV kinetic energies. |
| GPUMD 5.3 on cluster-alkemie | PBS `386059.mu01` on `gpu05`, 96 GeTe atoms, 0 and 0.2 fs. Initial PKA kinetic energy recovered from the dumped velocity was 0 and 0.99999990 eV. |

All 23 returned result and receipt files and 18 staged input files
matched their SHA-256 manifests. The transferred trajectories
passed a second read with the current library.
