# Development scope

TDEkit uses ASE structures and separates the recoil protocol from engine-specific
units and input. It keeps engine inputs and outputs in individual case directories.
Validation starts with a zero-energy control and a defined nonzero recoil, checks
atom identity and initial velocities, and then checks finite energies and
trajectory completeness. These cases verify software behavior for the given
engines and potentials.

| Reference | Practice relevant to TDEkit |
| --- | --- |
| [GPUMD-Wizard](https://github.com/Jonsnow-willow/GPUMD-Wizard), AGPL-3.0 | Use ASE structures, keep worked examples separately from implementation and let users provide their own engine. |
| [ezcascades](https://github.com/mb4512/ezcascades), CC0-1.0 | Save explicit case parameters and provide separate workflows for displacement thresholds and multicascade simulations. |
| [irradiapy](https://github.com/acgc99/irradiapy), MIT | Record provenance for cascade data and keep post-processing independent of execution. |

The project links these references for workflow design. This work uses TDEkit's
own implementation and retains its MIT license.

## Development

See the [validation record](validation.md) for the executed LAMMPS, CP2K
and GPUMD cases. The [NEP–ZBL workflow](nep-zbl.md) describes the reference
and holdout inputs for fitting amplitudes with a fixed NEP backbone.

For longer recoil studies, preserve the pristine reference and the actual
initial state, document independent thermal repeats, and verify a
material-appropriate integration step and observation window. Project
potentials and production PBS configuration remain site inputs.
