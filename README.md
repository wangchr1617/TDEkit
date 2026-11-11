# TDEkit

Python library for primary knock-on atom (PKA) simulation, directional threshold
displacement energy (TDE) workflows and configuration selection for
neuroevolution potentials (NEP).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[fit]"
python -m unittest discover -s tests -v
```

ASE and NumPy are sufficient for the core API. Install `.[analysis]` in the
Python environment that runs OVITO, and `.[selection]` for NEP descriptor and
force analysis. Add `.[umap]` to use UMAP. MD engines and potential files are
provided by the user.

- [Single-recoil API and trajectory acceptance](docs/single-recoil.md)
- [Recoil protocols and OVITO setup](docs/recoil-workflow.md)
- [CP2K label acceptance and NEP dataset export](docs/cp2k-nep-dataset.md)
- [Flexible NEP–ZBL fitting and comparison](docs/nep-zbl.md)
- [Engine validation results](docs/validation.md)
- [Related projects and development scope](docs/development.md)
- `tdekit/` contains the library, `tests/` the regression tests, and
  `examples/` small engine validation cases.
- `test/` and the root `tdesearch.py` contain historical scripts and small
  validation fixtures. The legacy descriptor examples require local files
  at `test/xyzs/train.xyz` and `test/xyzs/test.xyz`.

Training datasets and generated run outputs are stored outside version control.

The source code is distributed under the [MIT license](LICENSE).
