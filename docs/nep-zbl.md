# Flexible NEP–ZBL fitting and comparison

Install the optional SciPy dependency with `python -m pip install ".[fit]"`.
The library reads legacy flexible ZBL models with a `nep3_zbl`, `nep4_zbl`
or `nep5_zbl` header and `zbl 0 0`. It verifies the scalar count against
the legacy header and returns pair parameters in the model's species order.
The pair evaluator returns the cosine-switched ZBL contribution in eV and
the outward radial force in eV/Å.

```python
from tdekit.zbl import read_flexible_zbl

first = read_flexible_zbl("nep_a.txt")
second = read_flexible_zbl("nep_b.txt")
a = next(key for key in first if sorted(key) == ["Ge", "Te"])
b = next(key for key in second if sorted(key) == ["Ge", "Te"])
distances = [0.5, 0.8, 1.0, 1.5, 2.0]
energy_a, force_a = first[a].evaluate(a, distances)
energy_b, force_b = second[b].evaluate(b, distances)
```

Prepare a CSV with `split,distance_angstrom,target_energy_ev,target_force_ev_per_angstrom`.
Use `train` and `holdout` labels with disjoint distances. The target values
represent the desired ZBL contribution. For a fixed NEP backbone, subtract its
energy and radial force from the reference, using the same energy zero and
force sign throughout. Absolute dimer total energies require an explicit
reference before subtraction.

```bash
python examples/fit_nep_zbl.py nep.txt pair.csv fit-ge-te \
  --pair Ge Te --energy-scale 10 --force-scale 100
```

The fit keeps both cutoffs and all four decay rates fixed, and estimates four
nonnegative amplitudes by least squares. The scales divide the energy and
force residuals in their respective units. They should reflect the data and
relative weighting desired. The exported candidate preserves the original
NEP backbone and all unfitted ZBL pairs numerically. The script records input
and output SHA256 hashes, fitted parameters, and train and holdout energy and
force RMSE. The holdout is evaluated after fitting.

The supplied regression tests cover analytical forces, switching boundaries,
parameter recovery on synthetic data, preservation of original negative
amplitudes when loading, and model export. A fitted ZBL contribution and a
full NEP+ZBL prediction answer different questions. Evaluate the exported
candidate with the intended engine and independent total energies and forces
before adopting it for irradiation calculations. See the
[GPUMD flexible ZBL parameter definition](https://gpumd.org/nep/input_parameters/zbl.html).
