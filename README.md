# TDEkit
TDEkit is a Python toolkit for rapid calculation of threshold displacement energy (TDE)​​.

## Usage

Prepare the structural file `model.xyz` and potential function file `nep.txt` in the current directory, 
then simply run `python test.py`.

```shell
.
├── model.xyz
├── nep.txt
└── test.py
```

The contents of `test.py` are as follows:
```python
from tdekit import find_tde
find_tde("./")
```
