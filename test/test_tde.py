from tdekit import TDESearch
import numpy as np
ts = TDESearch(init_energy=2, max_energy=1000, precision=0.01, nx=15, ny=9, nz=2, thickness=7, direction=np.array([0, 0, 1]))
ts.find_tde("./")
