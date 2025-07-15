from pathlib import Path
from tdekit import TDESearch, generate_direction_table
import numpy as np
import os
import shutil

ts = TDESearch(
    base_dir="./",
    init_energy=2, 
    max_energy=1000, 
    precision=0.01, 
    nx=15, 
    ny=9, 
    nz=2, 
    thickness=7
)
relax_restart = Path("./relax/restart.xyz")
if not relax_restart.exists(): 
    relax_restart = ts.run_relax_simulation()

direction_table = generate_direction_table(5)
tde_list = []
for d in direction_table:
    direction = np.array([d[0], d[1], d[2]])
    dirname = f"D_{d[0]}{d[1]}{d[2]}"
    workdir = os.path.join("./", dirname)
    os.makedirs(workdir, exist_ok=True)
    ts.set_base_dir(workdir)
    ts.set_direction(direction)    
    shutil.copy("./model.xyz", os.path.join(workdir, "model.xyz"))
    shutil.copy("./0.txt", os.path.join(workdir, "nep.txt"))
    tde = ts.find_tde(input_file=relax_restart, index=None, symbol="Ge", scaled_position=(0.5, 0.5 ,0.5))
    tde_list.append(tde)
tde_array = np.array(tde_list)
print(np.mean(tde_array), flush=True) 
