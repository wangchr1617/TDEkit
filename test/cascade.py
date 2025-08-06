from pathlib import Path
from tdekit import TDESearch, generate_direction_table
import numpy as np
import os
import shutil

ts = TDESearch(
    base_dir="./",
    nx=4, 
    ny=3, 
    nz=2, 
    thickness=4
)
prepath = Path('./potentials').resolve()
run_in_relax = [
    f'potential {prepath}/0.txt',
    'velocity 300',
    'time_step 1',
    'ensemble npt_scr 300 300 100 0 100 1000',
    'dump_thermo 100',
    'dump_restart 50000',
    'dump_exyz 100 0 0',
    'run 50000'
]      
run_in_cascade = [
    f'potential {prepath}/0.txt',
    f'potential {prepath}/1.txt',
    f'potential {prepath}/2.txt',
    f'potential {prepath}/3.txt',
    'velocity 300',
    'time_step 0',
    'ensemble nve',
    'dump_exyz 1',
    'run 1',
    'time_step 1 0.01',
    'ensemble heat_nhc 300 100 0 0 1',
    'compute 0 200 10 temperature',
    'dump_exyz 200 0 0',
    'active 200 0 1 0.05',
    'run 10000'
]

relax_restart = Path("./relax/restart.xyz").resolve()
if not relax_restart.exists(): 
    relax_restart = ts.run_relax_simulation(
        input_file=Path("./xyzs/model.xyz").resolve(), 
        run_in = run_in_relax
    )
tde_list = []
direction_table = generate_direction_table(1)
for d in direction_table:
    direction = np.array([d[0], d[1], d[2]])
    dirname = f"D_{d[0]}{d[1]}{d[2]}"
    workdir = os.path.join("./", dirname)
    os.makedirs(workdir, exist_ok=True)
    ts.set_work_dir(workdir)
    ts.set_direction(direction)    
    tde = ts.run_cascade_simulation(
        energy=20, 
        input_file=relax_restart, 
        index=None, 
        symbol="Ge", 
        scaled_position=(0.5,0.5,0.5), 
        run_in=run_in_cascade
    )
    tde_list.append(tde)
tde_array = np.array(tde_list)
print(np.mean(tde_array), flush=True) 

