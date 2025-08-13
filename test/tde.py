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
    nx=24, 
    ny=14, 
    nz=9, 
    thickness=10
)
prepath = Path('/home/changruiwang-ICME/Irradiation/train/GT').resolve()
run_in_relax = [
    f'potential {prepath}/0/nep.txt',
    'velocity 300',
    'time_step 1',
    'ensemble npt_scr 300 300 100 0 100 1000',
    'dump_thermo 100',
    'dump_restart 50000',
    'dump_exyz 100 0 0',
    'run 50000'
]      
run_in_cascade = [
    f'potential {prepath}/0/nep.txt',
    'velocity 300',
    'time_step 0',
    'ensemble nve',
    'dump_exyz 1',
    'run 1',
    'time_step 1 0.01',
    'ensemble heat_nhc 300 100 0 0 1',
    'compute 0 200 10 temperature',
    'dump_exyz 200 0 0',
    'run 30000'
]

relax_restart = Path("./relax/restart.xyz").resolve()
if not relax_restart.exists(): 
    relax_restart = ts.run_relax_simulation(
        input_file=Path("./model.xyz").resolve(), 
        run_in=run_in_relax
    )
tde_list = []
direction_table = generate_direction_table(4)
for d in direction_table:
    direction = np.array([d[0], d[1], d[2]])
    dirname = f"D_{d[0]}{d[1]}{d[2]}"
    workdir = os.path.join("./", dirname)
    os.makedirs(workdir, exist_ok=True)
    ts.set_work_dir(workdir)
    ts.set_direction(direction)    
    tde = ts.find_tde(input_file=relax_restart, 
                      index=None, 
                      symbol="Te", 
                      scaled_position=(0.5,0.5,0.5),
                      run_in=run_in_cascade)
    tde_list.append(tde)
    with open("tde_data.csv", 'a') as f:
        f.write(f"{','.join(map(str, d))},{tde:.4e}\n")
tde_array = np.array(tde_list)
print(np.mean(tde_array), flush=True) 

