from pathlib import Path
from tdekit import TDESearch, generate_direction_table, generate_uniform_direction_table, save_direction_table_to_csv, read_direction_table_from_csv
import numpy as np
import os
import shutil

ts = TDESearch(
    base_dir="./",
    init_energy=2,
    max_energy=1000,
    precision=0.1
)

prepath = Path('/home/changruiwang-ICME/Irradiation/train/GT').resolve()

run_in_relax = [
    f'potential {prepath}/0/nep.txt',
    'velocity 300',
    'time_step 1',
    'ensemble npt_scr 300 300 100 0 100 1000',
    'dump_thermo 100',
    'dump_restart 30000',
    'run 30000'
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
    'dump_exyz 10000 0 0',
    'run 10000'
]

relax_restart = Path("./relax/restart.xyz").resolve()
if not relax_restart.exists(): 
    relax_restart = ts.run_relax_simulation(
        input_file=Path("./model.xyz").resolve(), 
        run_in=run_in_relax,
        nx=12,
        ny=7,
        nz=5,
        thickness=3
    )

tde_list = []
csv_filename = "uniform_directions.csv"

if os.path.exists(csv_filename):
    direction_table = read_direction_table_from_csv(csv_filename)
    print(f"Loaded {len(direction_table)} directions from CSV")
else:
    t1 = generate_direction_table(hmax=5)
    t2 = generate_uniform_direction_table(n=500)
    
    combined_table = t1 + t2
    combined_table.sort(key=lambda x: (x[3], x[4]))
    
    save_direction_table_to_csv(combined_table, csv_filename)
    direction_table = read_direction_table_from_csv(csv_filename)
    print(f"Generated and saved {len(direction_table)} directions to CSV")
    
for idx, d in enumerate(direction_table, start=1):
    direction = np.array([d[0], d[1], d[2]])
    dirname = f"ID_{idx}"
    workdir = Path("./") / dirname
    workdir.mkdir(exist_ok=True)
    ts.set_work_dir(str(workdir))
    ts.set_direction(direction)    
    tde, prob = ts.find_tde(input_file=relax_restart, 
                            symbol="Ge",
                            random_select=True,
                            center_pka=True, 
                            num_samples=10, 
                            verification_samples=10, 
                            threshold_prob=0.5,
                            run_in=run_in_cascade)
    tde_list.append(tde)
    with open("tde_data.csv", 'a') as f:
        f.write(f"{','.join(map(str, d))},{tde:.2f},{prob:.2f}\n")
tde_array = np.array(tde_list)
print(np.mean(tde_array), flush=True)

