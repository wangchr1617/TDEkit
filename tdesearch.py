from pathlib import Path
from tdekit import TDESearch, generate_direction_table, generate_uniform_direction_table, save_direction_table_to_csv, read_direction_table_from_csv
import numpy as np
import os
import shutil

ts = TDESearch()

nep_file = Path('/home/changruiwang-ICME/Irradiation/train/GT/0/nep.txt').resolve()

run_in_relax = [
    f'potential {nep_file}',
    'velocity 300',
    'time_step 1',
    'ensemble npt_scr 300 300 100 0 100 1000',
    'dump_thermo 100',
    'dump_restart 30000',
    'run 30000'
]      

run_in_cascade_1 = [
    f'potential {nep_file}',
    'velocity 300',
    'time_step 0',
    'ensemble nve',
    'dump_exyz 1',
    'run 1',
    'time_step 3 0.01',
    'ensemble heat_nhc 300 100 0 0 1',
    'dump_exyz 500 0 0',
    'run 500'
]

run_in_cascade_2 = [
    f'potential {nep_file}',
    'velocity 300',
    'time_step 0',
    'ensemble nve',
    'dump_exyz 1',
    'run 1',
    'time_step 3 0.01',
    'ensemble heat_nhc 300 100 0 0 1',
    'dump_exyz 2000 0 0',
    'run 2000'
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

def find_tde(**kwargs):
    print("\n" + "="*60)
    print("STARTING TDE SEARCH")
    print("="*60)
    
    step = 5
    precision = 1
    threshold_prob = 0.5
    
    no_damage_energy, damage_energy = ts.exponential_growth_search(init_energy, max_energy, run_in=run_in_cascade_1, **kwargs)
    
    energy, prob = ts.incremental_search(
        no_damage_energy / 2, damage_energy * 2, 
        step=step, num_samples=3,
        threshold_prob=threshold_prob,
        run_in=run_in_cascade_1,
        **kwargs
    )
    
    tde_estimate, low_energy, low_prob, high_energy, high_prob = self.binary_search(
            energy - 5, energy + 5, 
            precision=precision, 
            num_samples=10,
            threshold_prob=threshold_prob,
            run_in=run_in_cascade_2,
            **kwargs
        )
        
    verified_probability = ts.get_fp_probability(tde_estimate, verification_samples, **kwargs)
    
    if verified_probability < threshold_prob:
        tde_estimate = high_energy
        verified_probability = high_prob
    
    print(f"\nVerification results at {tde_estimate:.3f}eV:")
    print(f"Damage probability: {verified_probability:.2f}")
    
    print("\n" + "="*60)
    print("TDE SEARCH COMPLETE")
    print("="*60)
    print(f"TDE along <{ts.direction}> direction: {tde_estimate:.3f} ± {precision:.3f} eV")
    print(f"At this energy, FP formation probability is {verified_probability:.2f}")
    print("="*60)
    
    return tde_estimate, verified_probability    
    
for idx, d in enumerate(direction_table, start=1):
    direction = np.array([d[0], d[1], d[2]])
    dirname = f"ID_{idx}"
    
    workdir = Path("./") / dirname
    if workdir.exists() and workdir.is_dir():
        print(f"Directory {dirname} already exists, skipping...")
        continue
    workdir.mkdir(exist_ok=True)
    
    ts.set_work_dir(str(workdir))
    ts.set_direction(direction)
    
    tde, prob = find_tde(input_file=relax_restart, 
                            symbol="Ge",
                            random_select=True,
                            center_pka=True)
    tde_list.append(tde)
    
    result_file = workdir / "tde_data.csv"
    with open(result_file, 'w') as f:
        f.write(f"{','.join(map(str, d))},{tde:.2f},{prob:.2f}\n")
