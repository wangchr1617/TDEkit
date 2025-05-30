import os
import shutil
import subprocess
from pathlib import Path

base_dir = Path(__file__).parent
relax_path = base_dir / "relax" / "restart.xyz"
cascade_dir = base_dir / "cascade"
relax_script = "relax.py"
cascade_script = "cascade.py"
defect_script = "defects_analyzer.py"

init_energy = 2.0    
max_energy = 1000.0  
precision = 0.01     

if not relax_path.exists():
    print("restart.xyz not found. Running relax.py...")
    subprocess.run(["python", relax_script], check=True)
else:
    print("restart.xyz already exists")

def clean_cascade_dir():
    if cascade_dir.exists():
        print("Cleaning cascade directory...")
        shutil.rmtree(cascade_dir)

def update_energy(energy):
    with open(cascade_script, 'r') as f:
        lines = f.readlines()
    
    with open(cascade_script, 'w') as f:
        for line in lines:
            if line.strip().startswith("energy"):
                f.write(f"energy = {energy:.2f}  # eV\n")
            else:
                f.write(line)

def run_sim():
    subprocess.run(["python", cascade_script], check=True)
    
    apptainer_cmd = [
        "apptainer", 
        "exec", 
        "/opt/software/ovito-3.12.3/bin/ovito_python.sif",
        "python", 
        defect_script
    ]
    subprocess.run(apptainer_cmd, check=True)

def read_fp_count():
    summary_file = base_dir / "final_defect_summary.txt"
    with open(summary_file, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return 0
        return int(lines[-1].split("\t")[-1])

print("Starting exponential growth search...")
low_energy = init_energy
high_energy = None
current_energy = low_energy

while current_energy <= max_energy:
    print(f"Testing energy: {current_energy:.2f} eV")
    clean_cascade_dir()
    update_energy(current_energy)
    run_sim()
    fp_count = read_fp_count()
    print(f"Frenkel pairs: {fp_count}")
    
    if fp_count > 0:
        high_energy = current_energy
        break
    else:
        low_energy = current_energy
        current_energy *= 2

if high_energy is None:
    raise RuntimeError("Maximum energy limit exceeded")

print("Starting binary search...")
while abs(high_energy - low_energy) > precision:
    mid_energy = round((low_energy + high_energy) / 2, 2)
    print(f"Testing energy: {mid_energy:.2f} eV")
    
    clean_cascade_dir()
    update_energy(mid_energy)
    run_sim()
    fp_count = read_fp_count()
    print(f"Frenkel pairs: {fp_count}")

    if fp_count > 0:
        high_energy = mid_energy
    else:
        low_energy = mid_energy

print("\nSearch completed!")
print(f"TDE: {high_energy:.2f} eV")

clean_cascade_dir()