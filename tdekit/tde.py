from itertools import combinations_with_replacement, product
from math import gcd
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .core import run_cascade, run_relax

def generate_crystal_directions(hmax):
    directions_set = set()
    indices = list(range(-hmax, hmax+1))
    for h, k, l in product(indices, indices, indices):
        if h == 0 and k == 0 and l == 0:
            continue
        g = gcd(gcd(abs(h), abs(k)), abs(l))
        if g == 0: 
            continue
        h_norm = h // g
        k_norm = k // g
        l_norm = l // g
        if l_norm < 0:
            h_norm, k_norm, l_norm = -h_norm, -k_norm, -l_norm
        elif l_norm == 0 and k_norm < 0:
            h_norm, k_norm, l_norm = -h_norm, -k_norm, -l_norm
        elif l_norm == 0 and k_norm == 0 and h_norm < 0:
            h_norm, k_norm, l_norm = -h_norm, -k_norm, -l_norm
        directions_set.add((h_norm, k_norm, l_norm))
    return sorted(directions_set)

def calculate_angles(h, k, l):
    r = np.sqrt(h**2 + k**2 + l**2)
    theta_rad = np.arccos(l / r)
    theta_deg = np.degrees(theta_rad)
    phi_rad = np.arctan2(k, h)
    phi_deg = np.degrees(phi_rad)
    if phi_deg < 0:
        phi_deg += 360
    return theta_deg, phi_deg

def generate_direction_table(hmax):
    directions = generate_crystal_directions(hmax)
    table = []
    for d in directions:
        h, k, l = d
        if h == 0 and k == 0 and l == 0:
            continue
        theta, phi = calculate_angles(h, k, l)
        if not (0 <= theta <= 90) or not (0 <= phi < 360):
            continue
        table.append((h, k, l, theta, phi))
    table.sort(key=lambda x: (x[3], x[4]))
    return table

RELAXDIR = "relax/"
CASCADEDIR = "cascade/"
class TDESearch:
    def __init__(self, base_dir="./", init_energy=1, max_energy=1000, precision=0.01, 
                 nx=1, ny=1, nz=1, thickness=1, direction=np.array([0, 0, 1])):
        self.base_dir = Path(base_dir).resolve()
        self.work_dir = self.base_dir / CASCADEDIR
        self.init_energy = init_energy
        self.max_energy = max_energy
        self.precision = precision
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.thickness = thickness
        self.direction = direction
    
    def set_work_dir(self, work_dir):
        self.work_dir = Path(work_dir).resolve()
    
    def set_direction(self, direction):
        self.direction = direction

    def clean_directory(self, directory):
        if directory.exists():
            print(f"Cleaning directory: {directory}", flush=True)
            shutil.rmtree(directory)

    def run_defects_analyzer(self, energy):
        script_path = Path(__file__).resolve().parent / "defects_analyzer.py"
        command = ["apptainer", "exec", 
                   "/opt/software/ovito-3.12.3/bin/ovito_python.sif",
                   "python", str(script_path),
                   "--work_dir", str(self.work_dir),
                   "--cascade_dir", str(self.work_dir / CASCADEDIR),
                   "--energy", str(energy)]
        subprocess.run(command, check=True)

    def read_frenkel_pairs(self):
        summary_file = self.work_dir / "final_defect_summary.txt"
        if not summary_file.exists():
            return 0
        with summary_file.open() as f:
            last_line = f.readlines()[-1].strip()
            return int(last_line.split("\t")[-1])
            
    def run_relax_simulation(self, **kwargs):
        relax_dir = self.base_dir / RELAXDIR
        run_relax(dirname=relax_dir,
                  nx=self.nx,
                  ny=self.ny,
                  nz=self.nz,
                  thickness=self.thickness, 
                  **kwargs)
        restart_path = relax_dir / "restart.xyz"
        return restart_path

    def run_cascade_simulation(self, energy, **kwargs):
        self.clean_directory(self.work_dir / CASCADEDIR)
        run_cascade(
            dirname=self.work_dir / CASCADEDIR,
            energy=energy,
            direction=self.direction,
            **kwargs
        )
        self.run_defects_analyzer(energy)
        return self.read_frenkel_pairs()

    def exponential_growth_search(self, input_file, **kwargs):
        print("\nStarting exponential growth search...", flush=True)
        current_energy = self.init_energy
        high_energy = None
        while current_energy <= self.max_energy:
            fp_count = self.run_cascade_simulation(current_energy, input_file, **kwargs)
            print(f"Energy: {current_energy:.2f}eV -> Frenkel pairs: {fp_count}", flush=True) 
            if fp_count > 0:
                high_energy = current_energy
                print(f"First damage found at {high_energy:.2f}eV", flush=True)
                break 
            current_energy *= 2 
        if high_energy is None:
            raise RuntimeError(f"No damage found below {self.max_energy}eV") 
        return high_energy, current_energy / 2

    def binary_search(self, low_energy, high_energy, input_file, **kwargs):
        print("\nStarting binary search...", flush=True)
        iteration = 0
        max_iterations = 100
        while abs(high_energy - low_energy) > self.precision and iteration < max_iterations:
            mid_energy = (low_energy + high_energy) / 2
            fp_count = self.run_cascade_simulation(mid_energy, input_file, **kwargs)
            print(f"Energy: {mid_energy:.2f}eV -> Frenkel pairs: {fp_count}", flush=True)
            if fp_count > 0:
                high_energy = mid_energy
            else:
                low_energy = mid_energy
            iteration += 1
        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations ({max_iterations}) reached", flush=True)
            print(f"Final values: Low={low_energy:.3f}eV, High={high_energy:.3f}eV", flush=True)
        return high_energy

    def find_tde(self, input_file, **kwargs):
        if input_file is None:
            print("Please run relax simulation first!", flush=True)
        damage_energy, no_damage_energy = self.exponential_growth_search(input_file, **kwargs)
        tde = self.binary_search(no_damage_energy, damage_energy, input_file, **kwargs)
        self.clean_directory(self.work_dir / CASCADEDIR)
        print(f"\nSearch completed! TDE: {tde:.3f}eV", flush=True)
        return tde
    
