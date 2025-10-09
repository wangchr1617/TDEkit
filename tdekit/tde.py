from .core import run_cascade, run_relax
from itertools import combinations_with_replacement, product
from math import gcd
from pathlib import Path
import csv
import hashlib
import numpy as np
import random
import shutil
import subprocess
import time


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

def generate_uniform_directions(n):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    mask = z >= 0
    x = x[mask]
    y = y[mask]
    z = z[mask]
    points = [(x[i], y[i], z[i]) for i in range(len(x))]
    return points

def generate_uniform_direction_table(n):
    directions = generate_uniform_directions(n)
    table = []    
    for d in directions:
        x, y, z = d
        theta, phi = calculate_angles(x, y, z)
        if 0 <= theta <= 90 and 0 <= phi < 360:
            table.append((x, y, z, theta, phi))
    table.sort(key=lambda x: (x[3], x[4]))
    return table

def save_direction_table_to_csv(table, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if table and len(table[0]) == 5:
            if isinstance(table[0][0], int):
                writer.writerow(['h', 'k', 'l', 'theta', 'phi'])
            else:
                writer.writerow(['x', 'y', 'z', 'theta', 'phi'])
        for row in table:
            writer.writerow(row)

def read_direction_table_from_csv(filename):
    table = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            if len(row) == 5:
                try:
                    if '.' not in row[0]:
                        h = int(row[0])
                        k = int(row[1])
                        l = int(row[2])
                        theta = float(row[3])
                        phi = float(row[4])
                        table.append((h, k, l, theta, phi))
                        continue
                except ValueError:
                    pass
                x = float(row[0])
                y = float(row[1])
                z = float(row[2])
                theta = float(row[3])
                phi = float(row[4])
                table.append((x, y, z, theta, phi))
    return table

RELAXDIR = "relax/"
CASCADEDIR = "cascade/"
class TDESearch:
    def __init__(self, base_dir="./", init_energy=1, max_energy=1000, 
                 precision=1, direction=np.array([0, 0, 1])):
        self.base_dir = Path(base_dir).resolve()
        self.work_dir = self.base_dir / CASCADEDIR
        self.init_energy = init_energy
        self.max_energy = max_energy
        self.precision = precision
        self.direction = direction
        self.n_simulation = 0
    
    def set_work_dir(self, work_dir):
        self.work_dir = Path(work_dir).resolve()
        print(f"Set work directory to: {self.work_dir}")
    
    def set_direction(self, direction):
        self.direction = direction
        print(f"Set PKA direction to: {self.direction}")

    def clean_directory(self, directory):
        if directory.exists():
            shutil.rmtree(directory)

    def get_unique_dir_id(self):
        self.n_simulation += 1
        dir_name = f"cascade_id{self.n_simulation:04d}"        
        hash_input = f"{dir_name}_{time.time()}_{random.randint(0, 1000000)}"
        unique_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        dir_name += f"_{unique_hash}"        
        return dir_name

    def run_defects_analyzer(self, energy, work_dir=None, cascade_dir=None):
        if work_dir is None:
            work_dir = self.work_dir
        if cascade_dir is None:
            cascade_dir = CASCADEDIR
        script_path = Path(__file__).resolve().parent / "defects_analyzer.py"
        command = ["apptainer", "exec", 
                   "/opt/software/ovito-3.12.3/bin/ovito_python.sif",
                   "python", str(script_path),
                   "--work_dir", str(work_dir),
                   "--cascade_dir", str(work_dir / cascade_dir),
                   "--energy", str(energy)]
        subprocess.run(command, check=True)

    def read_frenkel_pairs(self, work_dir=None):
        if work_dir is None:
            work_dir = self.work_dir
        summary_file = work_dir / "final_defect_summary.txt"
        if not summary_file.exists():
            print(f"Warning: Summary file not found at {summary_file}")
            return 0
        with summary_file.open() as f:
            lines = f.readlines()
            if not lines:
                print("Warning: Summary file is empty")
                return 0
            last_line = lines[-1].strip()
            fp_count = int(last_line.split("\t")[-1])
            print(f"Read {fp_count} Frenkel pairs from summary file")
            return fp_count
            
    def run_relax_simulation(self, relax_dir=None, **kwargs):
        if relax_dir is None:
            relax_dir = self.base_dir / RELAXDIR
        print("Running relaxation simulation...")
        run_relax(dirname=relax_dir, **kwargs)
        restart_path = relax_dir / "restart.xyz"
        return restart_path

    def run_cascade_simulation(self, energy, work_dir=None, clean=True, defects_analysis=True, **kwargs):
        if work_dir is None:
            work_dir = self.work_dir
        cascade_dir = self.get_unique_dir_id()
        print(f"[Energy {energy}eV] Running cascade simulation in directory: {cascade_dir}")

        run_cascade(
            dirname=work_dir / cascade_dir,
            energy=energy,
            direction=self.direction,
            **kwargs
        )
        
        if defects_analysis:
            self.run_defects_analyzer(energy, work_dir, cascade_dir)
            frenkel_pairs = self.read_frenkel_pairs(work_dir)
        else:
            frenkel_pairs = -1
            
        if clean:
            self.clean_directory(work_dir / cascade_dir)
            
        return frenkel_pairs, cascade_dir

    def exponential_growth_search(self, **kwargs):
        print("\n=== Starting exponential growth search ===")
        current_energy = self.init_energy
        high_energy = None
        
        while current_energy <= self.max_energy:
            print(f"Testing energy: {current_energy:.2f}eV")
            fp_count, _ = self.run_cascade_simulation(current_energy, **kwargs)
            print(f"Energy: {current_energy:.2f}eV -> Frenkel pairs: {fp_count}", flush=True) 
            if fp_count > 0:
                high_energy = current_energy
                low_energy = current_energy / 2
                print(f"Damage found at {high_energy:.2f}eV")
                print(f"Exponential search complete. Damage bounds: [{low_energy:.2f}, {high_energy:.2f}] eV")
                return high_energy, low_energy                
            current_energy *= 2
            print(f"No damage found, doubling energy to {current_energy:.2f}eV")            
        raise RuntimeError(f"No damage detected up to maximum energy {self.max_energy}eV!")

    def binary_search(self, low_energy, high_energy, num_samples, threshold_prob, **kwargs):
        print("\n=== Starting binary search with statistical sampling ===")
        print(f"Initial bounds: [{low_energy:.3f}, {high_energy:.3f}] eV")
        print(f"Sampling parameters: {num_samples} samples per point, threshold probability: {threshold_prob}")
        
        iteration = 0
        max_iterations = 50
        
        def get_fp_probability(energy, n_samples):
            print(f"\n--- Sampling at {energy:.3f}eV ---")
            fp_count = 0
            for i in range(n_samples):
                count, _ = self.run_cascade_simulation(energy, **kwargs)
                if count > 0:
                    fp_count += 1
                    print(f"Sample {i+1}: Damage detected ({count} FP)")
                else:
                    print(f"Sample {i+1}: No damage detected")
            probability = fp_count / n_samples
            print(f"Energy {energy:.3f}eV: FP probability = {probability:.2f} ({fp_count}/{n_samples})")
            return probability
        
        print("Verifying initial bounds...")
        low_prob = get_fp_probability(low_energy, num_samples)
        high_prob = get_fp_probability(high_energy, num_samples)
        
        while low_prob >= threshold_prob and iteration < max_iterations:
            print(f"Lower bound {low_energy:.3f}eV has probability {low_prob:.2f} >= threshold")
            high_energy = low_energy
            high_prob = low_prob
            low_energy = low_energy / 2
            print(f"Adjusting bounds: [{low_energy:.3f}, {high_energy:.3f}] eV")
            low_prob = get_fp_probability(low_energy, num_samples)
            iteration += 1
            
        while high_prob < threshold_prob and iteration < max_iterations:
            print(f"Upper bound {high_energy:.3f}eV has probability {high_prob:.2f} < threshold")
            low_energy = high_energy
            low_prob = high_prob
            high_energy = high_energy * 2
            print(f"Adjusting bounds: [{low_energy:.3f}, {high_energy:.3f}] eV")
            high_prob = get_fp_probability(high_energy, num_samples)
            iteration += 1
        
        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations ({max_iterations}) reached during boundary adjustment!")
            final_estimate = (low_energy + high_energy) / 2
            print(f"Returning early with estimate: {final_estimate:.3f}eV")
            return final_estimate
        
        while abs(high_energy - low_energy) > self.precision and iteration < max_iterations:
            if abs(high_energy - low_energy) > 40:
                n_samples = 1
            elif abs(high_energy - low_energy) > 20:
                n_samples = 5
            else:
                n_samples = num_samples
                
            iteration += 1
            print(f"\n--- Binary Search Iteration {iteration} ---")
            print(f"Current bounds: [{low_energy:.3f}, {high_energy:.3f}] eV")
            print(f"Interval size: {high_energy - low_energy:.3f}eV (target: <{self.precision}eV)")
            
            mid_energy = (low_energy + high_energy) / 2
            print(f"Testing midpoint: {mid_energy:.3f}eV")
            prob = get_fp_probability(mid_energy, n_samples)
            
            if prob >= threshold_prob:
                high_energy = mid_energy
                high_prob = prob
                print(f"Probability {prob:.2f} >= threshold {threshold_prob}, setting high_energy = {high_energy:.3f}eV")
            else:
                low_energy = mid_energy
                low_prob = prob
                print(f"Probability {prob:.2f} < threshold {threshold_prob}, setting low_energy = {low_energy:.3f}eV")
        
        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations ({max_iterations}) reached!")
            print(f"Final bounds: Low={low_energy:.3f}eV, High={high_energy:.3f}eV")
        
        final_estimate = (low_energy + high_energy) / 2
        print(f"\nBinary search completed after {iteration} iterations")
        print(f"Final TDE estimate: {final_estimate:.3f}eV")
        
        return final_estimate, low_energy, low_prob, high_energy, high_prob

    def find_tde(self, num_samples=10, verification_samples=10, threshold_prob=0.5, **kwargs):
        print("\n" + "="*60)
        print("STARTING TDE SEARCH")
        print("="*60)
        damage_energy, no_damage_energy = self.exponential_growth_search(**kwargs)
        tde_estimate, low_energy, low_prob, high_energy, high_prob = self.binary_search(
            no_damage_energy, 
            damage_energy, 
            num_samples=num_samples,
            threshold_prob=threshold_prob,
            **kwargs
        )
        print("\n=== Starting verification phase ===")
        print(f"Running {verification_samples} verification samples at {tde_estimate:.3f}eV")
        
        fp_detected = 0
        for i in range(verification_samples):           
            count, _ = self.run_cascade_simulation(tde_estimate, **kwargs)            
            if count > 0:
                fp_detected += 1
                print(f"Sample {i+1}: Damage detected ({count} FP)")
            else:
                print(f"Sample {i+1}: No damage detected")        
        verified_probability = fp_detected / verification_samples
        
        if verified_probability < threshold_prob:
            tde_estimate = high_energy
            verified_probability = high_prob
        
        print(f"\nVerification results at {tde_estimate:.3f}eV:")
        print(f"Damage probability: {verified_probability:.2f}")
        
        print("\n" + "="*60)
        print("TDE SEARCH COMPLETE")
        print("="*60)
        print(f"TDE along <{self.direction}> direction: {tde_estimate:.3f} ± {self.precision:.3f} eV")
        print(f"At this energy, FP formation probability is {verified_probability:.2f}")
        print("="*60)
        
        return tde_estimate, verified_probability
    
