from pathlib import Path
import shutil
import subprocess
import numpy as np
from .core import run_cascade, run_relax

class TDESearch:
    def __init__(self, init_energy=2, max_energy=1000, precision=0.01, 
                 nx=1, ny=1, nz=1, thickness=5, direction=np.array([0, 0, 1]), 
                 cascade_dir="cascade", defects_script="defects_analyzer.py"):
        self.init_energy = init_energy
        self.max_energy = max_energy
        self.precision = precision
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.thickness = thickness
        self.direction = direction
        self.cascade_dir = cascade_dir
        self.defects_script = defects_script

    def clean_directory(self, dir_path):
        if dir_path.exists():
            print(f"Cleaning directory: {dir_path}")
            shutil.rmtree(dir_path)

    def run_defects_analyzer(self, base_dir, energy):
        script_path = Path(__file__).resolve().parent / self.defects_script
        command = [
            "apptainer", "exec", 
            "/opt/software/ovito-3.12.3/bin/ovito_python.sif",
            "python", str(script_path), 
            "--base_dir", str(base_dir), 
            "--energy", str(energy)
        ]
        subprocess.run(command, check=True)

    def read_frenkel_pairs(self, base_dir):
        summary_file = base_dir / "final_defect_summary.txt"
        if not summary_file.exists():
            return 0
        with summary_file.open() as f:
            last_line = f.readlines()[-1].strip()
            return int(last_line.split("\t")[-1])

    def run_cascade_simulation(self, base_dir, energy, input_file):
        cascade_dir = base_dir / self.cascade_dir
        self.clean_directory(cascade_dir)
        run_cascade(input_file=input_file, energy=energy, direction=self.direction)
        self.run_defects_analyzer(base_dir, energy)
        return self.read_frenkel_pairs(base_dir)

    def exponential_growth_search(self, base_dir, input_file):
        print("\nStarting exponential growth search...")
        current_energy = self.init_energy
        high_energy = None
        while current_energy <= self.max_energy:
            fp_count = self.run_cascade_simulation(base_dir, current_energy, input_file)
            print(f"Energy: {current_energy:.2f}eV -> Frenkel pairs: {fp_count}") 
            if fp_count > 0:
                high_energy = current_energy
                print(f"First damage found at {high_energy:.2f}eV")
                break 
            current_energy *= 2 
        if high_energy is None:
            raise RuntimeError(f"No damage found below {self.max_energy}eV") 
        return high_energy, current_energy / 2

    def binary_search(self, base_dir, input_file, low_energy, high_energy):
        print("\nStarting binary search...")
        iteration = 0
        max_iterations = 100
        while abs(high_energy - low_energy) > self.precision and iteration < max_iterations:
            mid_energy = (low_energy + high_energy) / 2
            fp_count = self.run_cascade_simulation(base_dir, mid_energy, input_file)
            print(f"Energy: {mid_energy:.2f}eV -> Frenkel pairs: {fp_count}")
            if fp_count > 0:
                high_energy = mid_energy
            else:
                low_energy = mid_energy
            iteration += 1
        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations ({max_iterations}) reached")
            print(f"Final values: Low={low_energy:.3f}eV, High={high_energy:.3f}eV")
        return high_energy

    def find_tde(self, base_dir):
        base_dir = Path(base_dir)
        xyz_path = base_dir / "model.xyz"
        relax_path = base_dir / "relax" / "restart.xyz"
        cascade_dir = base_dir / self.cascade_dir
        if not relax_path.exists():
            print("Running relaxation calculation...")
            run_relax(input_file=xyz_path, nx=self.nx, ny=self.ny, nz=self.nz, thickness=self.thickness)
        else:
            print("Using existing relaxation results")
        damage_energy, no_damage_energy = self.exponential_growth_search(base_dir, relax_path)
        tde = self.binary_search(base_dir, relax_path, no_damage_energy, damage_energy)
        self.clean_directory(cascade_dir)
        print(f"\nSearch completed! TDE: {tde:.3f}eV")
        return tde
    