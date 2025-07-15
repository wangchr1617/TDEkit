import matplotlib.pyplot as plt
import numpy as np
import os
from .core import read_xyz, dump_xyz
from time import time

class ForceAnalyzer:
    def __init__(self, calculators: list, frame_paths: list, frame_labels: list, 
                 minimum: float=0.25, maximum: float=0.5, bin_edges=np.linspace(0, 0.75, 15), 
                 load_max_delta=False, max_delta_dir='max_deltas', xyz_dir='xyzs'):
        self.calculators = calculators
        self.frame_paths = frame_paths
        self.frame_labels = frame_labels
        self.minimum = minimum
        self.maximum = maximum
        self.bin_edges = bin_edges
        self.load_max_delta = load_max_delta
        self.max_delta_dir = max_delta_dir
        self.xyz_dir = xyz_dir
    
    def _compute_max_deltas(self, frames, path, label):
        max_deltas = []
        for idx, atoms in enumerate(frames):
            n_atoms = len(atoms)
            force_matrix = np.empty((len(self.calculators), n_atoms, 3))
            for calc_idx, calculator in enumerate(self.calculators):
                atoms.calc = calculator
                force_matrix[calc_idx] = atoms.get_forces()
            mean_force = np.mean(force_matrix, axis=0)
            deltas = force_matrix - mean_force
            delta_norms = np.linalg.norm(deltas, axis=2)
            max_delta = np.max(delta_norms)
            max_deltas.append(max_delta)
        max_deltas = np.array(max_deltas)
        self.save_max_deltas_to_txt(max_deltas, path, label)
        return max_deltas

    def compute_max_deltas(self, frames, path, label):
        print(f"[COMPUTE] Starting max_delta computation for {label} ({len(frames)} frames)", flush=True)
        max_deltas = np.empty(len(frames))
        start_time = time()
        last_report_time = start_time
        
        for idx, atoms in enumerate(frames):
            n_atoms = len(atoms) 
            force_matrix = np.empty((len(self.calculators), n_atoms, 3))
            
            for calc_idx, calculator in enumerate(self.calculators):
                atoms.calc = calculator
                forces = atoms.get_forces()
                if len(forces) != n_atoms:
                    print(f"[WARNING] Frame {idx}: Force array shape mismatch ({len(forces)} vs {n_atoms})", flush=True)
                    forces = np.zeros((n_atoms, 3))
                force_matrix[calc_idx] = forces
            
            mean_force = np.mean(force_matrix, axis=0)
            deltas = force_matrix - mean_force
            delta_norms = np.linalg.norm(deltas, axis=2)
            max_deltas[idx] = np.max(delta_norms)
            
            current_time = time()
            if current_time - last_report_time > 60 or idx % 100 == 0 or idx == len(frames)-1:
                elapsed = current_time - start_time
                speed = (idx + 1) / elapsed if elapsed > 0 else float('inf')
                remaining = (len(frames) - idx - 1) / speed if speed > 0 else 0
                print(f"[PROGRESS] {path}: Processed {idx+1}/{len(frames)} frames "
                      f"({(idx+1)/len(frames)*100:.1f}%) | Speed: {speed:.1f} fps | "
                      f"ETA: {remaining:.0f}s | Current max_delta: {max_deltas[idx]:.6f}", flush=True)
                last_report_time = current_time
        
        total_time = time() - start_time
        print(f"[COMPLETE] Finished {label} in {total_time:.1f}s "
              f"({total_time/len(frames):.4f}s/frame) | Atoms varied: {len(frames[0])}→{len(frames[-1])}", flush=True)
        
        self.save_max_deltas_to_txt(max_deltas, path, label)
        return max_deltas

    def save_max_deltas_to_txt(self, max_deltas, path, label):
        os.makedirs(self.max_delta_dir, exist_ok=True)
        base_name = os.path.basename(path).replace('.', '_')
        filename = f"{self.max_delta_dir}/{base_name}_{label}_max_deltas.txt"
        with open(filename, 'w') as f:
            f.write(f"# Path: {path}\n")
            f.write(f"# Label: {label}\n")
            f.write(f"# Num_frames: {len(max_deltas)}\n")
            f.write("\n".join(map(str, max_deltas)))
        print(f"Saved max_deltas to: {filename}", flush=True)

    def load_max_deltas_from_txt(self, filename, path, label):
        with open(filename, 'r') as f:
            lines = f.readlines()
        max_deltas = []
        for line in lines[4:]:
            max_deltas.append(float(line.strip()))
        return np.array(max_deltas)

    def compute_frequency(self, max_deltas):
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        bin_centers = np.arange(self.bin_edges[0] + bin_width/2, self.bin_edges[-1] + bin_width/2, bin_width)
        counts, _ = np.histogram(max_deltas, bins=self.bin_edges)
        frequencies = counts / len(max_deltas) * 100
        return bin_centers, frequencies

    def split_xyz(self, frames, max_deltas, path, label):
        print(f"[SPLIT] Starting XYZ splitting for {label}", flush=True)
        os.makedirs(self.xyz_dir, exist_ok=True)
        base_name = os.path.basename(path).replace('.', '_')
        selected_filename = f"{self.xyz_dir}/{base_name}_{label}_selected.xyz"
        unselected_filename = f"{self.xyz_dir}/{base_name}_{label}_unselected.xyz"
        with open(selected_filename, 'a') as f_sel, open(unselected_filename, 'a') as f_unsel:
            for max_delta, atoms in zip(max_deltas, frames):
                if self.minimum < max_delta < self.maximum:
                    dump_xyz(f_sel, atoms)
                else:
                    dump_xyz(f_unsel, atoms)

    def plot_max_force_differences(self, ax, if_split=False):
        for path, label in zip(self.frame_paths, self.frame_labels):
            print(f"Processing path: {path} for label: {label}", flush=True)
            frames = read_xyz(path)
            print(f"Read {len(frames)} frames from {path}", flush=True)
            if self.load_max_delta:
                base_name = os.path.basename(path).replace('.', '_')
                filename = f"{self.max_delta_dir}/{base_name}_{label}_max_deltas.txt"
                max_deltas = self.load_max_deltas_from_txt(filename, path, label)
            else:
                max_deltas = self.compute_max_deltas(frames, path, label)
            if if_split:
                self.split_xyz(frames, max_deltas, path, label)
            bin_centers, frequencies = self.compute_frequency(max_deltas)
            ax.plot(bin_centers, frequencies, label=label, marker='o', markersize=4)
        ax.axvline(x=self.minimum, color='k')
        ax.axvline(x=self.maximum, color='k')
