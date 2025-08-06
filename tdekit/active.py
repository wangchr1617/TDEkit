import math
import matplotlib.pyplot as plt
import numpy as np
import os
from .core import read_xyz, dump_xyz
from calorine.nep import get_descriptors
from scipy.spatial.distance import cdist
from time import time

class FarthestPointSample:
    def __init__(self, min_distance=0.1, min_select=1, max_select=None, metric='euclidean', metric_para={}):
        self.min_distance = min_distance
        self.min_select = min_select
        self.max_select = max_select
        self.metric = metric
        self.metric_para = metric_para

    def select(self, points, selected_points=[]):
        max_select = self.max_select or len(points)
        to_add = []
        if len(points) == 0:
            return to_add
        if len(selected_points) == 0:
            to_add.append(0)
            selected_points.append(points[0])
        else:
            for point in selected_points:
                try:
                    index = points.index(point)
                    to_add.append(index)
                except ValueError:
                    continue
        distances = np.min(cdist(points, selected_points, metric=self.metric, **self.metric_para), axis=1)
        while np.max(distances) > self.min_distance or len(to_add) < self.min_select:
            i = np.argmax(distances)
            to_add.append(i)
            if len(to_add) >= max_select:
                break
            distances = np.minimum(distances, cdist([points[i]], points, metric=self.metric)[0])
        return to_add
    
class DescriptorAnalyzer:
    def __init__(self, model_filename, method='pca'):
        self.descriptors = [] 
        self.structure_indices_per_atom = []
        self.frames = [] 
        self.nframes = 0
        self.natoms = []
        self.labels = []
        self.method = method 
        self.model_filename = model_filename
    
    def add_xyz_file(self, xyz_path, label):
        frames = read_xyz(xyz_path)
        natoms = 0
        descriptors = []
        structure_indices_per_atom = []
        for i, atoms in enumerate(frames):
            descriptors.append(get_descriptors(atoms, model_filename=self.model_filename))
            structure_indices_per_atom.extend([i + self.nframes] * len(atoms))
            natoms += len(atoms)
        self.descriptors.extend(descriptors)
        self.frames.extend(frames)
        self.nframes += len(frames)
        print(f"Number of frames in {xyz_path}: {len(frames)}!")
        self.natoms.append(natoms)
        self.labels.append(label)
        self.structure_indices_per_atom.extend(structure_indices_per_atom)
        
    def perform_decomposition(self):
        all_descriptors = np.concatenate(self.descriptors, axis=0)
        if self.method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            return pca.fit_transform(all_descriptors)
        elif self.method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=10, learning_rate='auto', init='pca', random_state=0, method="barnes_hut")
            return tsne.fit_transform(all_descriptors)
        elif self.method == 'umap':
            import umap.umap_ as umap
            umap_model = umap.UMAP(n_components=2)
            return umap_model.fit_transform(all_descriptors)
        
    def _plot(self, ax, points, selected_points=None, **kwargs):
        start = 0
        for label, num in zip(self.labels, self.natoms):
            end = start + num
            ax.scatter(points[start:end, 0], points[start:end, 1], label=label, **kwargs)
            start = end
        if selected_points is not None:
            ax.scatter(selected_points[:, 0], selected_points[:, 1], label='selected', **kwargs)

    def perform_latent_analysis(self, ax, min_distance=0.1, min_select=1, max_select=None, level='structure', if_split=False, **kwargs):
        if level == 'structure':
            descriptors = np.array([np.mean(d, axis=0) for d in self.descriptors])
            structure = np.arange(len(descriptors))  
        elif level == 'atomic':
            descriptors = np.concatenate(self.descriptors) 
            structure = np.array(self.structure_indices_per_atom)
        if min_select < 1:
            min_select = math.floor(len(descriptors) * min_select)
        sampler = FarthestPointSample(min_distance, min_select, max_select)
        selected_indices = sampler.select(descriptors, [])        
        if level == 'structure':
            selected_set = set(selected_indices)
        elif level == 'atomic': 
            selected_set = {self.structure_indices_per_atom[i] for i in selected_indices}
        unselected_set = set(range(len(self.frames))) - selected_set
        indices_list = []
        for value in selected_set:
            indices = [index for index, elem in enumerate(self.structure_indices_per_atom) if elem == value]
            indices_list.extend(indices)
        points = self.perform_decomposition()
        if if_split:
            self._plot(ax, points, points[indices_list], **kwargs)
        else:
            self._plot(ax, points, **kwargs)
        return selected_set, unselected_set

    def split_xyz(self, filename, selected_set):
        if os.path.exists(filename):
            os.remove(filename)
        selected_frames = [self.frames[i] for i in selected_set]
        print(f"Number of selected frames: {len(selected_frames)}!")
        with open(filename, 'a') as f_sel:
            for atoms in selected_frames:
                dump_xyz(f_sel, atoms)

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
        if len(frames) == 0:
            print("[COMPLETE] Empty frame list - nothing to process", flush=True)
            return np.array([])
        
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
        atom_info = ""
        if frames: 
            initial_atoms = len(frames[0])
            final_atoms = len(frames[-1])
            atom_info = f" | Atoms varied: {initial_atoms}→{final_atoms}"
        
        print(f"[COMPLETE] Finished {label} in {total_time:.1f}s "
              f"({total_time/len(frames):.4f}s/frame){atom_info}", flush=True)
        
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
        for line in lines[3:]:
            max_deltas.append(float(line.strip()))
        return np.array(max_deltas)

    def compute_frequency(self, max_deltas):
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        bin_centers = np.arange(self.bin_edges[0] + bin_width/2, self.bin_edges[-1] + bin_width/2, bin_width)
        counts, _ = np.histogram(max_deltas, bins=self.bin_edges)
        frequencies = counts / len(max_deltas) * 100
        return bin_centers, frequencies

    def split_xyz(self, frames, max_deltas, path, label, if_unsel):
        print(f"[SPLIT] Starting XYZ splitting for {label}", flush=True)
        os.makedirs(self.xyz_dir, exist_ok=True)
        base_name = os.path.basename(path).replace('.', '_')
        selected_filename = f"{self.xyz_dir}/{base_name}_{label}_selected.xyz"
        with open(selected_filename, 'a') as f_sel:
            if if_unsel:
                unselected_filename = f"{self.xyz_dir}/{base_name}_{label}_unselected.xyz"
                with open(unselected_filename, 'a') as f_unsel:
                    for max_delta, atoms in zip(max_deltas, frames):
                        if self.minimum <= max_delta < self.maximum:
                            dump_xyz(f_sel, atoms)
                        else:
                            dump_xyz(f_unsel, atoms)
            else:
                for max_delta, atoms in zip(max_deltas, frames):
                    if self.minimum <= max_delta < self.maximum:
                        dump_xyz(f_sel, atoms)

    def plot_max_force_differences(self, ax, if_split=False, if_unsel=True):
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
                self.split_xyz(frames, max_deltas, path, label, if_unsel)
            bin_centers, frequencies = self.compute_frequency(max_deltas)
            ax.plot(bin_centers, frequencies, label=label, marker='o', markersize=4)
        ax.axvline(x=self.minimum, color='k')
        ax.axvline(x=self.maximum, color='k')
        ax.axvspan(xmin=self.minimum, xmax=self.maximum, facecolor="#E5A79A", alpha=0.5, zorder=0)
