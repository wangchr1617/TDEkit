import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from .core import read_xyz, dump_xyz
from calorine.nep import get_descriptors
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.spatial import distance, KDTree
from sklearn.decomposition import IncrementalPCA, PCA
from time import time
from typing import List, Set, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FarthestPointSample:
    def __init__(self, min_distance: float = 0.1, min_select: int = 1,
                 max_select: int = None, metric: str = 'euclidean',
                 metric_para: dict = None):
        self.min_distance = min_distance
        self.min_select = min_select
        self.max_select = max_select
        self.metric = metric
        self.metric_para = metric_para or {}

    def select(self, points: np.ndarray, selected_indices: List[int] = None) -> List[int]:
        points = np.asarray(points)
        n = points.shape[0]
        dims = points.shape[1]

        logger.info(f"Starting FPS on {n} points (dims={dims})")
        logger.info(f"Params: min_dist={self.min_distance}, min_select={self.min_select}, max_select={self.max_select}")

        max_select = min(self.max_select or n, n)
        min_select = min(self.min_select, max_select)
        logger.info(f"Select range: {min_select} to {max_select} points")

        selected = np.zeros(n, dtype=bool)
        min_dists = np.full(n, np.inf)
        selected_idx_list = []

        if selected_indices:
            logger.info(f"Processing {len(selected_indices)} pre-selected points")
            pre_points = points[selected_indices]
            dist_matrix = distance.cdist(points, pre_points,
                                         metric=self.metric, **self.metric_para)
            min_dists = np.min(dist_matrix, axis=1)
            selected[selected_indices] = True
            selected_idx_list.extend(selected_indices)
            logger.info(f"Pre-selected processed. Max distance: {np.max(min_dists):.4f}")
        else:
            start_idx = np.random.randint(n)
            selected[start_idx] = True
            selected_idx_list.append(start_idx)
            min_dists = distance.cdist(points, [points[start_idx]],
                                       metric=self.metric, **self.metric_para)[:, 0]
            logger.info(f"Random start point: index={start_idx}")

        logger.info("Starting main sampling loop...")
        log_interval = max(1, max_select // 10)
        iteration = 0

        while ((len(selected_idx_list) < min_select) or
               (np.max(min_dists) > self.min_distance)) and (len(selected_idx_list) < max_select):

            candidate_idx = np.argmax(min_dists)
            selected[candidate_idx] = True
            selected_idx_list.append(candidate_idx)

            if iteration % log_interval == 0 or iteration < 5:
                max_dist = np.max(min_dists)
                coverage = len(selected_idx_list) / max_select * 100
                logger.info(
                    f"Iter {iteration}: Selected {len(selected_idx_list)}/{max_select} points "
                    f"({coverage:.1f}%), Max dist={max_dist:.4f}, Candidate={candidate_idx}"
                )

            new_dists = distance.cdist(points, [points[candidate_idx]],
                                       metric=self.metric, **self.metric_para)[:, 0]
            min_dists = np.minimum(min_dists, new_dists)
            min_dists[selected] = 0
            iteration += 1

        max_dist = np.max(min_dists)
        logger.info(
            f"FPS completed in {iteration} iterations. "
            f"Selected {len(selected_idx_list)} points. "
            f"Final max distance={max_dist:.4f} (threshold={self.min_distance})"
        )
        return selected_idx_list

class DescriptorAnalyzer:
    def __init__(self, model_filename: str, method: str = 'pca',
                 batch_size: int = 1000, store_frames: bool = True):
        self.model_filename = model_filename
        self.method = method.lower()
        self.batch_size = batch_size
        self.store_frames = store_frames

        self.labels = []
        self.natoms_per_frame = []
        self.frames_per_file = []
        self.frames = []

        self.descriptors = []
        self.flat_descriptors = None
        self.structure_indices = None

        self.latent_cache = {}
        self.latent_time = 0

        self.progress_counter = 0
        self.total_frames = 0

    def add_xyz_file(self, xyz_path: str, label: str):
        start_time = time()
        frames = read_xyz(xyz_path)
        n_frames = len(frames)
        self.total_frames += n_frames

        logger.info(f"Processing {n_frames} frames from {os.path.basename(xyz_path)}")
        logger.debug(f"XYZ path: {xyz_path}")

        if self.store_frames:
            self.frames.extend(frames)
            logger.debug(f"Storing {n_frames} frames in memory")

        descriptors = []
        logger.info(f"Computing descriptors ({n_frames} frames)")
        log_interval = max(1, n_frames // 10)

        for i, atoms in enumerate(frames):
            desc = get_descriptors(atoms, self.model_filename)
            descriptors.append(desc)
            self.progress_counter += 1

            if (i + 1) % log_interval == 0 or (i + 1) == n_frames:
                local_progress = (i + 1) / n_frames * 100
                global_progress = self.progress_counter / self.total_frames * 100
                logger.info(
                    f"Processed: {i + 1}/{n_frames} frames (local: {local_progress:.1f}%, "
                    f"global: {global_progress:.1f}%)"
                )

        self.descriptors.extend(descriptors)
        self.labels.append(label)

        file_natoms = [len(atoms) for atoms in frames]
        self.natoms_per_frame.extend(file_natoms)
        self.frames_per_file.append(n_frames)

        self.latent_cache.clear()
        self.flat_descriptors = None

        duration = time() - start_time
        atom_count = sum(file_natoms)
        fps = n_frames / duration if duration > 0 else float('inf')
        logger.info(
            f"Completed {n_frames} frames ({atom_count} atoms) in {duration:.1f}s ({fps:.1f} fps)"
        )

    def _get_flat_descriptors(self) -> np.ndarray:
        if self.flat_descriptors is None:
            start_time = time()
            self.flat_descriptors = np.concatenate(self.descriptors)

            self.structure_indices = np.concatenate([
                np.full(len(d), i) for i, d in enumerate(self.descriptors)
            ])

            logger.info(f"Flattened descriptors created in {time() - start_time:.1f}s")
        return self.flat_descriptors

    def _get_structure_descriptors(self) -> np.ndarray:
        start_time = time()
        struct_desc = np.array([np.mean(d, axis=0) for d in self.descriptors])
        logger.info(f"Structure descriptors computed in {time() - start_time:.1f}s")
        return struct_desc

    def perform_decomposition(self, data: np.ndarray = None, force_recompute: bool = False) -> np.ndarray:
        cache_key = f"{self.method}_{data.shape if data is not None else 'atomic'}"

        if not force_recompute and cache_key in self.latent_cache:
            logger.debug(f"Using cached {cache_key} results")
            return self.latent_cache[cache_key]

        if data is None:
            data = self._get_flat_descriptors()
            data_type = "atomic-level"
        else:
            data_type = f"{data.shape[1]}D structure-level"

        start_time = time()
        n_points = len(data)
        logger.info(f"Starting {self.method.upper()} on {n_points} {data_type} points")

        if n_points > 10000 and self.method == 'pca':
            logger.info(f"Using IncrementalPCA for large dataset")
            ipca = IncrementalPCA(n_components=2, batch_size=self.batch_size)

            for i in range(0, n_points, self.batch_size):
                batch = data[i:i + self.batch_size]
                ipca.partial_fit(batch)

            points = []
            for i in range(0, n_points, self.batch_size):
                batch = data[i:i + self.batch_size]
                points.append(ipca.transform(batch))

            points = np.concatenate(points)
        else:
            if self.method == 'pca':
                logger.info("Using standard PCA")
                pca = PCA(n_components=2)
                points = pca.fit_transform(data)

            elif self.method == 'tsne':
                logger.info("Using t-SNE")
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, perplexity=min(30, n_points // 3),
                            n_iter=1000, random_state=42)
                points = tsne.fit_transform(data)

            elif self.method == 'umap':
                logger.info("Using UMAP")
                import umap
                reducer = umap.UMAP(n_components=2, n_neighbors=min(15, n_points - 1))
                points = reducer.fit_transform(data)

        self.latent_cache[cache_key] = points
        self.latent_time = time() - start_time
        logger.info(f"{self.method.upper()} completed in {self.latent_time:.1f}s")
        return points

    def _plot(self, ax, points: np.ndarray, selected_mask: np.ndarray = None,
              level: str = 'atomic', **kwargs):
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        unique_labels = sorted(set(self.labels))
        label_colors = {label: color_cycle[i % len(color_cycle)]
                        for i, label in enumerate(unique_labels)}

        if level == 'atomic':
            start_idx = 0
            for file_idx, (label, n_frames) in enumerate(zip(self.labels, self.frames_per_file)):
                file_natoms = sum(self.natoms_per_frame[start_idx:start_idx + n_frames])
                end_idx = start_idx + file_natoms

                file_points = points[start_idx:end_idx]
                color = label_colors[label]

                ax.scatter(file_points[:, 0], file_points[:, 1],
                           alpha=0.5, label=label, color=color, **kwargs)

                if selected_mask is not None:
                    file_selected = selected_mask[start_idx:end_idx]
                    if np.any(file_selected):
                        ax.scatter(file_points[file_selected, 0], file_points[file_selected, 1],
                                   color='red', edgecolor='black', alpha=0.75, zorder=5)

                start_idx = end_idx

        else:
            start_idx = 0
            for file_idx, (label, n_frames) in enumerate(zip(self.labels, self.frames_per_file)):
                end_idx = start_idx + n_frames
                file_points = points[start_idx:end_idx]
                color = label_colors[label]

                ax.scatter(file_points[:, 0], file_points[:, 1],
                           alpha=0.5, label=label, color=color, **kwargs)

                if selected_mask is not None:
                    file_selected = selected_mask[start_idx:end_idx]
                    if np.any(file_selected):
                        ax.scatter(file_points[file_selected, 0], file_points[file_selected, 1],
                                   color='red', edgecolor='black', alpha=0.75, zorder=5)

                start_idx = end_idx

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = sorted(set(labels), key=labels.index)
        unique_handles = [handles[labels.index(label)] for label in unique_labels]

        if unique_handles:
            ax.legend(unique_handles, unique_labels)

    def _divide_conquer_sampling(self, descriptors: np.ndarray, min_select: int,
                                 max_select: int, min_distance: float) -> np.ndarray:
        n = len(descriptors)
        chunk_size = 50000
        chunks = []

        for i in range(0, n, chunk_size):
            chunks.append({
                'start': i,
                'end': min(i + chunk_size, n),
                'data': descriptors[i:min(i + chunk_size, n)]
            })

        logger.info(f"Divide-and-conquer: {len(chunks)} chunks for {n} points")
        sampled_indices = []

        for chunk in chunks:
            chunk_data = chunk['data']
            start_idx = chunk['start']

            chunk_select = min(1000, len(chunk_data), max(1, min_select // 3))

            logger.debug(f"Processing chunk [{start_idx}-{start_idx + len(chunk_data)}] "
                         f"with {len(chunk_data)} points, sampling {chunk_select} points")

            sampler = FarthestPointSample(min_distance=min_distance,
                                          min_select=chunk_select,
                                          max_select=chunk_select)
            local_indices = sampler.select(chunk_data)
            global_indices = [start_idx + idx for idx in local_indices]
            sampled_indices.extend(global_indices)

        final_data = descriptors[sampled_indices]
        min_select_final = min(min_select, len(final_data))
        max_select_final = min(max_select or len(final_data), len(final_data))

        logger.info(f"Final sampling on {len(sampled_indices)} sampled points")

        sampler = FarthestPointSample(min_distance=min_distance,
                                      min_select=min_select_final,
                                      max_select=max_select_final)
        final_local_indices = sampler.select(final_data)
        global_indices = [sampled_indices[i] for i in final_local_indices]

        return np.array(global_indices)

    def perform_latent_analysis(self, ax=None, min_distance: float = 0.1,
                                min_select: Union[int, float] = 1, max_select: int = None,
                                level: str = 'structure', strategy=None,
                                **kwargs) -> Set[int]:
        start_time = time()
        total_atoms = sum(len(d) for d in self.descriptors)
        logger.info(f"Starting latent analysis (level={level}, atoms={total_atoms})")

        if level == 'structure':
            logger.info("Using structure-level analysis")
            descriptors = self._get_structure_descriptors()
            points = self.perform_decomposition(descriptors, force_recompute=False)
            n_points = len(descriptors)
            logger.info(f"Structure-level: {n_points} points")
        else:
            logger.info("Using atomic-level analysis")
            descriptors = self._get_flat_descriptors()
            points = self.perform_decomposition(descriptors, force_recompute=False)
            n_points = len(descriptors)
            logger.info(f"Atomic-level: {n_points} points")

        if isinstance(min_select, float):
            min_select = max(1, int(n_points * min_select))
        min_select = min(min_select, n_points)

        max_select = max_select or n_points
        max_select = min(max_select, n_points)

        logger.info(f"Sampling params: min={min_select}, max={max_select}, min_dist={min_distance}")

        if strategy is not None:
            logger.warning(f"Large dataset ({n_points} points), using {strategy} strategy")

            if strategy == 'divide':
                selected_idx = self._divide_conquer_sampling(descriptors, min_select, max_select, min_distance)
            else:
                selected_idx = np.random.choice(n_points, size=min_select, replace=False)
        else:
            logger.info("Using standard FPS sampling")
            sampler = FarthestPointSample(
                min_distance=min_distance,
                min_select=min_select,
                max_select=max_select)
            selected_idx = sampler.select(descriptors)

        if level == 'structure':
            selected_structures = set(selected_idx)
        else:
            selected_structures = set(self.structure_indices[selected_idx])

        all_structures = set(range(len(self.descriptors)))
        unselected_structures = all_structures - selected_structures

        logger.info(f"Selected {len(selected_structures)} structures")

        if ax:
            if level == 'atomic':
                selected_mask = np.zeros(n_points, dtype=bool)
                selected_mask[selected_idx] = True
            else:
                selected_mask = np.zeros(len(points), dtype=bool)
                selected_mask[list(selected_structures)] = True

            self._plot(ax, points, selected_mask=selected_mask,
                       level=level, **kwargs)

            method_name = self.method.upper()
            ax.set_title(f"{method_name} Projection ({level} level)\n"
                         f"Selected: {len(selected_structures)} structures")

        duration = time() - start_time
        logger.info(f"Latent analysis completed in {duration:.1f}s")
        return selected_structures, unselected_structures

    def split_xyz(self, filename: str, selected_set: Set[int], overwrite: bool = True) -> str:
        if not self.store_frames or not self.frames:
            raise AttributeError("Frames not stored (initialize with store_frames=True)")

        if overwrite and os.path.exists(filename):
            logger.warning(f"Overwriting existing file: {filename}")
            os.remove(filename)

        selected_list = sorted(selected_set)
        n_selected = len(selected_list)

        logger.info(f"Exporting {n_selected} structures to {filename}")

        with open(filename, 'w', encoding='utf-8') as f:
            for i, idx in enumerate(selected_list):
                if idx < 0 or idx >= len(self.frames):
                    logger.warning(f"Invalid frame index {idx}, skipping")
                    continue

                dump_xyz(f, self.frames[idx])

                if i > 0 and i % 100 == 0:
                    logger.info(f"Exported {i}/{n_selected} structures")

        logger.info(f"Successfully exported {n_selected} structures to {filename}")
        return filename

def main():
    # 1. 生成随机点云，二维坐标范围[0,1)
    np.random.seed(42)
    n_points = 200
    points = np.random.rand(n_points, 2)

    # 2. 创建FPS采样器实例
    fps = FarthestPointSample(
        min_distance=0.2,   # 最小采样间距
        min_select=10,      # 最少采样点
        max_select=20,      # 最多采样点
        use_kdtree=True     # 使用KDTree加速
    )

    # 3. 执行采样
    selected_indices = fps.select(points, selected_indices=[1, 22])
    print(f"Selected indices: {selected_indices}")

    # 4. 可视化
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Farthest Point Sampling Process', fontsize=16)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    all_points = ax.scatter(points[:, 0], points[:, 1], c='lightgray', s=15, alpha=0.7)
    selected_points = ax.scatter([], [], c='red', s=60, edgecolor='black', zorder=3)
    dist_image = ax.scatter([], [], c=[], s=15, alpha=0.8, cmap='viridis_r', vmin=0, vmax=0.5)
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 0.5), cmap='viridis_r'), ax=ax, label='Distance to Selected Points')
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top', fontsize=11,
                        bbox=dict(facecolor='white', alpha=0.8))

    # 5. 动画更新
    def update(frame):
        current_selected = selected_indices[:frame + 1]
        selected_points.set_offsets(points[current_selected])
        if len(current_selected) > 0:
            min_dists = distance.cdist(points, points[current_selected]).min(axis=1)
        else:
            min_dists = np.zeros(len(points))
        all_points.set_array(min_dists)
        info_text.set_text(
            f'Step: {frame + 1}/{len(selected_indices)}\n'
            f'Selected: {len(current_selected)} points\n'
            f'Max Distance: {min_dists.max():.4f}\n'
            f'Min Distance: {fps.min_distance:.2f}'
        )
        return all_points, selected_points, info_text

    # 6. 创建动画
    animation = FuncAnimation(
        fig,
        update,
        frames=len(selected_indices),
        interval=800,  # 每帧间隔(毫秒)
        blit=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()