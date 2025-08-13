import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import umap.umap_ as umap
from .core import read_xyz, dump_xyz
from calorine.nep import get_descriptors
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.spatial import distance, KDTree
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.manifold import TSNE
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FarthestPointSample:
    def __init__(self, min_distance=0.1, min_select=1, max_select=None,
                 metric='euclidean', metric_para={}, use_kdtree=True):
        """
        min_distance=0.1：点间最小距离阈值
        min_select=1：最少采样点数
        max_select=None：最大采样点数（默认无限制）
        metric='euclidean'：距离度量标准
        metric_para={}：距离计算参数
        use_kdtree=True：是否使用 KDTree 加速计算
        """
        self.min_distance = min_distance
        self.min_select = min_select
        self.max_select = max_select
        self.metric = metric
        self.metric_para = metric_para
        self.use_kdtree = use_kdtree

    def select(self, points, selected_indices=None):
        """
        points: 输入的点云，一个二维数组（点集）。
        selected_indices: 可选，已经选中的点的索引列表。如果提供，则在这些点的基础上继续选择；否则从零开始。
        """
        points = np.asarray(points)
        n = len(points)
        logging.info(f"Starting sampling on {n} points (dims={points.shape[1]})")
        logging.info(f"Params: min_dist={self.min_distance}, min_select={self.min_select}, max_select={self.max_select}")
        logging.info(f"Metric: {self.metric} ({'KDTree' if self.use_kdtree and n > 1000 else 'cdist'})")
        max_select = min(self.max_select or n, n)
        min_select = min(self.min_select, max_select)
        logging.info(f"Final select range: {min_select} to {max_select} points")
        selected = np.zeros(n, dtype=bool) # 布尔数组标记已选点
        min_dists = np.full(n, np.inf) # 保存每个点距已选点的最小距离
        new_indices = [] # 存储采样点索引
        if not selected_indices:
            start_idx = np.random.randint(n)
            selected[start_idx] = True
            new_indices.append(start_idx)
            logging.info(f"Random initial point selected: index={start_idx}")
            if self.use_kdtree and n > 1000:
                logging.info("Building KDTree for large point cloud")
                self.kdtree = KDTree(points)
                min_dists, _ = self.kdtree.query(points[selected], k=1, workers=-1)
                min_dists = min_dists.ravel()
                logging.debug(f"Initial distances computed via KDTree")
            else:
                min_dists = distance.cdist(points, [points[start_idx]], metric=self.metric, **self.metric_para)[:, 0]
                logging.debug(f"Initial distances computed via direct cdist")
        else:
            logging.info(f"Processing {len(selected_indices)} pre-selected points")
            if self.use_kdtree and n > 1000:
                logging.info("Building KDTree for large point cloud")
                self.kdtree = KDTree(points)
            for idx in selected_indices:
                selected[idx] = True
                new_indices.append(idx)
                if hasattr(self, 'kdtree'):
                    dists, _ = self.kdtree.query([points[idx]], k=1)
                    new_dists = dists.ravel()
                else:
                    new_dists = distance.cdist(points, [points[idx]], metric=self.metric, **self.metric_para)[:, 0]
                min_dists = np.minimum(min_dists, new_dists)
            logging.info(f"Pre-selected points processed. Current min distance: {np.max(min_dists):.4f}")
        logging.info("Entering main sampling loop...")
        log_interval = max(1, max_select // 10)
        iteration = 0
        while ((min_select and len(new_indices) < min_select) or
               (np.max(min_dists) > self.min_distance)) and len(new_indices) < max_select:
            candidate_idx = np.argmax(min_dists)
            candidate_point = points[candidate_idx]
            selected[candidate_idx] = True
            new_indices.append(candidate_idx)
            if iteration % log_interval == 0 or iteration < 5:
                max_dist = np.max(min_dists)
                coverage = len(new_indices) / max_select * 100
                logging.info(
                    f"Iter {iteration}: Selected {len(new_indices)}/{max_select} points "
                    f"(coverage={coverage:.1f}%), Max distance={max_dist:.4f}, "
                    f"Candidate index={candidate_idx}"
                )
            if self.use_kdtree and hasattr(self, 'kdtree') and n > 1000:
                new_dists, _ = self.kdtree.query([candidate_point], k=1)
                new_dists = new_dists.flatten()
            else:
                new_dists = distance.cdist(points, [candidate_point], metric=self.metric, **self.metric_para)[:, 0]
            min_dists = np.minimum(min_dists, new_dists)
            min_dists[selected] = 0
            iteration += 1
        max_dist = np.max(min_dists)
        logging.info(
            f"Sampling complete after {iteration} iterations. "
            f"Selected {len(new_indices)} points. "
            f"Final max distance={max_dist:.4f}, "
            f"Min distance threshold={self.min_distance}"
        )
        return new_indices

class DescriptorAnalyzer:
    def __init__(self, model_filename, method='pca',
                 batch_size=1000, store_frames=True):
        self.model_filename = model_filename
        self.method = method.lower()
        self.batch_size = batch_size
        self.store_frames = store_frames
        self.labels = []
        self.natoms_per_frame = []
        self.frames_per_file = []
        self.frames = []
        self.latent_cache = {}
        self.latent_time = 0
        self.descriptors = []
        self.flat_descriptors = None
        self.structure_indices = None
        self.progress_counter = 0
        self.total_frames = 0

    def add_xyz_file(self, xyz_path, label):
        start_time = time()
        frames = read_xyz(xyz_path)
        n_frames = len(frames)
        self.total_frames += n_frames
        logger.info(f"Processing {n_frames} frames from {os.path.basename(xyz_path)}")
        logger.debug(f"XYZ file path: {xyz_path}")
        if self.store_frames:
            self.frames.extend(frames)
            logger.debug(f"Storing {n_frames} frames in memory")
        descriptors = []
        logger.info(f"Serial descriptor computation ({n_frames} frames)")
        log_interval = max(1, n_frames // 10)
        for i, atoms in enumerate(frames):
            desc = get_descriptors(atoms, self.model_filename)
            descriptors.append(desc)
            self.progress_counter += 1
            if (i+1) % log_interval == 0 or (i+1) == n_frames:
                local_progress = (i+1) / n_frames * 100
                global_progress = self.progress_counter / self.total_frames * 100
                logger.info(
                    f"Processed: {i+1}/{n_frames} frames (local: {local_progress:.1f}%, "
                    f"global: {global_progress:.1f}%)"
                )
        self.descriptors.extend(descriptors)
        self.labels.append(label)
        self.natoms_per_frame.extend([len(atoms) for atoms in frames])
        self.frames_per_file.append(n_frames)
        self.latent_cache.clear()
        self.flat_descriptors = None
        duration = time() - start_time
        atom_count = sum(self.natoms_per_frame[-n_frames:])
        fps = n_frames / duration if duration > 0 else float('inf')
        logger.info(
            f"Processed {n_frames} frames ({atom_count} atoms) from {os.path.basename(xyz_path)} in {duration:.1f}s "
            f"({fps:.1f} fps)"
        )
        logger.debug(f"Cleared latent space cache")

    def _get_flat_descriptors(self):
        if self.flat_descriptors is None:
            start_time = time()
            self.flat_descriptors = np.concatenate(self.descriptors)
            self.structure_indices = np.repeat(np.arange(len(self.descriptors)), [len(d) for d in self.descriptors])
            logger.info(f"Generated flattened descriptors in {time() - start_time:.1f}s")
        return self.flat_descriptors

    def _get_structure_descriptors(self):
        start_time = time()
        return np.array([np.mean(d, axis=0) for d in self.descriptors])

    def perform_decomposition(self, force_recompute=False):
        cache_key = self.method
        if not force_recompute and cache_key in self.latent_cache:
            logger.debug(f"Using cached {self.method.upper()} results")
            return self.latent_cache[cache_key]
        start_time = time()
        n_points = sum(len(d) for d in self.descriptors)
        logger.info(f"Starting {self.method.upper()} dimensionality reduction on {n_points} points")
        if n_points > 10000 and self.method == 'pca':
            logger.info(f"Using IncrementalPCA for large dataset ({n_points} points)...")
            ipca = IncrementalPCA(n_components=2, batch_size=self.batch_size)
            for i in range(0, len(self.descriptors), self.batch_size):
                batch = np.concatenate(self.descriptors[i:i + self.batch_size])
                ipca.partial_fit(batch)
                logger.debug(f"Processed batch {i} to {min(i + self.batch_size, len(self.descriptors))}")
            latent = []
            for i in range(0, len(self.descriptors), self.batch_size):
                batch = np.concatenate(self.descriptors[i:i + self.batch_size])
                latent.append(ipca.transform(batch))
                logger.debug(f"Transformed batch {i} to {min(i + self.batch_size, len(self.descriptors))}")
            points = np.concatenate(latent)
        else:
            data = self._get_flat_descriptors()
            if self.method == 'pca':
                logger.info("Using standard PCA...")
                pca = PCA(n_components=2)
                points = pca.fit_transform(data)
            elif self.method == 'tsne':
                logger.info("Using t-SNE...")
                tsne = TSNE(n_components=2, perplexity=10, learning_rate='auto', init='pca', random_state=0, method="barnes_hut")
                points = tsne.fit_transform(data)
            elif self.method == 'umap':
                logger.info("Using UMAP...")
                umap_model = umap.UMAP(n_components=2)
                points = umap_model.fit_transform(data)
        self.latent_cache[cache_key] = points
        self.latent_time = time() - start_time
        logger.info(f"{self.method.upper()} completed in {self.latent_time:.1f}s")
        return points

    def _plot(self, ax, points, selected_atoms=None, **kwargs):
        start = 0
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (label, n_frames) in enumerate(zip(self.labels, self.frames_per_file)):
            file_atom_count = n_frames * self.natoms_per_frame[0]
            end = start + file_atom_count
            file_points = points[start:end]
            ax.scatter(file_points[:, 0], file_points[:, 1],
                       color=color_cycle[i % len(color_cycle)],
                       alpha=0.6, label=label, **kwargs)
            if selected_atoms is not None:
                if len(selected_atoms) >= end:
                    sel_mask = selected_atoms[start:end]
                    ax.scatter(file_points[sel_mask, 0], file_points[sel_mask, 1],
                               color='red', marker='o', s=50, edgecolors='black')
                else:
                    logger.warning(f"selected_atoms length mismatch ({len(selected_atoms)} vs total points {len(points)})")
            start = end

    def perform_latent_analysis(self, ax=None, min_distance=0.1, min_select=1,
                                max_select=None, level='structure', **kwargs):
        """
        Optimized latent space analysis
        level: 'structure' for structure-level, 'atomic' for atomic-level
        """
        start_time = time()
        logger.info(f"Starting latent space analysis (level: {level})")
        total_atoms = sum(len(d) for d in self.descriptors)
        if level == 'atomic' and total_atoms > 100000:
            logger.warning(
                f"Atomic-level analysis disabled for large dataset ({total_atoms} atoms), "
                f"switching to structure-level"
            )
            level = 'structure'
        if level == 'structure':
            logger.info("Using structure-level descriptors...")
            descriptors = self._get_structure_descriptors()
            indices = np.arange(len(descriptors))
            points = self.perform_decomposition()
            structure_points = np.zeros((len(self.descriptors), 2))
            start = 0
            for i, desc in enumerate(self.descriptors):
                end = start + len(desc)
                structure_points[i] = np.mean(points[start:end], axis=0)
                start = end
            points = structure_points
        else:
            logger.info("Using atomic-level descriptors...")
            descriptors = self._get_flat_descriptors()
            points = self.perform_decomposition()
            indices = np.arange(len(descriptors))
        if min_select < 1:
            min_select = max(1, int(len(descriptors) * min_select))
        max_select = max(min_select, min(max_select or len(descriptors), len(descriptors)))
        logger.info(f"Sampling parameters: min_select={min_select}, max_select={max_select}, min_distance={min_distance}")
        if len(descriptors) > 100000:
            logger.info("Dataset too large (>100,000 points), using random sampling instead of FPS")
            selected_idx = np.random.choice(len(descriptors), size=min_select, replace=False)
        else:
            logger.info("Using Farthest Point Sampling...")
            sampler = FarthestPointSample(min_distance, min_select, max_select)
            selected_idx = sampler.select(descriptors)
        all_structures = set(range(len(self.descriptors)))
        if level == 'structure':
            selected_structures = set(selected_idx)
        else:
            selected_structures = set(np.unique(self.structure_indices[selected_idx]))
        unselected_structures = all_structures - selected_structures
        logger.info(f"Selected {len(selected_structures)} structures")
        if ax:
            selection_mask = None
            if level == 'atomic':
                selection_mask = np.zeros(len(points), dtype=bool)
                selection_mask[selected_idx] = True
            self._plot(ax, points, selected_atoms=selection_mask, **kwargs)
        return selected_structures, unselected_structures

    def split_xyz(self, filename, selected_set, overwrite=True):
        logger.info(f"Exporting structures to {filename}")
        if not hasattr(self, 'frames') or not self.frames:
            raise AttributeError("Original frames not stored, initialize with store_frames=True")
        if overwrite and os.path.exists(filename):
            logger.warning(f"File {filename} exists and will be overwritten")
            os.remove(filename)
        if isinstance(selected_set, np.ndarray):
            selected_set = set(selected_set.flatten())
        elif not isinstance(selected_set, (set, list, tuple)):
            selected_set = {int(selected_set)}
        logger.info(f"Exporting {len(selected_set)} selected frames...")
        selected_frames = [self.frames[i] for i in selected_set]
        with open(filename, 'a', encoding='utf-8') as f_sel:
            for i, atoms in enumerate(selected_frames):
                dump_xyz(f_sel, atoms)
                if i % 100 == 0:
                    logger.debug(f"Exported {i + 1}/{len(selected_frames)} frames")
        logger.info(f"Successfully saved {len(selected_frames)} frames to {filename}")
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