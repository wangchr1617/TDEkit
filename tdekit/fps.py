import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
from typing import Tuple, Set, List, Dict, Optional, Union, Any
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.spatial import distance

# 导入本地模块与外部计算库
from .core import read_xyz, dump_xyz
from calorine.nep import get_descriptors

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义通用的路径类型
PathLike = Union[str, Path]

class FarthestPointSample:
    """
    最远点采样 (FPS) 算法实现。
    用于在特征空间或坐标空间中均匀、发散地挑选具有代表性的样本点。
    """
    def __init__(self, 
                 min_distance: float = 0.1, 
                 min_select: int = 1,
                 max_select: Optional[int] = None, 
                 metric: str = 'euclidean',
                 metric_para: Optional[Dict[str, Any]] = None):
        self.min_distance = min_distance
        self.min_select = min_select
        self.max_select = max_select
        self.metric = metric
        self.metric_para = metric_para or {}

    def select(self, points: Union[np.ndarray, List[List[float]]], 
               selected_indices: Optional[List[int]] = None) -> List[int]:
        points = np.asarray(points)
        n, dims = points.shape

        logger.info(f"Starting FPS on {n} points (dims={dims})")
        logger.info(f"Params: min_dist={self.min_distance}, min_select={self.min_select}, max_select={self.max_select}")

        max_select = min(self.max_select or n, n)
        min_select = min(self.min_select, max_select)
        logger.info(f"Select range: {min_select} to {max_select} points")

        selected = np.zeros(n, dtype=bool)
        min_dists = np.full(n, np.inf)
        selected_idx_list = []

        # 处理预先选定的点
        if selected_indices:
            logger.info(f"Processing {len(selected_indices)} pre-selected points")
            pre_points = points[selected_indices]
            dist_matrix = distance.cdist(points, pre_points, metric=self.metric, **self.metric_para)
            min_dists = np.min(dist_matrix, axis=1)
            selected[selected_indices] = True
            selected_idx_list.extend(selected_indices)
            logger.info(f"Pre-selected processed. Max distance: {np.max(min_dists):.4f}")
        else:
            # 随机选择第一个点
            start_idx = np.random.randint(n)
            selected[start_idx] = True
            selected_idx_list.append(start_idx)
            min_dists = distance.cdist(points, [points[start_idx]], metric=self.metric, **self.metric_para)[:, 0]
            logger.info(f"Random start point: index={start_idx}")

        logger.info("Starting main sampling loop...")
        log_interval = max(1, max_select // 10)
        iteration = 0

        # 主采样循环：直到满足最小数量，或者距离低于阈值（但不能超过最大数量）
        while ((len(selected_idx_list) < min_select) or 
               (np.max(min_dists) > self.min_distance)) and (len(selected_idx_list) < max_select):

            candidate_idx = int(np.argmax(min_dists))
            selected[candidate_idx] = True
            selected_idx_list.append(candidate_idx)

            if iteration % log_interval == 0 or iteration < 5:
                max_dist = np.max(min_dists)
                coverage = len(selected_idx_list) / max_select * 100
                logger.info(
                    f"Iter {iteration}: Selected {len(selected_idx_list)}/{max_select} points "
                    f"({coverage:.1f}%), Max dist={max_dist:.4f}, Candidate={candidate_idx}"
                )

            # 更新最小距离数组
            new_dists = distance.cdist(points, [points[candidate_idx]], metric=self.metric, **self.metric_para)[:, 0]
            min_dists = np.minimum(min_dists, new_dists)
            min_dists[selected] = 0.0
            iteration += 1

        logger.info(
            f"FPS completed in {iteration} iterations. "
            f"Selected {len(selected_idx_list)} points. "
            f"Final max distance={np.max(min_dists):.4f} (threshold={self.min_distance})"
        )
        return selected_idx_list


class DescriptorAnalyzer:
    """
    结构描述符计算与降维分析器。
    用于计算原子的 NEP 描述符，执行降维 (PCA/t-SNE/UMAP) 以及在隐空间中采样。
    """
    def __init__(self, model_filename: PathLike, method: str = 'pca',
                 batch_size: int = 1000, store_frames: bool = True):
        self.model_filename = Path(model_filename)
        self.method = method.lower()
        self.batch_size = batch_size
        self.store_frames = store_frames

        self.labels: List[str] = []
        self.natoms_per_frame: List[int] = []
        self.frames_per_file: List[int] = []
        self.frames: List[Any] = []  # 存储 Atoms 对象

        self.descriptors: List[np.ndarray] = []
        self.flat_descriptors: Optional[np.ndarray] = None
        self.structure_indices: Optional[np.ndarray] = None

        self.latent_cache: Dict[str, np.ndarray] = {}
        self.latent_time: float = 0.0

        self.progress_counter: int = 0
        self.total_frames: int = 0

    def add_xyz_file(self, xyz_path: PathLike, label: str) -> None:
        """加载 XYZ 轨迹文件并计算 NEP 描述符。"""
        file_path = Path(xyz_path)
        start_time = time()
        
        frames = read_xyz(file_path)
        n_frames = len(frames)
        self.total_frames += n_frames

        logger.info(f"Processing {n_frames} frames from {file_path.name}")

        if self.store_frames:
            self.frames.extend(frames)
            logger.debug(f"Storing {n_frames} frames in memory")

        descriptors = []
        processed_natoms = [] 
        logger.info(f"Computing descriptors ({n_frames} frames)")
        log_interval = max(1, n_frames // 10)

        for i, atoms in enumerate(frames):
            try:
                # 强健壮性设计：使用 try...except 捕获单个帧的计算错误
                desc = get_descriptors(atoms, str(self.model_filename))
                descriptors.append(desc)
                processed_natoms.append(len(atoms)) 
            except Exception as e:
                logger.warning(f"Skipping frame {i} due to error: {str(e)}")
                continue 
            finally:
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

        self.natoms_per_frame.extend(processed_natoms)
        self.frames_per_file.append(len(descriptors))
        
        self.latent_cache.clear()
        self.flat_descriptors = None

        duration = time() - start_time
        atom_count = sum(processed_natoms)
        processed_frames = len(descriptors)
        fps_speed = processed_frames / duration if duration > 0 else float('inf')
        
        logger.info(
            f"Completed {processed_frames}/{n_frames} frames ({atom_count} atoms) "
            f"in {duration:.1f}s ({fps_speed:.1f} fps)"
        )
        if processed_frames < n_frames:
            logger.warning(f"Skipped {n_frames - processed_frames} frames due to errors")

    def _get_flat_descriptors(self) -> np.ndarray:
        """获取所有原子的原子级展平描述符矩阵。"""
        if self.flat_descriptors is None:
            start_time = time()
            self.flat_descriptors = np.concatenate(self.descriptors)

            self.structure_indices = np.concatenate([
                np.full(len(d), i) for i, d in enumerate(self.descriptors)
            ])
            logger.info(f"Flattened descriptors created in {time() - start_time:.1f}s")
            
        return self.flat_descriptors

    def _get_structure_descriptors(self) -> np.ndarray:
        """获取结构级描述符矩阵 (通过平均原子描述符获得)。"""
        start_time = time()
        struct_desc = np.array([np.mean(d, axis=0) for d in self.descriptors])
        logger.info(f"Structure descriptors computed in {time() - start_time:.1f}s")
        return struct_desc

    def perform_decomposition(self, data: Optional[np.ndarray] = None, force_recompute: bool = False) -> np.ndarray:
        """执行降维分析 (PCA/IPCA/t-SNE/UMAP)。"""
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

        # 延迟加载机器学习库以节省内存和启动时间
        if n_points > 1000000 and self.method == 'pca':
            logger.info(f"Using IncrementalPCA (IPCA) for large dataset")
            from sklearn.decomposition import IncrementalPCA
            
            ipca = IncrementalPCA(n_components=2, batch_size=self.batch_size)
            for i in range(0, n_points, self.batch_size):
                batch = data[i:i + self.batch_size]
                ipca.partial_fit(batch)
                
            points_list = []
            for i in range(0, n_points, self.batch_size):
                batch = data[i:i + self.batch_size]
                points_list.append(ipca.transform(batch))
            points = np.concatenate(points_list)            
            
        else:
            if self.method == 'pca':
                logger.info("Using standard PCA")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                points = pca.fit_transform(data)
                
            elif self.method == 'tsne':
                logger.info("Using t-SNE")
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, method='barnes_hut', 
                            perplexity=min(30, n_points // 3),
                            learning_rate='auto', random_state=42)
                points = tsne.fit_transform(data)
                
            elif self.method == 'umap':
                logger.info("Using UMAP")
                import umap
                reducer = umap.UMAP(n_components=2, n_neighbors=min(15, n_points - 1), random_state=42)
                points = reducer.fit_transform(data)
            else:
                raise ValueError(f"Unknown decomposition method: {self.method}")

        self.latent_cache[cache_key] = points
        self.latent_time = time() - start_time
        logger.info(f"{self.method.upper()} completed in {self.latent_time:.1f}s")
        return points

    def _plot(self, ax: plt.Axes, points: np.ndarray, selected_mask: Optional[np.ndarray] = None,
              level: str = 'atomic', **kwargs) -> None:
        """在给定坐标轴上绘制隐空间投影散点图。"""
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        unique_labels = sorted(set(self.labels))
        label_colors = {label: color_cycle[i % len(color_cycle)] for i, label in enumerate(unique_labels)}

        start_idx = 0
        for label, n_frames in zip(self.labels, self.frames_per_file):
            if level == 'atomic':
                step_size = sum(self.natoms_per_frame[start_idx:start_idx + n_frames])
            else:
                step_size = n_frames
                
            end_idx = start_idx + step_size
            file_points = points[start_idx:end_idx]
            color = label_colors[label]

            ax.scatter(file_points[:, 0], file_points[:, 1], alpha=0.5, label=label, color=color, **kwargs)

            if selected_mask is not None:
                file_selected = selected_mask[start_idx:end_idx]
                if np.any(file_selected):
                    ax.scatter(file_points[file_selected, 0], file_points[file_selected, 1],
                               color='red', edgecolor='black', alpha=0.75, zorder=5)

            start_idx = end_idx

        # 处理图例重复问题
        handles, labels = ax.get_legend_handles_labels()
        unique_labels_sorted = sorted(set(labels), key=labels.index)
        unique_handles = [handles[labels.index(lbl)] for lbl in unique_labels_sorted]

        if unique_handles:
            ax.legend(unique_handles, unique_labels_sorted)

    def _divide_conquer_sampling(self, descriptors: np.ndarray, min_select: int,
                                 max_select: int, min_distance: float) -> np.ndarray:
        """分治采样子程序：用于解决巨大描述符数组下的内存和计算时间瓶颈。"""
        n = len(descriptors)
        chunk_size = 50000
        chunks = [{'start': i, 'end': min(i + chunk_size, n), 'data': descriptors[i:min(i + chunk_size, n)]} 
                  for i in range(0, n, chunk_size)]

        logger.info(f"Divide-and-conquer: {len(chunks)} chunks for {n} points")
        sampled_indices = []

        # 局部采样
        for chunk in chunks:
            chunk_data = chunk['data']
            start_idx = chunk['start']
            chunk_select = min(1000, len(chunk_data), max(1, min_select // 3))

            sampler = FarthestPointSample(min_distance=min_distance, min_select=chunk_select, max_select=chunk_select)
            local_indices = sampler.select(chunk_data)
            sampled_indices.extend([start_idx + idx for idx in local_indices])

        # 全局合并采样
        final_data = descriptors[sampled_indices]
        min_select_final = min(min_select, len(final_data))
        max_select_final = min(max_select or len(final_data), len(final_data))

        logger.info(f"Final sampling on {len(sampled_indices)} pre-sampled points")

        sampler = FarthestPointSample(min_distance=min_distance, min_select=min_select_final, max_select=max_select_final)
        final_local_indices = sampler.select(final_data)
        
        return np.array([sampled_indices[i] for i in final_local_indices])

    def perform_latent_analysis(self, ax: Optional[plt.Axes] = None, min_distance: float = 0.1,
                                min_select: Union[int, float] = 1, max_select: Optional[int] = None,
                                level: str = 'structure', strategy: Optional[str] = None,
                                **kwargs) -> Tuple[Set[int], Set[int]]:
        """执行隐空间投影分析并基于 FPS 选取结构。"""
        start_time = time()
        total_atoms = sum(len(d) for d in self.descriptors)
        logger.info(f"Starting latent analysis (level={level}, atoms={total_atoms})")

        if level == 'structure':
            descriptors = self._get_structure_descriptors()
        else:
            descriptors = self._get_flat_descriptors()
            
        points = self.perform_decomposition(descriptors, force_recompute=False)
        n_points = len(descriptors)
        
        if isinstance(min_select, float):
            min_select = max(1, int(n_points * min_select))
        min_select = min(min_select, n_points)
        max_select = min(max_select or n_points, n_points)

        logger.info(f"Sampling params: min={min_select}, max={max_select}, min_dist={min_distance}")

        if strategy == 'divide':
            logger.warning(f"Large dataset ({n_points} points), using divide-and-conquer strategy")
            selected_idx = self._divide_conquer_sampling(descriptors, min_select, max_select, min_distance)
        elif strategy == 'random':
            logger.warning(f"Large dataset ({n_points} points), using random sampling strategy")
            selected_idx = np.random.choice(n_points, size=min_select, replace=False)
        else:
            logger.info("Using standard FPS sampling")
            sampler = FarthestPointSample(min_distance=min_distance, min_select=min_select, max_select=max_select)
            selected_idx = sampler.select(descriptors)

        if level == 'structure':
            selected_structures = set(selected_idx)
        else:
            selected_structures = set(self.structure_indices[selected_idx])

        all_structures = set(range(len(self.descriptors)))
        unselected_structures = all_structures - selected_structures

        logger.info(f"Selected {len(selected_structures)} structures")

        if ax:
            selected_mask = np.zeros(len(points), dtype=bool)
            if level == 'atomic':
                selected_mask[selected_idx] = True
            else:
                selected_mask[list(selected_structures)] = True

            self._plot(ax, points, selected_mask=selected_mask, level=level, **kwargs)
            ax.set_title(f"{self.method.upper()} Projection ({level} level)\n"
                         f"Selected: {len(selected_structures)} structures")

        logger.info(f"Latent analysis completed in {time() - start_time:.1f}s")
        return selected_structures, unselected_structures

    def split_xyz(self, filename: PathLike, selected_set: Set[int], overwrite: bool = True) -> Path:
        """根据选取结果将 Atoms 结构分离并导出到新的 XYZ 文件。"""
        if not self.store_frames or not self.frames:
            raise AttributeError("Frames not stored (initialize DescriptorAnalyzer with store_frames=True)")

        file_path = Path(filename)
        if overwrite and file_path.exists():
            logger.warning(f"Overwriting existing file: {file_path}")
            file_path.unlink()

        selected_list = sorted(selected_set)
        n_selected = len(selected_list)
        logger.info(f"Exporting {n_selected} structures to {file_path.name}")

        with file_path.open('w', encoding='utf-8') as f:
            for i, idx in enumerate(selected_list):
                if idx < 0 or idx >= len(self.frames):
                    logger.warning(f"Invalid frame index {idx}, skipping")
                    continue
                dump_xyz(f, self.frames[idx])
                if i > 0 and i % 100 == 0:
                    logger.info(f"Exported {i}/{n_selected} structures")

        logger.info(f"Successfully exported {n_selected} structures to {file_path}")
        return file_path


def main():
    """本地 FPS 采样演示与动画可视化。"""
    # 1. 生成随机点云，二维坐标范围[0,1)
    np.random.seed(42)
    n_points = 200
    points = np.random.rand(n_points, 2)

    # 2. 创建FPS采样器实例
    # 注意: 移除了无效参数 use_kdtree=True，维持严谨的距离更新法则
    fps = FarthestPointSample(
        min_distance=0.2,   # 最小采样间距
        min_select=10,      # 最少采样点
        max_select=20       # 最多采样点
    )

    # 3. 执行采样
    selected_indices = fps.select(points, selected_indices=[1, 22])
    print(f"Selected indices: {selected_indices}")

    # 4. 可视化基础设置
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Farthest Point Sampling Process', fontsize=16)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    all_points = ax.scatter(points[:, 0], points[:, 1], c='lightgray', s=15, alpha=0.7)
    selected_points = ax.scatter([], [], c='red', s=60, edgecolor='black', zorder=3)
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 0.5), cmap='viridis_r'), ax=ax, label='Distance to Selected Points')
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top', fontsize=11,
                        bbox=dict(facecolor='white', alpha=0.8))

    # 5. 动画更新逻辑
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

    # 6. 运行动画
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
