import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
from typing import List, Optional, Union, Any, Tuple

from ase import Atoms
from .core import read_xyz, dump_xyz

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义通用的路径类型
PathLike = Union[str, Path]

class ForceAnalyzer:
    """
    计算并分析由于不同势函数（Committee Models）引起的原子受力差异 (Max Force Delta)。
    常用于主动学习中筛选具有高不确定性的构型。
    """
    def __init__(self, 
                 calculators: List[Any], 
                 frame_paths: List[PathLike], 
                 frame_labels: List[str], 
                 minimum: float = 0.25, 
                 maximum: float = 0.5, 
                 bin_edges: Optional[np.ndarray] = None, 
                 load_max_delta: bool = False, 
                 max_delta_dir: PathLike = 'max_deltas', 
                 xyz_dir: PathLike = 'xyzs'):
        self.calculators = calculators
        self.frame_paths = [Path(p) for p in frame_paths]
        self.frame_labels = frame_labels
        self.minimum = minimum
        self.maximum = maximum
        # 默认使用 0 到 0.75 划分为 14 个区间的 bin
        self.bin_edges = bin_edges if bin_edges is not None else np.linspace(0, 0.75, 15)
        self.load_max_delta = load_max_delta
        self.max_delta_dir = Path(max_delta_dir)
        self.xyz_dir = Path(xyz_dir)

    def compute_max_deltas(self, frames: List[Atoms], path: Path, label: str) -> np.ndarray:
        """调用各个 calculator 计算构型受力，提取原子的最大受力偏差。"""
        logger.info(f"[COMPUTE] Starting max_delta computation for {label} ({len(frames)} frames)")
        if not frames:
            logger.warning("[COMPLETE] Empty frame list - nothing to process")
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
                    logger.warning(f"Frame {idx}: Force array shape mismatch ({len(forces)} vs {n_atoms})")
                    forces = np.zeros((n_atoms, 3))
                force_matrix[calc_idx] = forces
            
            # 计算受力偏差: 距离均值的最大范数
            mean_force = np.mean(force_matrix, axis=0)
            deltas = force_matrix - mean_force
            delta_norms = np.linalg.norm(deltas, axis=2)
            max_deltas[idx] = np.max(delta_norms)
            
            # 进度报告
            current_time = time()
            if current_time - last_report_time > 60 or idx % 100 == 0 or idx == len(frames) - 1:
                elapsed = current_time - start_time
                speed = (idx + 1) / elapsed if elapsed > 0 else float('inf')
                remaining = (len(frames) - idx - 1) / speed if speed > 0 else 0
                logger.info(
                    f"[{path.name}] {idx+1}/{len(frames)} "
                    f"({(idx+1)/len(frames)*100:.1f}%) | Speed: {speed:.1f} fps | "
                    f"ETA: {remaining:.0f}s | Current max_delta: {max_deltas[idx]:.6f}"
                )
                last_report_time = current_time
        
        total_time = time() - start_time
        atom_info = f" | Atoms varied: {len(frames[0])}→{len(frames[-1])}" if frames else ""
        
        logger.info(f"[COMPLETE] Finished {label} in {total_time:.1f}s "
                    f"({total_time/len(frames):.4f}s/frame){atom_info}")
        
        self.save_max_deltas_to_txt(max_deltas, path, label)
        return max_deltas

    def save_max_deltas_to_txt(self, max_deltas: np.ndarray, path: Path, label: str) -> None:
        """使用 numpy 的 safetxt 保存一维的最大偏差数组。"""
        self.max_delta_dir.mkdir(parents=True, exist_ok=True)
        base_name = path.name.replace('.', '_')
        filename = self.max_delta_dir / f"{base_name}_{label}_max_deltas.txt"
        
        with filename.open('w', encoding='utf-8') as f:
            f.write(f"# Path: {path}\n")
            f.write(f"# Label: {label}\n")
            f.write(f"# Num_frames: {len(max_deltas)}\n")
            np.savetxt(f, max_deltas, fmt='%.6f')
            
        logger.info(f"Saved max_deltas to: {filename}")

    def load_max_deltas_from_txt(self, path: Path, label: str) -> np.ndarray:
        """从文件中加载一维的最大偏差数组。"""
        base_name = path.name.replace('.', '_')
        filename = self.max_delta_dir / f"{base_name}_{label}_max_deltas.txt"
        
        if not filename.exists():
            raise FileNotFoundError(f"Cannot find max deltas file: {filename}")
            
        # np.loadtxt 自动忽略以 '#' 开头的注释行
        return np.loadtxt(filename)

    def compute_frequency(self, max_deltas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """根据分箱边缘计算分布直方图的频率（百分比）。"""
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        bin_centers = np.arange(self.bin_edges[0] + bin_width / 2, self.bin_edges[-1] + bin_width / 2, bin_width)
        
        counts, _ = np.histogram(max_deltas, bins=self.bin_edges)
        frequencies = counts / len(max_deltas) * 100
        return bin_centers, frequencies

    def split_xyz(self, frames: List[Atoms], max_deltas: np.ndarray, 
                  path: Path, label: str, if_unsel: bool = True) -> None:
        """根据受力偏差的不确定性范围，分离并导出被选定和未选定的构型。"""
        logger.info(f"[SPLIT] Starting XYZ splitting for {label}")
        self.xyz_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = path.name.replace('.', '_')
        selected_filename = self.xyz_dir / f"{base_name}_{label}_selected.xyz"
        unselected_filename = self.xyz_dir / f"{base_name}_{label}_unselected.xyz"

        # 修复: 直接使用 'w' 模式打开以覆写原有内容，防止多次运行导致帧追加重复
        with selected_filename.open('w', encoding='utf-8') as f_sel:
            if if_unsel:
                with unselected_filename.open('w', encoding='utf-8') as f_unsel:
                    for max_delta, atoms in zip(max_deltas, frames):
                        if self.minimum <= max_delta < self.maximum:
                            dump_xyz(f_sel, atoms)
                        else:
                            dump_xyz(f_unsel, atoms)
            else:
                for max_delta, atoms in zip(max_deltas, frames):
                    if self.minimum <= max_delta < self.maximum:
                        dump_xyz(f_sel, atoms)

    def plot_max_force_differences(self, ax: plt.Axes, if_split: bool = False, if_unsel: bool = True) -> None:
        """主入口：计算/加载误差，绘制频率分布曲线，并可选择是否分割导出 XYZ。"""
        for path, label in zip(self.frame_paths, self.frame_labels):
            logger.info(f"Processing path: {path} for label: {label}")
            frames = read_xyz(path)
            logger.info(f"Read {len(frames)} frames from {path}")
            
            if self.load_max_delta:
                max_deltas = self.load_max_deltas_from_txt(path, label)
            else:
                max_deltas = self.compute_max_deltas(frames, path, label)
                
            if if_split:
                self.split_xyz(frames, max_deltas, path, label, if_unsel)
                
            bin_centers, frequencies = self.compute_frequency(max_deltas)
            ax.plot(bin_centers, frequencies, label=label, marker='o', markersize=4)
            
        ax.axvline(x=self.minimum, color='k', linestyle='--')
        ax.axvline(x=self.maximum, color='k', linestyle='--')
        ax.axvspan(xmin=self.minimum, xmax=self.maximum, facecolor="#E5A79A", alpha=0.5, zorder=0)