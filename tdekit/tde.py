import csv
import hashlib
import numpy as np
import random
import shutil
import subprocess
import time
from itertools import combinations_with_replacement, product
from math import gcd
from pathlib import Path
from typing import List, Tuple, Union, Optional
from .core import run_cascade, run_relax

# 定义通用的路径类型
PathLike = Union[str, Path]

def generate_crystal_directions(hmax: int) -> List[Tuple[int, int, int]]:
    """生成最大指数为 hmax 的所有独立的晶体学方向 (米勒指数)。"""
    directions_set = set()
    indices = list(range(-hmax, hmax + 1))
    
    for h, k, l in product(indices, indices, indices):
        if h == 0 and k == 0 and l == 0:
            continue
        g = gcd(gcd(abs(h), abs(k)), abs(l))
        if g == 0: 
            continue
            
        h_norm, k_norm, l_norm = h // g, k // g, l // g
        
        # 规约到对称等效的独立区间
        if l_norm < 0:
            h_norm, k_norm, l_norm = -h_norm, -k_norm, -l_norm
        elif l_norm == 0 and k_norm < 0:
            h_norm, k_norm, l_norm = -h_norm, -k_norm, -l_norm
        elif l_norm == 0 and k_norm == 0 and h_norm < 0:
            h_norm, k_norm, l_norm = -h_norm, -k_norm, -l_norm
            
        directions_set.add((h_norm, k_norm, l_norm))
        
    return sorted(list(directions_set))

def calculate_angles(h: float, k: float, l: float) -> Tuple[float, float]:
    """将方向向量转换为球坐标系下的角度 (theta, phi)，单位为度。"""
    r = np.sqrt(h**2 + k**2 + l**2)
    theta_rad = np.arccos(l / r)
    theta_deg = float(np.degrees(theta_rad))
    
    phi_rad = np.arctan2(k, h)
    phi_deg = float(np.degrees(phi_rad))
    if phi_deg < 0:
        phi_deg += 360.0
        
    return theta_deg, phi_deg

def generate_direction_table(hmax: int) -> List[Tuple[int, int, int, float, float]]:
    """生成包含 (h, k, l, theta, phi) 的晶向表。"""
    directions = generate_crystal_directions(hmax)
    table = []
    
    for d in directions:
        h, k, l = d
        if h == 0 and k == 0 and l == 0:
            continue
        theta, phi = calculate_angles(h, k, l)
        # 限制在第一卦限及有效角度范围内
        if 0 <= theta <= 90 and 0 <= phi < 360:
            table.append((h, k, l, theta, phi))
            
    table.sort(key=lambda x: (x[3], x[4]))
    return table

def generate_uniform_directions(n: int) -> List[Tuple[float, float, float]]:
    """在半球面上生成 n 个近似均匀分布的方向向量 (基于斐波那契网格)。"""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    mask = z >= 0
    x, y, z = x[mask], y[mask], z[mask]
    
    return [(float(x[i]), float(y[i]), float(z[i])) for i in range(len(x))]

def generate_uniform_direction_table(n: int) -> List[Tuple[float, float, float, float, float]]:
    """生成均匀分布的 (x, y, z, theta, phi) 方向表。"""
    directions = generate_uniform_directions(n)
    table = []    
    
    for d in directions:
        x, y, z = d
        theta, phi = calculate_angles(x, y, z)
        if 0 <= theta <= 90 and 0 <= phi < 360:
            table.append((x, y, z, theta, phi))
            
    table.sort(key=lambda x: (x[3], x[4]))
    return table

def save_direction_table_to_csv(table: List[Tuple], filename: PathLike) -> None:
    """将方向表保存为 CSV 文件。"""
    file_path = Path(filename)
    with file_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if table and len(table[0]) == 5:
            if isinstance(table[0][0], int):
                writer.writerow(['h', 'k', 'l', 'theta', 'phi'])
            else:
                writer.writerow(['x', 'y', 'z', 'theta', 'phi'])
        for row in table:
            writer.writerow(row)

def read_direction_table_from_csv(filename: PathLike) -> List[Tuple]:
    """从 CSV 文件读取方向表。"""
    file_path = Path(filename)
    table = []
    with file_path.open('r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) == 5:
                try:
                    if '.' not in row[0]:
                        table.append((int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4])))
                        continue
                except ValueError:
                    pass
                table.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))
    return table

class TDESearch:
    """执行阈值位移能 (TDE) 搜索的自动化工作流控制器。"""
    
    RELAX_DIR = "relax"
    CASCADE_DIR = "cascade"

    def __init__(self, base_dir: PathLike = "./", direction: Union[List[float], np.ndarray] = np.array([0, 0, 1])):
        self.base_dir = Path(base_dir).resolve()
        self.work_dir = self.base_dir / self.CASCADE_DIR
        self.direction = np.asarray(direction, dtype=float)
        self.n_simulation = 0
    
    def set_work_dir(self, work_dir: PathLike) -> None:
        self.work_dir = Path(work_dir).resolve()
        print(f"[配置] 工作目录设定为: {self.work_dir}")
    
    def set_direction(self, direction: Union[List[float], np.ndarray]) -> None:
        self.direction = np.asarray(direction, dtype=float)
        print(f"[配置] PKA 方向设定为: {self.direction}")

    def clean_directory(self, directory: PathLike) -> None:
        dir_path = Path(directory)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)

    def get_unique_dir_id(self) -> str:
        self.n_simulation += 1
        dir_name = f"cascade_id{self.n_simulation:04d}"        
        hash_input = f"{dir_name}_{time.time()}_{random.randint(0, 1000000)}"
        unique_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{dir_name}_{unique_hash}"

    def run_defects_analyzer(self, energy: float, work_dir: Optional[Path] = None, cascade_dir: Optional[str] = None) -> None:
        """调用 Apptainer 运行 ovito_python 分析缺陷。"""
        wd = work_dir or self.work_dir
        cd = cascade_dir or self.CASCADE_DIR
        script_path = Path(__file__).resolve().parent / "defects_analyzer.py"
        
        container_path = Path("/opt/software/ovito-3.12.3/bin/ovito_python.sif")
        if not container_path.exists():
            print(f"[警告] 未找到 Apptainer 镜像: {container_path}，这可能导致缺陷分析失败。")

        command = [
            "apptainer", "exec", str(container_path),
            "python", str(script_path),
            "--work_dir", str(wd),
            "--cascade_dir", str(wd / cd),
            "--energy", str(energy)
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[错误] 缺陷分析脚本执行失败: \n{e.stderr}")
            raise

    def read_frenkel_pairs(self, work_dir: Optional[Path] = None) -> int:
        wd = work_dir or self.work_dir
        summary_file = wd / "final_defect_summary.txt"
        
        if not summary_file.exists():
            print(f"[警告] 缺陷摘要文件不存在: {summary_file}")
            return 0
            
        with summary_file.open('r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines or len(lines) < 2:  # 只有表头或为空
                print("[警告] 缺陷摘要文件为空或格式错误")
                return 0
                
            last_line = lines[-1]
            try:
                fp_count = int(last_line.split("\t")[-1])
                return fp_count
            except (ValueError, IndexError):
                print(f"[警告] 无法解析 Frenkel pair 数量: {last_line}")
                return 0
            
    def run_relax_simulation(self, relax_dir: Optional[PathLike] = None, **kwargs) -> Path:
        rd = Path(relax_dir) if relax_dir else self.base_dir / self.RELAX_DIR
        print("[状态] 开始运行晶格弛豫模拟...")
        run_relax(dirname=rd, **kwargs)
        return rd / "restart.xyz"

    def run_cascade_simulation(self, energy: float, work_dir: Optional[Path] = None, 
                               clean: bool = True, defects_analysis: bool = True, 
                               **kwargs) -> Tuple[int, str]:
        wd = work_dir or self.work_dir
        cascade_dir_name = self.get_unique_dir_id()
        cascade_path = wd / cascade_dir_name
        
        # print(f"[级联] 能量 {energy:.3f} eV, 目录: {cascade_dir_name}")

        run_cascade(
            dirname=cascade_path,
            energy=energy,
            direction=self.direction,
            **kwargs
        )
        
        frenkel_pairs = -1
        if defects_analysis:
            self.run_defects_analyzer(energy, wd, cascade_dir_name)
            frenkel_pairs = self.read_frenkel_pairs(wd)
            
        if clean:
            self.clean_directory(cascade_path)
            
        return frenkel_pairs, cascade_dir_name

    def get_fp_probability(self, energy: float, n_samples: int, **kwargs) -> float:
        """在特定能量下多次采样，计算产生 Frenkel pair 的概率。"""
        print(f"\n[统计] 采样能量: {energy:.3f} eV | 样本总数: {n_samples}")
        fp_count = 0
        for i in range(n_samples):
            count, _ = self.run_cascade_simulation(energy, **kwargs)
            if count > 0:
                fp_count += 1
                # print(f"  样本 {i+1}/{n_samples}: 发现损伤 ({count} FP)")
        
        probability = fp_count / n_samples
        print(f"[统计结果] {energy:.3f} eV -> 损伤概率 = {probability:.2f} ({fp_count}/{n_samples})")
        return probability

    def exponential_growth_search(self, init_energy: float = 2.0, max_energy: float = 1000.0, **kwargs) -> Tuple[float, float]:
        """指数增长法粗略定位损伤发生能量区间。"""
        print("\n=== 开始指数增长法区间搜索 ===")
        current_energy = init_energy
        
        while current_energy <= max_energy:
            print(f"-> 测试能量: {current_energy:.2f} eV")
            fp_count, _ = self.run_cascade_simulation(current_energy, **kwargs)
            
            if fp_count > 0:
                high_energy = current_energy
                low_energy = current_energy / 2.0
                print(f"[区间定位] 在 {high_energy:.2f} eV 处首次发现损伤。")
                print(f"[区间定位] 损伤边界粗定为: [{low_energy:.2f}, {high_energy:.2f}] eV")
                return low_energy, high_energy
                
            current_energy *= 2.0
            
        raise RuntimeError(f"在最高设定能量 {max_energy} eV 下仍未检测到损伤！")

    def binary_search(self, low_energy: float, high_energy: float, 
                      precision: float = 1.0, num_samples: int = 10, 
                      threshold_prob: float = 0.5, **kwargs) -> Tuple[float, float, float, float, float]:
        """基于概率统计的二分查找算法精确定位 TDE。"""
        print("\n=== 开始二分统计概率搜索 ===")
        print(f"初始边界: [{low_energy:.3f}, {high_energy:.3f}] eV")
        print(f"参数设定: 概率阈值 {threshold_prob}, 样本数 {num_samples}/点, 精度 < {precision} eV")
        
        iteration = 0
        max_iterations = 50                
        
        print("\n[初始化] 验证初始边界概率...")
        low_prob = self.get_fp_probability(low_energy, num_samples, **kwargs)
        high_prob = self.get_fp_probability(high_energy, num_samples, **kwargs)
        
        # 边界自适应调整
        while low_prob >= threshold_prob and iteration < max_iterations:
            print(f"[边界修正] 下界 {low_energy:.3f} eV 概率 ({low_prob:.2f}) 过高。向下拓展。")
            high_energy, high_prob = low_energy, low_prob
            low_energy /= 2.0
            low_prob = self.get_fp_probability(low_energy, num_samples, **kwargs)
            iteration += 1
            
        while high_prob < threshold_prob and iteration < max_iterations:
            print(f"[边界修正] 上界 {high_energy:.3f} eV 概率 ({high_prob:.2f}) 过低。向上拓展。")
            low_energy, low_prob = high_energy, high_prob
            high_energy *= 2.0
            high_prob = self.get_fp_probability(high_energy, num_samples, **kwargs)
            iteration += 1
        
        # 核心二分循环
        while (high_energy - low_energy) > precision and iteration < max_iterations:
            iteration += 1
            mid_energy = (low_energy + high_energy) / 2.0
            
            print(f"\n--- 二分迭代 {iteration} ---")
            print(f"当前区间: [{low_energy:.3f}, {high_energy:.3f}] eV (区间宽度: {high_energy - low_energy:.3f} eV)")
            
            prob = self.get_fp_probability(mid_energy, num_samples, **kwargs)
            
            if prob >= threshold_prob:
                high_energy, high_prob = mid_energy, prob
                print(f"-> 概率 {prob:.2f} >= 阈值，收紧上界至 {high_energy:.3f} eV")
            else:
                low_energy, low_prob = mid_energy, prob
                print(f"-> 概率 {prob:.2f} < 阈值，收紧下界至 {low_energy:.3f} eV")
        
        if iteration >= max_iterations:
            print(f"\n[警告] 达到最大迭代次数 ({max_iterations})，搜索提前终止。")
        
        final_estimate = (low_energy + high_energy) / 2.0
        print(f"\n[二分搜索完成] 迭代次数: {iteration}")
        
        return final_estimate, low_energy, low_prob, high_energy, high_prob
        
    def incremental_search(self, low_energy: float, high_energy: float, 
                           step: float = 2.0, num_samples: int = 10, 
                           threshold_prob: float = 0.5, **kwargs) -> Tuple[Optional[float], float]:
        """线性增量搜索方法 (作为二分法的替代)。"""
        print("\n=== 开始增量线性搜索 ===")
        print(f"搜索区间: [{low_energy:.3f}, {high_energy:.3f}] eV, 步长: {step} eV")
        
        current_energy = low_energy        
        iteration = 0
        
        while current_energy <= high_energy:
            iteration += 1
            prob = self.get_fp_probability(current_energy, num_samples, **kwargs)
            
            if prob >= threshold_prob:
                print(f"\n[增量搜索完成] 在 {current_energy:.3f} eV 处跨越概率阈值 ({prob:.2f})")
                return current_energy, prob
                
            current_energy += step
        
        print(f"\n[警告] 在 {high_energy:.3f} eV 之前未发现超过阈值概率的 TDE。")
        return None, 0.0

    def find_tde(self, init_energy: float = 2.0, max_energy: float = 1000.0, 
                 precision: float = 1.0, num_samples: int = 10, verification_samples: int = 10, 
                 threshold_prob: float = 0.5, **kwargs) -> Tuple[float, float]:
        """执行端到端的 TDE 搜索主控工作流。"""
        print("\n" + "="*60)
        print("启动 TDE 自动化搜索流程")
        print("="*60)
        
        # 1. 指数增长找到粗边界
        no_damage_energy, damage_energy = self.exponential_growth_search(init_energy, max_energy, **kwargs)
        
        # 2. 二分法统计确认精确值
        tde_estimate, low_eng, low_prob, high_eng, high_prob = self.binary_search(
            no_damage_energy, damage_energy, precision, 
            num_samples=num_samples, threshold_prob=threshold_prob, **kwargs
        )
        
        # 3. 最终验证采样
        print(f"\n=== 执行最终验证采样 (测试能量: {tde_estimate:.3f} eV) ===")
        verified_prob = self.get_fp_probability(tde_estimate, verification_samples, **kwargs)
        
        if verified_prob < threshold_prob:
            print(f"[验证修正] 估计值概率 ({verified_prob:.2f}) 未达标，回退至安全上界 {high_eng:.3f} eV")
            tde_estimate = high_eng
            verified_prob = high_prob
        
        print("\n" + "="*60)
        print("TDE 搜索流程结束")
        print("="*60)
        print(f"方向 <{self.direction}> 下的估计 TDE: {tde_estimate:.3f} ± {precision:.3f} eV")
        print(f"在该能量下，Frenkel Pair 形成的概率为: {verified_prob:.2f}")
        print("="*60)
        
        return tde_estimate, verified_prob
