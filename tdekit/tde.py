import csv
import hashlib
import json
import numpy as np
import random
import shutil
import subprocess
import sys
import time
from itertools import combinations_with_replacement, product
from math import gcd
from pathlib import Path
from typing import List, Tuple, Union, Optional, Sequence
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

    def __init__(self, base_dir: PathLike = "./",
                 direction: Union[List[float], np.ndarray] = (0, 0, 1), *,
                 ovito_command: Optional[Sequence[str]] = None):
        self.base_dir = Path(base_dir).resolve()
        self.work_dir = self.base_dir / self.CASCADE_DIR
        self.direction = np.asarray(direction, dtype=float)
        self.n_simulation = 0
        if isinstance(ovito_command, (str, bytes)) or (ovito_command is not None and not ovito_command):
            raise ValueError("ovito_command must be a nonempty sequence of command arguments")
        self.ovito_command = tuple(ovito_command) if ovito_command is not None else (sys.executable,)
    
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

    def run_defects_analyzer(self, energy: float, work_dir: Optional[Path] = None,
                             cascade_dir: Optional[str] = None, *,
                             output_dir: Optional[Path] = None) -> None:
        """Analyze a cascade with the configured Python interpreter."""
        wd = work_dir or self.work_dir
        cd = cascade_dir or self.CASCADE_DIR
        script_path = Path(__file__).resolve().with_name("defects_analyzer.py")
        command = [
            *self.ovito_command, str(script_path),
            "--work_dir", str(wd),
            "--cascade_dir", str(wd / cd),
            "--energy", str(energy),
        ]
        if output_dir is not None:
            command.extend(["--output_dir", str(output_dir)])

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[错误] 缺陷分析脚本执行失败: \n{e.stderr}")
            raise

    def read_frenkel_pairs(self, work_dir: Optional[Path] = None) -> int:
        wd = work_dir or self.work_dir
        summary_file = wd / "final_defect_summary.txt"
        lines = summary_file.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) != 2 or lines[0] != "Timestep\tFrenkel_pairs":
            raise ValueError(f"Invalid defect summary: {summary_file}")
        try:
            frame, count = map(int, lines[1].split("\t"))
        except ValueError as error:
            raise ValueError(f"Invalid defect summary: {summary_file}") from error
        if frame < 0 or count < 0:
            raise ValueError(f"Negative frame or defect count: {summary_file}")
        return count

    def run_relax_simulation(self, relax_dir: Optional[PathLike] = None, **kwargs) -> Path:
        rd = Path(relax_dir) if relax_dir else self.base_dir / self.RELAX_DIR
        print("[状态] 开始运行晶格弛豫模拟...")
        run_relax(dirname=rd, **kwargs)
        return rd / "restart.xyz"

    def run_cascade_simulation(self, energy: float, work_dir: Optional[Path] = None, 
                               clean: bool = True, defects_analysis: bool = True, 
                               **kwargs) -> Tuple[int, str]:
        wd = Path(work_dir) if work_dir is not None else self.work_dir
        cascade_dir_name = self.get_unique_dir_id()
        cascade_path = wd / cascade_dir_name
        source = Path(kwargs["input_file"]).resolve() if kwargs.get("input_file") is not None else None
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest() if source is not None else None
        run_cascade(
            dirname=cascade_path, energy=energy, direction=self.direction, **kwargs
        )

        frenkel_pairs = -1
        if defects_analysis:
            self.run_defects_analyzer(energy, wd, cascade_dir_name, output_dir=cascade_path)
            frenkel_pairs = self.read_frenkel_pairs(cascade_path)
        receipt = {
            "case_id": cascade_dir_name,
            "energy_ev": float(energy),
            "direction_cartesian": self.direction.tolist(),
            "mode": kwargs.get("mode", "replace"),
            "input_file": str(source) if source is not None else None,
            "input_sha256": source_hash,
            "prepared_inputs_sha256": {
                name: hashlib.sha256((cascade_path / name).read_bytes()).hexdigest()
                for name in ("model.xyz", "run.in", "nep.txt")
                if (cascade_path / name).is_file()
            },
            "frenkel_pairs": frenkel_pairs if defects_analysis else None,
        }
        with (wd / f"{cascade_dir_name}.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if clean:
            self.clean_directory(cascade_path)
        return frenkel_pairs, cascade_dir_name

    @staticmethod
    def _sample_set(inputs, n_samples):
        if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        if inputs is None:
            return None, set()
        if isinstance(inputs, (str, bytes, Path)):
            raise ValueError("sample_inputs must be a sequence of restart paths")
        paths = tuple(Path(path) for path in inputs)
        if len(paths) != n_samples:
            raise ValueError("sample_inputs must contain exactly n_samples restart files")
        hashes = {hashlib.sha256(path.read_bytes()).digest() for path in paths}
        if len(hashes) != n_samples:
            raise ValueError("sample_inputs must contain distinct restart files")
        return paths, hashes

    def get_fp_probability(self, energy: float, n_samples: int, *,
                           sample_inputs=None, **kwargs) -> float:
        """Return the observed fraction of samples with a Frenkel pair."""
        paths, _ = self._sample_set(sample_inputs, n_samples)
        if not kwargs.get("defects_analysis", True):
            raise ValueError("Frenkel-pair probability requires defect analysis")
        if paths is not None and "input_file" in kwargs:
            raise ValueError("Choose input_file or sample_inputs")
        damage_count = 0
        for index in range(n_samples):
            case_options = kwargs if paths is None else {**kwargs, "input_file": paths[index]}
            count, _ = self.run_cascade_simulation(energy, **case_options)
            if count < 0:
                raise ValueError("Frenkel-pair count is unavailable")
            damage_count += count > 0
        probability = damage_count / n_samples
        print(f"{energy:.6g} eV: {damage_count}/{n_samples} runs formed a Frenkel pair", flush=True)
        return probability

    @staticmethod
    def _energy_interval(low, high):
        if not np.isfinite([low, high]).all() or not 0 <= low <= high:
            raise ValueError("Expected finite energies with 0 <= low <= high")

    @staticmethod
    def _threshold(threshold_prob):
        if not 0 < threshold_prob <= 1:
            raise ValueError("threshold_prob must be in (0, 1]")

    @staticmethod
    def _growth_energies(start, limit):
        energy = start
        while energy < limit:
            yield energy
            energy = min(2 * energy, limit)
        yield limit

    def exponential_growth_search(self, init_energy: float = 2.0,
                                  max_energy: float = 1000.0, **kwargs) -> Tuple[float, float]:
        """Locate the first sampled energy with a positive defect count."""
        self._energy_interval(init_energy, max_energy)
        if init_energy == 0:
            raise ValueError("init_energy must be positive for exponential growth")
        previous = init_energy / 2
        for energy in self._growth_energies(init_energy, max_energy):
            count, _ = self.run_cascade_simulation(energy, **kwargs)
            if count < 0:
                raise ValueError("Frenkel-pair count is unavailable")
            if count > 0:
                return previous, energy
            previous = energy
        raise RuntimeError(f"No sampled damage through {max_energy:g} eV")

    @staticmethod
    def _bisect(low, low_prob, high, high_prob, precision, threshold, probability):
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError("precision must be positive and finite")
        if not low_prob < threshold <= high_prob:
            raise RuntimeError("Observed probabilities do not bracket the threshold")
        while high - low > precision:
            midpoint = low + (high - low) / 2
            if not low < midpoint < high:
                raise RuntimeError("Requested precision is below floating-point resolution")
            measured = probability(midpoint)
            if measured >= threshold:
                high, high_prob = midpoint, measured
            else:
                low, low_prob = midpoint, measured
        return low + (high - low) / 2, low, low_prob, high, high_prob

    def binary_search(self, low_energy: float, high_energy: float,
                      precision: float = 1.0, num_samples: int = 10,
                      threshold_prob: float = 0.5, *,
                      max_energy: Optional[float] = None, **kwargs) -> Tuple[float, float, float, float, float]:
        """Bisect a probability bracket with a finite expansion limit."""
        self._energy_interval(low_energy, high_energy)
        limit = max(high_energy, 1000.0) if max_energy is None else max_energy
        self._energy_interval(high_energy, limit)
        self._threshold(threshold_prob)
        if high_energy == 0 or not np.isfinite(precision) or precision <= 0:
            raise ValueError("high_energy and precision must be positive and finite")

        def probability(energy):
            return self.get_fp_probability(energy, num_samples, **kwargs)

        low_prob = probability(low_energy)
        if low_prob >= threshold_prob and low_energy > 0:
            high_energy, high_prob = low_energy, low_prob
            low_energy, low_prob = 0.0, probability(0.0)
        else:
            for energy in self._growth_energies(high_energy, limit):
                high_prob = probability(energy)
                if high_prob >= threshold_prob:
                    high_energy = energy
                    break
                low_energy, low_prob = energy, high_prob
            else:
                raise RuntimeError(f"No observed threshold crossing through {limit:g} eV")
        return self._bisect(low_energy, low_prob, high_energy, high_prob,
                            precision, threshold_prob, probability)

    def incremental_search(self, low_energy: float, high_energy: float,
                           step: float = 2.0, num_samples: int = 10,
                           threshold_prob: float = 0.5, **kwargs) -> Tuple[Optional[float], float]:
        """Scan in increasing energy, including the exact upper limit."""
        self._energy_interval(low_energy, high_energy)
        self._threshold(threshold_prob)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be positive and finite")
        energy = low_energy
        while True:
            probability = self.get_fp_probability(energy, num_samples, **kwargs)
            if probability >= threshold_prob:
                return energy, probability
            if energy == high_energy:
                return None, 0.0
            next_energy = min(energy + step, high_energy)
            if next_energy == energy:
                raise RuntimeError("Requested step is below floating-point resolution")
            energy = next_energy

    def find_tde(self, init_energy: float = 2.0, max_energy: float = 1000.0,
                 precision: float = 1.0, num_samples: int = 10, verification_samples: int = 10,
                 threshold_prob: float = 0.5, *, sample_inputs=None,
                 verification_inputs=None, **kwargs) -> Tuple[float, float]:
        """Search for a probability crossing and verify the reported energy."""
        self._energy_interval(init_energy, max_energy)
        self._threshold(threshold_prob)
        if init_energy == 0 or not np.isfinite(precision) or precision <= 0:
            raise ValueError("init_energy and precision must be positive and finite")
        search_paths, search_hashes = self._sample_set(sample_inputs, num_samples)
        verify_paths, verify_hashes = self._sample_set(verification_inputs, verification_samples)
        if (search_paths is None) != (verify_paths is None) or search_hashes & verify_hashes:
            raise ValueError("Provide disjoint search and verification restart sets together")
        if search_paths is not None and "input_file" in kwargs:
            raise ValueError("Choose input_file or explicit restart sets")

        def probability(energy):
            return self.get_fp_probability(energy, num_samples, sample_inputs=search_paths, **kwargs)

        low, low_prob = None, None
        for high in self._growth_energies(init_energy, max_energy):
            high_prob = probability(high)
            if high_prob >= threshold_prob:
                break
            low, low_prob = high, high_prob
        else:
            raise RuntimeError(f"No observed threshold crossing through {max_energy:g} eV")
        if low is None:
            low, low_prob = 0.0, probability(0.0)
        midpoint, _, _, high, _ = self._bisect(
            low, low_prob, high, high_prob, precision, threshold_prob, probability
        )

        for energy in dict.fromkeys((midpoint, high)):
            observed = self.get_fp_probability(energy, verification_samples,
                                               sample_inputs=verify_paths, **kwargs)
            if observed >= threshold_prob:
                return energy, observed
        raise RuntimeError("Neither the midpoint nor the upper bracket passed verification")
