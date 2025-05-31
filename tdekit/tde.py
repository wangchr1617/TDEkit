from pathlib import Path
import argparse
import shutil
import subprocess
import numpy as np
from .core import run_cascade, run_relax

# 配置常量
INIT_ENERGY = 2.0
MAX_ENERGY = 1000.0
PRECISION = 0.01
CASCADE_DIR_NAME = "cascade"
DEFECTS_SCRIPT = "defects_analyzer.py"

def clean_directory(dir_path):
    """清理指定目录"""
    if dir_path.exists():
        print(f"Cleaning directory: {dir_path}")
        shutil.rmtree(dir_path)

def run_defects_analyzer(base_dir, energy):
    """运行缺陷分析脚本"""
    script_path = Path(__file__).resolve().parent / DEFECTS_SCRIPT
    command = [
        "apptainer", "exec", 
        "/opt/software/ovito-3.12.3/bin/ovito_python.sif",
        "python", str(script_path), 
        "--base_dir", str(base_dir), 
        "--energy", str(energy)
    ]
    subprocess.run(command, check=True)

def read_frenkel_pairs(base_dir):
    """从结果文件中读取Frenkel对数量"""
    summary_file = base_dir / "final_defect_summary.txt"
    if not summary_file.exists():
        return 0
    
    with summary_file.open() as f:
        last_line = f.readlines()[-1].strip()
        return int(last_line.split("\t")[-1])

def run_cascade_simulation(base_dir, energy, input_file):
    """运行级联模拟和分析"""
    cascade_dir = base_dir / CASCADE_DIR_NAME
    
    # 清理并运行级联模拟
    clean_directory(cascade_dir)
    run_cascade(
        input_file=input_file, 
        energy=energy, 
        direction=np.array([0, 0, 1])
    )
    
    # 运行缺陷分析
    run_defects_analyzer(base_dir, energy)
    return read_frenkel_pairs(base_dir)

def exponential_growth_search(base_dir, input_file):
    """指数增长搜索阶段"""
    print("\nStarting exponential growth search...")
    current_energy = INIT_ENERGY
    high_energy = None
    
    while current_energy <= MAX_ENERGY:
        fp_count = run_cascade_simulation(base_dir, current_energy, input_file)
        print(f"Energy: {current_energy:.2f}eV -> Frenkel pairs: {fp_count}")
        
        if fp_count > 0:
            high_energy = current_energy
            print(f"First damage found at {high_energy:.2f}eV")
            break
        
        current_energy *= 2
    
    if high_energy is None:
        raise RuntimeError(f"No damage found below {MAX_ENERGY}eV")
    
    return high_energy, current_energy / 2

def binary_search(base_dir, input_file, low, high):
    """二分搜索阶段"""
    print("\nStarting binary search...")
    iteration = 0
    max_iterations = 100
    while abs(high - low) > PRECISION and iteration < max_iterations:
        mid_energy = (low + high) / 2
        fp_count = run_cascade_simulation(base_dir, mid_energy, input_file)
        print(f"Energy: {mid_energy:.2f}eV -> Frenkel pairs: {fp_count}")
        if fp_count > 0:
            high = mid_energy
        else:
            low = mid_energy
        iteration += 1
    if iteration >= max_iterations:
        print(f"Warning: Maximum iterations ({max_iterations}) reached")
        print(f"Final values: Low={low:.3f}eV, High={high:.3f}eV")
    return high

def find_tde(base_dir):
    """查找位移能量阈值(TDE)"""
    # 准备文件路径
    base_dir = Path(base_dir)
    xyz_path = base_dir / "model.xyz"
    relax_path = base_dir / "relax" / "restart.xyz"
    cascade_dir = base_dir / CASCADE_DIR_NAME
    
    # 运行松弛计算（如果需需要）
    if not relax_path.exists():
        print("Running relaxation calculation...")
        run_relax(
            input_file=xyz_path, 
            nx=15, ny=9, nz=2, 
            thickness=7
        )
    else:
        print("Using existing relaxation results")
    
    # 搜索过程
    damage_energy, no_damage_energy = exponential_growth_search(base_dir, relax_path)
    tde = binary_search(base_dir, relax_path, no_damage_energy, damage_energy)
    
    # 清理并打印结果
    clean_directory(cascade_dir)
    print(f"\nSearch completed! TDE: {tde:.2f}eV")
    return tde

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Threshold Displacement Energy (TDE)")
    parser.add_argument("--base_dir", required=True, help="Base directory of simulation data")
    args = parser.parse_args()
    
    find_tde(Path(args.base_dir))
    
