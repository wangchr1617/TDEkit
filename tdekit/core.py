import random
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from ase import Atoms
from ase.io import read

# 定义通用的路径类型提示
PathLike = Union[str, Path]

def write_run(parameters: List[str], filename: PathLike = 'run.in') -> None:
    """写入 GPUMD 的 run.in 配置文件。"""
    file_path = Path(filename)
    with file_path.open('w', encoding='utf-8') as f:
        for param in parameters:
            f.write(f"{param}\n")

def parsed_properties(comment: str) -> Dict[str, slice]:
    """解析 Extended XYZ 格式的 properties 字符串。"""
    try:
        properties_str = comment.split('properties=')[1].split()[0]
    except IndexError:
        return {}
    
    properties = properties_str.split(':')
    parsed_props = {}
    start = 0
    for i in range(0, len(properties), 3):
        property_name = properties[i]
        property_count = int(properties[i+2])
        parsed_props[property_name] = slice(start, start + property_count)
        start += property_count
    return parsed_props

def read_symbols(words_in_line: List[str], parsed_props: Dict[str, slice]) -> str:
    symbol_slice = parsed_props['species']
    symbol = words_in_line[symbol_slice][0]
    return symbol.lower().capitalize()

def read_positions(words_in_line: List[str], parsed_props: Dict[str, slice]) -> List[float]:
    pos_slice = parsed_props['pos']
    pos = words_in_line[pos_slice]
    return [float(pos[0]), float(pos[1]), float(pos[2])]

def read_mass(words_in_line: List[str], parsed_props: Dict[str, slice]) -> Optional[float]:
    if 'mass' in parsed_props:
        mass_slice = parsed_props['mass']
        return float(words_in_line[mass_slice][0])
    return None

def read_force(words_in_line: List[str], parsed_props: Dict[str, slice]) -> Optional[List[float]]:
    force_key = 'forces' if 'forces' in parsed_props else 'force'
    if force_key in parsed_props:
        force_slice = parsed_props[force_key]
        force = words_in_line[force_slice]
        return [float(force[0]), float(force[1]), float(force[2])]
    return None
    
def read_group(words_in_line: List[str], parsed_props: Dict[str, slice]) -> Optional[int]:
    if 'group' in parsed_props:
        group_slice = parsed_props['group']
        return int(words_in_line[group_slice][0])
    return None

def read_velocity(words_in_line: List[str], parsed_props: Dict[str, slice]) -> Optional[List[float]]:
    if 'vel' in parsed_props:
        vel_slice = parsed_props['vel']
        vel = words_in_line[vel_slice]
        return [float(vel[0]), float(vel[1]), float(vel[2])]
    return None

def read_xyz(filename: PathLike) -> List[Atoms]:
    """读取扩展 XYZ 轨迹文件，返回包含 ASE Atoms 对象的列表。"""
    file_path = Path(filename)
    frames = []
    
    with file_path.open('r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
                
            natoms = int(line)
            comment = f.readline().lower().strip()
            
            # 解析 PBC
            if "pbc=\"" in comment:
                pbc_str = comment.split("pbc=\"")[1].split("\"")[0].strip()
                pbc = [pbc_value == "t" for pbc_value in pbc_str.split()]
            else:
                pbc = [True, True, True]
                
            # 解析晶胞 Lattice
            if "lattice=\"" in comment:
                lattice_str = comment.split("lattice=\"")[1].split("\"")[0].strip()
                values = [val for val in lattice_str.split() if val != '']
                if len(values) == 9:
                    floats = list(map(float, values))
                    cell = [floats[0:3], floats[3:6], floats[6:9]]
                else:
                    print(f"Warning: Invalid lattice format: {lattice_str}", flush=True)
                    cell = np.eye(3).tolist()
            else:
                cell = np.eye(3).tolist()
                
            # 解析能量 Energy
            energy = None
            if "energy=" in comment:
                try:
                    energy = float(comment.split("energy=")[1].split()[0])
                except ValueError:
                    pass
                    
            # 解析 Virial / Stress
            stress = None
            if "virial=" in comment:
                virials_str = comment.split("virial=\"")[1].split("\"")[0].strip()
                virials = np.array([float(x) for x in virials_str.split()]).reshape(3, 3)
                # 计算应力张量
                cell_det = np.linalg.det(cell)
                stress = -virials / cell_det if cell_det != 0 else None
                
            config_type = None
            if "config_type=" in comment:
                config_type = comment.split("config_type=")[1].split()[0].strip()
                
            parsed_properties_dict = parsed_properties(comment)
            
            symbols, positions, masses, forces, velocities, groups = [], [], [], [], [], []
            
            for _ in range(natoms):
                words = f.readline().split()
                if not words:
                    continue
                symbols.append(read_symbols(words, parsed_properties_dict))
                positions.append(read_positions(words, parsed_properties_dict))
                masses.append(read_mass(words, parsed_properties_dict))
                forces.append(read_force(words, parsed_properties_dict))
                velocities.append(read_velocity(words, parsed_properties_dict))
                groups.append(read_group(words, parsed_properties_dict))
                
            info = {
                'energy': energy,
                'stress': stress,
                'forces': forces,
                'velocities': velocities,
                'group': groups,
                'config_type': config_type
            }
            frames.append(Atoms(symbols=symbols, positions=positions, masses=masses, 
                                cell=cell, pbc=pbc, info=info))
    return frames

def dump_xyz(f: Any, atoms: Atoms) -> None:
    """将 ASE Atoms 对象以扩展 XYZ 格式写入文件对象中。"""
    def is_valid_key(key: str) -> bool:
        return (key in atoms.info and 
                atoms.info[key] is not None and 
                all(v is not None for v in atoms.info[key]))
                
    valid_keys = {key: is_valid_key(key) for key in ['stress', 'velocities', 'forces', 'group']}
    
    out_lines = []
    out_lines.append(str(len(atoms)))
    
    comment_parts = []
    comment_parts.append('pbc="' + ' '.join(['T' if p else 'F' for p in atoms.get_pbc()]) + '"')
    comment_parts.append('Lattice="' + ' '.join(map(str, atoms.get_cell().reshape(-1))) + '"')
    
    if atoms.info.get('energy') is not None:
        comment_parts.append(f"energy={atoms.info['energy']}")
        
    if valid_keys['stress']:
        stress_info = atoms.info['stress']
        if len(stress_info) == 6:
            virial = -stress_info[[0, 5, 4, 5, 1, 3, 4, 3, 2]] * atoms.get_volume()
        else:
            virial = -stress_info.reshape(-1) * atoms.get_volume()
        comment_parts.append('virial="' + ' '.join(map(str, virial)) + '"')
        
    props = "Properties=species:S:1:pos:R:3:mass:R:1"
    if valid_keys['velocities']: props += ":vel:R:3"
    if valid_keys['forces']: props += ":force:R:3"
    if valid_keys['group']: props += ":group:I:1"
    comment_parts.append(props)
    
    if atoms.info.get('config_type') is not None:
        comment_parts.append(f"config_type={atoms.info['config_type']}")
    if atoms.info.get('weight') is not None:
        comment_parts.append(f"weight={atoms.info['weight']}")
        
    out_lines.append(" ".join(comment_parts))
    
    velocities = atoms.info.get('velocities')
    forces = atoms.info.get('forces')
    groups = atoms.info.get('group')
    
    for i, atom in enumerate(atoms):
        line = f"{atom.symbol:2} {atom.position[0]:>15.8e} {atom.position[1]:>15.8e} {atom.position[2]:>15.8e} {atom.mass:>15.8e}"
        if valid_keys['velocities'] and velocities:
            line += f" {velocities[i][0]:>15.8e} {velocities[i][1]:>15.8e} {velocities[i][2]:>15.8e}"
        if valid_keys['forces'] and forces:
            line += f" {forces[i][0]:>15.8e} {forces[i][1]:>15.8e} {forces[i][2]:>15.8e}"
        if valid_keys['group'] and groups:
            line += f" {groups[i]}"
        out_lines.append(line)
        
    f.write("\n".join(out_lines) + "\n")
        
def read_restart(filename: PathLike) -> Atoms:
    """读取 GPUMD 的 restart.xyz 文件并返回 Atoms 对象。"""
    file_path = Path(filename)
    with file_path.open('r', encoding='utf-8') as f:
        natoms = int(f.readline().split()[0])
        comment = f.readline()
        
        pbc = [True, True, True]
        if "pbc=\"" in comment:
            pbc_str = comment.split("pbc=\"")[1].split("\"")[0].strip()
            pbc = [val == "T" for val in pbc_str.split()]
            
        lattice_str = comment.split("Lattice=\"")[1].split("\"")[0].strip()
        lattice = [list(map(float, row.split())) for row in lattice_str.split(" ")]
        cell = [
            [lattice[0], lattice[1], lattice[2]],
            [lattice[3], lattice[4], lattice[5]],
            [lattice[6], lattice[7], lattice[8]]
        ]
        
        symbols, positions, masses, velocities, groups = [], [], [], [], []
        has_group = "group" in comment
        
        for _ in range(natoms):
            line_data = f.readline().split()
            symbols.append(line_data[0].lower().capitalize())
            positions.append([float(line_data[1]), float(line_data[2]), float(line_data[3])])
            masses.append(float(line_data[4]))
            velocities.append([float(line_data[5]), float(line_data[6]), float(line_data[7])])
            if has_group:
                groups.append(int(line_data[8]))
                
        info: Dict[str, Any] = {'velocities': velocities}
        if has_group:
            info['group'] = groups
            
        return Atoms(symbols=symbols, positions=positions, masses=masses, 
                     cell=cell, pbc=pbc, info=info)
          
def set_pka(atoms: Atoms, 
            energy: float, 
            direction: Union[List[float], np.ndarray], 
            index: Optional[int] = None, 
            symbol: Optional[str] = None, 
            candidate_indices: Optional[List[int]] = None, 
            random_select: bool = False, 
            scaled_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
            center_pka: bool = False,
            **kwargs) -> None:
    """
    为给定的 Atoms 对象设置初级碰撞原子 (PKA) 的速度。
    
    参数:
        atoms: ASE Atoms 对象，包含模拟体系。必须预先包含 velocities 信息。
        energy: PKA 初始动能，单位 eV。
        direction: 三维数组或列表，表示 PKA 的运动方向。
        index: 指定 PKA 的原子索引（若提供，则覆盖其他选择逻辑）。
        symbol: 指定 PKA 原子的元素符号。
        candidate_indices: 候选 PKA 的索引列表。
        random_select: 是否从候选列表中随机选择 PKA。
        scaled_position: PKA 目标选择位置（分数坐标），当 random_select 为 False 时生效。
        center_pka: 是否将系统平移，使被选中的 PKA 位于盒子中心。
    """
    if atoms.info.get('velocities') is None:
        raise ValueError('原子的 velocities 属性未设置，无法分配 PKA 动能。')
    
    direction = np.asarray(direction, dtype=float)
    dir_norm = np.linalg.norm(direction)
    if dir_norm == 0:
        raise ValueError("PKA 方向向量不能为零。")
    direction = direction / dir_norm
    
    all_indices = list(range(len(atoms)))
    
    if candidate_indices is not None:
        pass
    elif symbol is not None:
        element_indices = [i for i, atom in enumerate(atoms) if atom.symbol == symbol]
        if not element_indices:
            raise ValueError(f"系统中未找到符号为 {symbol} 的原子。")
        candidate_indices = element_indices
    else:
        candidate_indices = all_indices
    
    # 确定 PKA 原子索引
    if index is not None:
        if not (0 <= index < len(atoms)):
            raise IndexError(f"指定的索引 {index} 超出范围 [0, {len(atoms)-1}]")
        if symbol is not None and atoms[index].symbol != symbol:
            raise ValueError(f"索引 {index} 处的原子为 {atoms[index].symbol}，期望值为 {symbol}")        
        selected_index = index
        print(f"[PKA 选择] 使用指定索引: {index} (元素: {atoms[index].symbol})")    
    else:
        if random_select:
            selected_index = random.choice(candidate_indices)
            print(f"[PKA 选择] 随机选取索引: {selected_index} (元素: {atoms[selected_index].symbol})")
        else:
            cell = atoms.get_cell()
            target_positions = sum(c * r for c, r in zip(cell, scaled_position))
            distances_sq = np.sum((atoms.positions[candidate_indices] - target_positions)**2, axis=1)
            min_idx = np.argmin(distances_sq)
            selected_index = candidate_indices[min_idx]            
            min_distance = np.sqrt(distances_sq[min_idx])
            print(f"[PKA 选择] 选取最接近目标的原子索引: {selected_index} "
                  f"(元素: {atoms[selected_index].symbol}, 距离目标: {min_distance:.3f} Å)")

    # 体系平移操作
    if center_pka:
        cell = atoms.get_cell()
        inv_cell = np.linalg.inv(cell.T)
        scaled_pos = np.dot(atoms.positions[selected_index], inv_cell)
        translation_vector = np.array([0.5, 0.5, 0.5]) - scaled_pos
        cart_translation = np.dot(translation_vector, cell)
        atoms.positions += cart_translation
        atoms.wrap()
        print(f"[PKA 平移] 系统已平移，使原子 {selected_index} 位于中心。位移向量: {cart_translation}")

    # 分配速度，确保动量守恒
    mass = atoms[selected_index].mass
    # 动能公式转化: v = sqrt(2E/m) / 10.18 (10.18 用于转换 eV, amu 之间的单位)
    v_mag = np.sqrt(2 * energy / mass) / 10.18
    v_pka = v_mag * direction
    
    current_vels = np.array(atoms.info['velocities'])
    delta_momentum = (current_vels[selected_index] - v_pka) * mass / (len(atoms) - 1)

    atoms_masses = np.array(atoms.get_masses())
    # 将动量变化平均分配给其他原子
    atoms.info['velocities'] = current_vels + delta_momentum / atoms_masses[:, np.newaxis]
    atoms.info['velocities'][selected_index] = v_pka.tolist()
    
    print("-" * 30)
    print(f"PKA 设置完毕 | 能量: {energy} eV")
    print(f"Index:    {selected_index}")
    print(f"Symbol:   {atoms[selected_index].symbol}")
    print(f"Position: {atoms[selected_index].position[0]:.6f}, {atoms[selected_index].position[1]:.6f}, {atoms[selected_index].position[2]:.6f}")
    print(f"Mass:     {atoms[selected_index].mass:.2f}")
    print(f"Velocity: {v_pka[0]:.6f}, {v_pka[1]:.6f}, {v_pka[2]:.6f}")
    print("-" * 30, flush=True)

def run_gpumd(atoms: Atoms, 
              dirname: PathLike, 
              run_in: List[str], 
              nep_path: PathLike = "./nep.txt",
              **kwargs) -> None:
    """在指定目录建立计算环境并调用 GPUMD 进行模拟。"""
    dir_path = Path(dirname)
    try:
        dir_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"计算目录 {dir_path} 已存在。")
        
    nep_file = Path(nep_path)
    if nep_file.exists():
        shutil.copy(nep_file, dir_path / 'nep.txt')
    else:
        print(f"[警告] 势函数文件 {nep_file} 未找到，请确保 GPUMD 可访问势函数！")
        
    original_directory = Path.cwd()
    os_chdir = getattr(Path, 'chdir', __import__('os').chdir)  # 兼容性处理
    
    try:
        os_chdir(dir_path)
        write_run(run_in)
        with open('model.xyz', 'w', encoding='utf-8') as f:
            dump_xyz(f, atoms)
            
        print(f"[GPUMD] 启动模拟: {dir_path}")
        with open("gpumd.out", "w") as out_file:
            subprocess.run(["gpumd"], stdout=out_file, stderr=subprocess.STDOUT, check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"[错误] GPUMD 运行异常中断，请检查 {dir_path}/gpumd.out")
        raise
    finally:
        os_chdir(original_directory)
    
def run_cascade(dirname: PathLike, 
                input_file: Optional[PathLike] = None, 
                energy: float = 1.0, 
                direction: Union[List[float], np.ndarray] = np.array([0, 0, 1]),
                **kwargs) -> None:
    """运行级联碰撞模拟。"""
    if input_file is None:
        raise ValueError("[错误] 必须提供 restart.xyz (input_file) 才能执行级联碰撞！")
        
    atoms = read_restart(input_file)
    set_pka(atoms, energy, direction, **kwargs)
    run_gpumd(atoms, dirname, **kwargs)
    
def run_relax(dirname: PathLike, 
              input_file: PathLike, 
              nx: int = 1, 
              ny: int = 1, 
              nz: int = 1, 
              thickness: float = 1.0, 
              **kwargs) -> None:
    """扩胞、按位置设定固定层 (group) 并运行弛豫模拟。"""
    atoms = read(input_file) * (nx, ny, nz)
    
    # 根据元素符号进行排序（GPUMD 多元素标准流程要求）
    sorted_indices = sorted(range(len(atoms)), key=lambda i: atoms[i].symbol)
    atoms = atoms[sorted_indices]
    
    cell = atoms.cell
    # 根据坐标分配分组，用于恒温器或固定边界设置
    # group 0: 边界底层; group 1: 边界顶层; group 2: 内部活动原子
    group = []
    for atom in atoms:
        pos = atom.position
        if pos[0] < thickness or pos[1] < thickness or pos[2] < thickness:
            group.append(0)
        elif pos[0] >= cell[0, 0] - thickness or pos[1] >= cell[1, 1] - thickness or pos[2] >= cell[2, 2] - thickness:
            group.append(1)
        else:
            group.append(2)
            
    atoms.info['group'] = group
    run_gpumd(atoms, dirname, **kwargs)