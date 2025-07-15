from ase import Atoms
from ase.io import read
import numpy as np
import os
import shutil

def write_run(parameters):
    with open('run.in', 'w') as f:
        for i in parameters:
            f.write(i + '\n')

def parsed_properties(comment):
    properties_str = comment.split('properties=')[1].split()[0]
    properties = properties_str.split(':')
    parsed_properties = {}
    start = 0
    for i in range(0, len(properties), 3):
        property_name = properties[i]
        property_count = int(properties[i+2])
        parsed_properties[property_name] = slice(start, start + property_count)
        start += property_count
    return parsed_properties

def read_symbols(words_in_line, parsed_properties):
    symbol_slice = parsed_properties['species']
    symbol = words_in_line[symbol_slice]
    symbol = symbol[0].lower().capitalize()
    return symbol

def read_positions(words_in_line, parsed_properties):
    pos_slice = parsed_properties['pos']
    pos = words_in_line[pos_slice]
    return [float(pos[0]), float(pos[1]), float(pos[2])]

def read_mass(words_in_line, parsed_properties):
    if 'mass' in parsed_properties:
        mass_slice = parsed_properties['mass']
        mass = words_in_line[mass_slice]
        return float(mass[0])
    else:
        return None

def read_force(words_in_line, parsed_properties):
    force_key = 'forces' if 'forces' in parsed_properties else 'force'
    if force_key in parsed_properties:
        force_slice = parsed_properties[force_key]
        force = words_in_line[force_slice]
        return [float(force[0]), float(force[1]), float(force[2])]
    else:
        return None
    
def read_group(words_in_line, parsed_properties):
    if 'group' in parsed_properties:
        group_slice = parsed_properties['group']
        group = words_in_line[group_slice]
        return int(group[0])
    else:
        return None

def read_velocity(words_in_line, parsed_properties):
    if 'vel' in parsed_properties:
        vel_slice = parsed_properties['vel']
        vel = words_in_line[vel_slice]
        return [float(vel[0]), float(vel[1]), float(vel[2])]
    else:
        return None

def read_xyz(filename):
    frames = []
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            symbols = []
            positions = []
            masses = []
            forces = []
            velocities = []
            group = []
            natoms = int(line.strip())
            comment = f.readline().lower().strip()
            if "pbc=\"" in comment:
                pbc_str = comment.split("pbc=\"")[1].split("\"")[0].strip()
                pbc = [True if pbc_value == "t" else False for pbc_value in pbc_str.split()]
            else:
                pbc = [True, True, True]
            lattice_str = comment.split("lattice=\"")[1].split("\"")[0].strip()
            values = [val for val in lattice_str.split() if val != '']
            if len(values) == 9:
                floats = list(map(float, values))
                cell = [floats[0:3], floats[3:6], floats[6:9]]
            else:
                print(f"Invalid lattice format: {lattice_str}", flush=True)
                cell = np.eye(3) 
            if "energy=" in comment:
                energy = float(comment.split("energy=")[1].split()[0])
            else: 
                energy = None
            if "virial=" in comment:
                virials = comment.split("virial=\"")[1].split("\"")[0].strip()
                virials = np.array([float(x) for x in virials.split()]).reshape(3, 3)
                stress = - virials / np.linalg.det(cell)
            else:
                stress = None
            if "config_type=" in comment:
                config_type = comment.split("config_type=")[1].split()[0].strip()
            else:
                config_type = None
            parsed_properties_dict = parsed_properties(comment)
            for _ in range(natoms):
                line = f.readline()
                words_in_line = line.split()
                symbols.append(read_symbols(words_in_line, parsed_properties_dict))
                positions.append(read_positions(words_in_line, parsed_properties_dict))
                masses.append(read_mass(words_in_line, parsed_properties_dict))
                forces.append(read_force(words_in_line, parsed_properties_dict))
                velocities.append(read_velocity(words_in_line, parsed_properties_dict))
                group.append(read_group(words_in_line, parsed_properties_dict))
            frames.append(Atoms(symbols=symbols, positions=positions, masses=masses, cell=cell, pbc=pbc, info={'energy': energy, 'stress': stress, 'forces': forces, 'velocities': velocities, 'group': group, 'config_type': config_type}))
    return frames

def dump_xyz(f, atoms):
    def is_valid_key(key):
        return key in atoms.info and atoms.info[key] is not None and all(v is not None for v in atoms.info[key])
    valid_keys = {key: is_valid_key(key) for key in ['stress', 'velocities', 'forces', 'group']}
    Out_string = ""
    Out_string += str(int(len(atoms))) + "\n"
    Out_string += "pbc=\"" + " ".join(["T" if pbc_value else "F" for pbc_value in atoms.get_pbc()]) + "\" "
    Out_string += "Lattice=\"" + " ".join(list(map(str, atoms.get_cell().reshape(-1)))) + "\" "
    if 'energy' in atoms.info and atoms.info['energy'] is not None:
        Out_string += " energy=" + str(atoms.info['energy']) + " "
    if valid_keys['stress']:
        if len(atoms.info['stress']) == 6:
                virial = -atoms.info['stress'][[0, 5, 4, 5, 1, 3, 4, 3, 2]] * atoms.get_volume()
        else:
            virial = -atoms.info['stress'].reshape(-1) * atoms.get_volume()
        Out_string += "virial=\"" + " ".join(list(map(str, virial))) + "\" "
    Out_string += "Properties=species:S:1:pos:R:3:mass:R:1"
    if valid_keys['velocities']:
        Out_string += ":vel:R:3"
    if valid_keys['forces']:
        Out_string += ":force:R:3"
    if valid_keys['group']:
        Out_string += ":group:I:1"
    if 'config_type' in atoms.info and atoms.info['config_type'] is not None:
        Out_string += " config_type="+ atoms.info['config_type']
    if 'weight' in atoms.info and atoms.info['weight'] is not None:
        Out_string += " weight="+ str(atoms.info['weight'])
    Out_string += "\n"
    for atom in atoms:
        Out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e}'.format(atom.symbol, *atom.position, atom.mass)
        if valid_keys['velocities']:
            Out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*atoms.info['velocities'][atom.index])
        if valid_keys['forces']:
            Out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*atoms.info['forces'][atom.index])
        if valid_keys['group']:
            Out_string += ' {}'.format(atoms.info['group'][atom.index])
        Out_string += '\n'
    f.write(Out_string)
        
def read_restart(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        natoms = int(line.split(' ')[0])
        symbols = []
        positions = []
        masses = []
        velocities = []
        group = []
        comment = f.readline()  
        if "pbc=\"" in comment:
            pbc_str = comment.split("pbc=\"")[1].split("\"")[0].strip()
            pbc = [True if pbc_value == "T" else False for pbc_value in pbc_str.split()]
        else:
            pbc = [True, True, True]
        lattice_str = comment.split("Lattice=\"")[1].split("\"")[0].strip()
        lattice = [list(map(float, row.split())) for row in lattice_str.split(" ")]
        cell = [lattice[0] + lattice[1] + lattice[2], lattice[3] + lattice[4] + lattice[5], lattice[6] + lattice[7] + lattice[8]]
        if "group" in comment:
            for _ in range(natoms):
                line = f.readline()
                symbol, x, y, z, mass, vx, vy, vz, group_info= line.split()[:9]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
                velocities.append([float(vx), float(vy), float(vz)])
                masses.append(mass)
                group.append(group_info)      
            atoms = Atoms(symbols=symbols, positions=positions, masses=masses, cell=cell, pbc=pbc, info={'velocities': velocities, 'group': group})
        else:
            for _ in range(natoms):
                line = f.readline()
                symbol, x, y, z, mass, vx, vy, vz= line.split()[:8]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
                velocities.append([float(vx), float(vy), float(vz)])
                masses.append(mass)  
            atoms = Atoms(symbols=symbols, positions=positions, masses=masses, cell=cell, pbc=pbc, info={'velocities': velocities})
    return atoms
        
def run_gpumd(atoms, dirname, run_in, nep_path='nep.txt', electron_stopping_path = 'electron_stopping_fit.txt'):
    if os.path.exists(dirname):
        raise FileExistsError('Directory already exists')
    os.makedirs(dirname)
    if os.path.exists(nep_path):
        shutil.copy(nep_path, os.path.join(dirname, 'nep.txt'))
    else:
        print(f'NEP file {nep_path} does not exist')
    original_directory = os.getcwd()
    os.chdir(dirname)
    write_run(run_in)
    with open('model.xyz', 'w') as f:
        dump_xyz(f, atoms)
    os.system('gpumd > gpumd.out 2>&1')
    os.chdir(original_directory)
          
def set_pka(atoms, energy, direction, index=None, symbol=None, scaled_position=(0.5, 0.5 ,0.5)):
    if atoms.info['velocities'] is None:
        raise ValueError('The velocities of atoms are not set.')

    if index is None:
        cell = atoms.get_cell()
        target_positions = sum(c * r for c, r in zip(cell, scaled_position))
        if symbol is None:
            index = np.argmin(np.sum((atoms.positions - target_positions)**2, axis=1))
        else:
            element_indices = [i for i, atom in enumerate(atoms) if atom.symbol == symbol]
            element_positions = atoms.positions[element_indices]
            index = element_indices[np.argmin(np.sum((element_positions - target_positions)**2, axis=1))]

    mass = atoms[index].mass
    vx = pow(2 * energy / mass , 0.5) * direction[0] / pow(np.sum(direction ** 2), 0.5) / 10.18
    vy = pow(2 * energy / mass , 0.5) * direction[1] / pow(np.sum(direction ** 2), 0.5) / 10.18
    vz = pow(2 * energy / mass , 0.5) * direction[2] / pow(np.sum(direction ** 2), 0.5) / 10.18
    delta_momentum = (np.array(atoms.info['velocities'][index]) - np.array([vx, vy, vz])) * mass / (len(atoms) - 1)

    atoms_masses = np.array(atoms.get_masses())
    atoms.info['velocities'] += delta_momentum / atoms_masses[:, np.newaxis]
    atoms.info['velocities'][index] = [vx, vy, vz]
    print(f'Index: {index}', flush=True)
    print(f'Symbol: {atoms[index].symbol}', flush=True)
    print(f'Position: {atoms[index].position[0]:.6f}, {atoms[index].position[1]:.6f}, {atoms[index].position[2]:.6f}', flush=True)
    print(f'Mass: {atoms[index].mass:.2f}', flush=True)
    print(f'Velocity: {atoms.info["velocities"][index][0]:.6f}, {atoms.info["velocities"][index][1]:.6f}, {atoms.info["velocities"][index][2]:.6f}', flush=True)

def run_cascade(run_in=None, 
                dirname='cascade', 
                input_file='relax/restart.xyz', 
                energy=3, 
                direction=np.array([0, 0, 1]), 
                **kwargs):
    atoms = read_restart(input_file)
    set_pka(atoms, energy, direction, **kwargs)
    if run_in is None:
        run_in = [
            'potential nep.txt',
            'velocity 300',
            'time_step 0',
            'ensemble nve',
            'dump_exyz 1',
            'run 1',
            'time_step 1 0.01',
            'ensemble heat_nhc 300 100 0 0 1',
            'compute 0 200 10 temperature',
            'dump_restart 10000',
            'dump_exyz 2000 1 1',
            'run 30000'
        ]
    run_gpumd(atoms, dirname, run_in)
    
def run_relax(run_in=None, 
              dirname='relax', 
              input_file="model.xyz", 
              nx=15, 
              ny=9, 
              nz=2, 
              thickness=7):
    atoms = read(input_file) * (nx, ny, nz)
    cell = atoms.cell
    group = [0 if (atom.position[0] < thickness or
                   atom.position[1] < thickness or
                   atom.position[2] < thickness)
             else (1 if (atom.position[0] >= cell[0, 0] - thickness or
                         atom.position[1] >= cell[1, 1] - thickness or
                         atom.position[2] >= cell[2, 2] - thickness)
                   else 2)
             for atom in atoms]
    atoms.info['group'] = group
    if run_in is None:
        run_in = [
            'potential nep.txt',
            'velocity 300',
            'time_step 1',
            'ensemble npt_scr 300 300 100 0 100 1000',
            'dump_thermo 1000',
            'dump_restart 50000',
            'dump_exyz 1000 1 1',
            'run 50000'
        ]
    
    run_gpumd(atoms, dirname, run_in)
