from ase.io import read
from core import run_gpumd

atoms = read("model.xyz") * (15, 9, 2)
group = []
thickness = 7
for atom in atoms:
    if atom.position[0] < thickness or atom.position[1] < thickness or atom.position[2] < thickness:
        group.append(0)
    elif atom.position[0] >= atoms.cell[0, 0] - thickness or atom.position[1] >= atoms.cell[1, 1] - thickness or atom.position[2] >= atoms.cell[2, 2] - thickness:
        group.append(1)
    else:
        group.append(2)
atoms.info['group'] = group

run_in = [
    'potential nep.txt',
    'velocity 300',
    'time_step 1',
    'ensemble npt_scr 300 300 100 0 100 1000',
    'dump_thermo 1000',
    'dump_restart 30000',
    'dump_exyz 1000 1 1',
    'run 50000'
]

run_gpumd(atoms, 'relax', run_in)

