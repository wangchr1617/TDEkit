from core import run_gpumd, read_restart, set_pka
import numpy as np

atoms = read_restart('relax/restart.xyz')
energy = 3.15  # eV
direction = np.array([0, 0, 1])
set_pka(atoms, energy, direction)

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
run_gpumd(atoms, 'cascade', run_in)

