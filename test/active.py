from calorine.calculators import CPUNEP, GPUNEP
from tdekit import ForceAnalyzer
import matplotlib.pyplot as plt
import numpy as np

prepath = '../potentials/'
calculators = [
    CPUNEP(f'{prepath}0.txt'), 
    CPUNEP(f'{prepath}1.txt'),
    CPUNEP(f'{prepath}2.txt'),
    CPUNEP(f'{prepath}3.txt'),
]
frame_paths = [
    '../xyzs/test.xyz',
]
frame_labels = [
    'test',
]

xmin = 0.05
xmax = 0.2
analyzer = ForceAnalyzer(
    calculators, frame_paths, frame_labels, 
    minimum=xmin, maximum=xmax, 
    bin_edges=np.linspace(0,5,100), 
    load_max_delta=False
)
fig, ax = plt.subplots(1, 1)
analyzer.plot_max_force_differences(ax, if_split=True, if_unsel=True)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, None)
ax.set_xlabel(r'$\sigma_f^{max} (eV/Å)$')
ax.set_ylabel('Frequency (%)')
ax.legend(loc="best", fontsize=plt.rcParams['font.size']-2)
plt.tight_layout()
plt.savefig(f'./ActiveLearning_{xmin}_{xmax}.png', bbox_inches='tight')
