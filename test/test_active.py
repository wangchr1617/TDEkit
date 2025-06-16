from calorine.calculators import CPUNEP
from tdekit import ForceAnalyzer
import matplotlib.pyplot as plt
import numpy as np

calculators = [CPUNEP('./0.txt'),
               CPUNEP('./1.txt'),
               CPUNEP('./2.txt'),
               CPUNEP('./3.txt'),]
frame_paths = ['./test.xyz']
frame_labels = ['test']
analyzer = ForceAnalyzer(calculators, frame_paths, frame_labels, 
                         minimum=0.25, maximum=0.5, 
                         bin_edges=np.linspace(0, 0.75, 15), 
                         load_max_delta=False)
fig, ax = plt.subplots(1, 1)
analyzer.plot_max_force_differences(ax, if_split=True)
ax.set_xlim(0, 0.75)
ax.set_ylim(0, None)
ax.set_xlabel(r'$\sigma_f^{max} (eV/Å)$')
ax.set_ylabel('Relative Frequency (%)')
ax.legend(loc="best", fontsize=plt.rcParams['font.size'] - 2)
plt.tight_layout()
plt.savefig('max_force_diff_train.png', bbox_inches='tight')
