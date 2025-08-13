from tdekit import DescriptorAnalyzer
import matplotlib.pyplot as plt

analyzer = DescriptorAnalyzer(model_filename='./potentials/0.txt',
                              method='tsne')
analyzer.add_xyz_file('./xyzs/train.xyz', 'train')
analyzer.add_xyz_file('./xyzs/test.xyz', 'test')

fig, ax = plt.subplots(1, 1)
selected_set, unselected_set = analyzer.perform_latent_analysis(ax, min_distance=0.01, min_select=0.3, max_select=None, level='structure')
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.legend(loc="upper right", frameon=False, fontsize=plt.rcParams['font.size'] - 2)
plt.tight_layout()
plt.savefig('tsne.png', bbox_inches='tight')

analyzer.split_xyz('./selected.xyz', selected_set)
analyzer.split_xyz('./unselected.xyz', unselected_set)