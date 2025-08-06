from tdekit import DescriptorAnalyzer
import matplotlib.pyplot as plt

analyzer = DescriptorAnalyzer(model_filename='./potentials/0.txt', method='pca')
analyzer.add_xyz_file('./xyzs/train.xyz', 'train')
analyzer.add_xyz_file('./xyzs/test.xyz', 'test')

fig, ax = plt.subplots(1, 1)
selected_set, unselected_set = analyzer.perform_latent_analysis(ax, min_distance=0.01, min_select=0.3, max_select=None, 
                                                                level='atomic', if_split=True)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.legend(loc="upper right", frameon=False, fontsize=plt.rcParams['font.size'] - 2)
plt.tight_layout()
plt.savefig('pca.png', bbox_inches='tight')

analyzer.split_xyz('./selected.xyz', selected_set)
analyzer.split_xyz('./unselected.xyz', unselected_set)