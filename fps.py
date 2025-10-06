from tdekit import DescriptorAnalyzer
import matplotlib.pyplot as plt

analyzer = DescriptorAnalyzer(model_filename='./nep.txt',
                              method='pca')
analyzer.add_xyz_file('./train.xyz', 'train')

fig, ax = plt.subplots(1, 1)
selected_set, unselected_set = analyzer.perform_latent_analysis(ax,
                                                                min_distance=0.0001,
                                                                min_select=50,
                                                                max_select=None,
                                                                level='structure',
                                                                strategy=None)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.legend(loc="upper right", frameon=False, fontsize=plt.rcParams['font.size'] - 2)
plt.tight_layout()
plt.savefig('descriptor_latent.png', bbox_inches='tight')

analyzer.split_xyz('./selected.xyz', selected_set)
analyzer.split_xyz('./unselected.xyz', unselected_set)
