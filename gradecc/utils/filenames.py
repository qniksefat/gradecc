from os import path
from joblib import Memory

# keep large data files out of repo to save on GitHub
data_inside = '/Users/qasem/PycharmProjects/gradients-rl-task/data/'
atlas_filename = path.join(data_inside, 'Schaefer2018_1000Parcels_7Networks_order.dlabel.nii')
labels_filename = path.join(data_inside, 'Schaefer2018_1000Parcels_labels.csv')
subjects_filename = path.join(data_inside, 'participants.tsv')
atlas_subc_filename = path.join(data_inside, 'subcortex_regions.csv')
atlas_subc_order_filename = path.join(data_inside, 'subcortex_regions_ordered_to_plot.csv')

dir_cache = path.join(data_inside, 'cache/')
measures_filename = path.join(dir_cache, 'measures.csv')
ttests_filename = path.join(dir_cache, 'ttests.csv')
rm_anova_filename = path.join(dir_cache, 'rm_anova.csv')


data_outside = '/Users/qasem/PycharmProjects/grad_ecc_RL_data/'
dir_dataset = path.join(data_outside, 'RL_dataset_Mar2022/')
dir_subcortical = path.join(data_outside, 'subcortical_data/subcortical_data2/')
dir_images = path.join(data_outside, 'output_plots/')

# To cache big func outputs in disk. Look for @memory.cache in repo.
memory = Memory(data_outside)
