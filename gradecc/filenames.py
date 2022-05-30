from os import path

# keep large data files out of repo to save on GitHub
data_inside = '/Users/qasem/Dropbox/JasonANDQasem_SHARED/codes/gradients-rl-task/data/'
atlas_filename = path.join(data_inside, 'Schaefer2018_1000Parcels_7Networks_order.dlabel.nii')
subjects_filename = path.join(data_inside, 'participants.tsv')
measures_filename = path.join(data_inside, 'measures.csv')

data_outside = '/Users/qasem/Dropbox/JasonANDQasem_SHARED/codes/grad_ecc_RL_data/'
dir_dataset = path.join(data_outside, 'RL_dataset_Mar2022/')
dir_images = path.join(data_outside, 'output_plots/')
