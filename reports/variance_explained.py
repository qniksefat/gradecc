from os import path
import seaborn as sns
import pandas as pd

from gradecc.compute.gradient import variance_explained
from gradecc.utils.filenames import dir_images

variance = variance_explained(epic_list=None, cum_sum=False, reference_epic='baseline')

fig = sns.lineplot(y='value', x='variable', hue='index',
                   style='index', markers=True,
                   data=pd.melt(variance.reset_index(), ['index']))

fig = fig.get_figure()

fig.savefig(path.join(dir_images, 'variance_explained.png'))
