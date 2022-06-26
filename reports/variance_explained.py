from os import path
import seaborn as sns
import pandas as pd

from gradecc.compute.variance import variance_explained
from gradecc.utils.filenames import dir_images

if __name__ == '__main__':
    variance = variance_explained(epoch_list=None, cum_sum=False, reference_epoch='baseline')

    fig = sns.lineplot(y='value', x='variable', hue='index',
                       style='index', markers=True,
                       data=pd.melt(variance.reset_index(), ['index']))

    fig = fig.get_figure()

    fig.savefig(path.join(dir_images, 'variance_explained.png'))
