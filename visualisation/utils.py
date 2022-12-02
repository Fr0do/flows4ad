import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns


def set_visualisation_options(config=None):
    sns.set(font_scale=1.3)
    sns.set_style("darkgrid", {"axes.facecolor": ".95"})

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype']  = 42
    matplotlib.rcParams['font.family'] = 'serif'


def visualise_prediction_histograms(predictions, config):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    left = np.quantile(predictions['in'], 0.01)
    right = np.quantile(predictions['in'], 0.99)

    pretty_dataset_name = f"{config.dataset_name.split('_')[-1].capitalize()}"

    ax.hist(predictions['in'], bins=30, density=True, range=(left, right), alpha=0.5, color='cyan', label='ID');
    ax.hist(predictions['out'], bins=20, density=True, range=(left, right), alpha=0.5, color='red', label='OOD');

    ax.set_xlabel(r'$\log p(x)$', fontsize=20);
    ax.legend(fontsize=20);
    ax.set_title(pretty_dataset_name, fontsize=24);

    visualisation_path = os.path.join(config.output_dir, f"{pretty_dataset_name}_id_ood_hist.pdf")
    fig.savefig(visualisation_path)