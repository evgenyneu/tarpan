"""Functions for dealing with plots"""

import numpy as np
import seaborn as sns


def remove_ticks_labels(ax, remove_x=True, remove_y=True):
    """Remove ticks and labels from axes"""

    if remove_x:
        ax.set_xticklabels([])
        ax.set_xticks([])

    if remove_y:
        ax.set_yticklabels([])
        ax.set_yticks([])


def plot_kde_fallback_hist(samples, **kwargs):
    """
    Plot KDE of the sample and fall back to histogram if KDE fails
    """

    try:
        sns.kdeplot(samples, **kwargs)
    except np.linalg.LinAlgError:
        sns.distplot(samples, norm_hist=True, kde=False, **kwargs)
