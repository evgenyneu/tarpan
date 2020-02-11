from dataclasses import dataclass, field
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.plot.utils import remove_ticks_labels



@dataclass
class ScatterKdeParams:
    title: str = None  # Plot's title
    xlabel: str = None
    ylabel1: str = None
    ylabel2: str = None
    legend_labels = None  # Labels for the plot legend. No legend if None

    marker_colors: List = field(
        default_factory=lambda:
        ["#00a6ff44", '#ff002144', '#8888FF44', '#BBBB1144'])

    marker_edgecolors: List = field(
        default_factory=lambda: ["#00a6ff", '#ff0021', '#8888FF', '#BBBB11'])

    errorbar_colors: List = field(
        default_factory=lambda:
        ["#00a6ff66", '#ff002144', '#8888FF44', '#BBBB1144'])

    markers: List = field(
        default_factory=lambda: ["^", "o", 'x', '*'])

    kde_facecolors: List = field(
        default_factory=lambda:
        ["#00a6ff44", '#ff002144', '#8888FF44', '#BBBB1144'])

    kde_edgecolors: List = field(
        default_factory=lambda: ["#00a6ff", '#ff0021', '#8888FF', '#BBBB11'])

    grid_color: str = "#aaaaaa"
    grid_alpha: float = 0.2
    markersize: float = 80

    plot_width: float = 6
    plot_height: float = 6


def save_scatter_and_kde(values,
                         uncertainties,
                         title=None,
                         xlabel=None,
                         ylabel=None,
                         info_path=InfoPath(),
                         scatter_kde_params=ScatterKdeParams(),
                         legend_labels=None):
    """
    Create a scatter plot and a KDE plot under it.
    The KDE plot uses uncertainties of each individual observation.

    Parameters
    ----------
    values: list of lists
        List of values to plot. Supply more than one list to
        see distributions shown with different colors and markers.
    uncertainties: list of lists
        Uncertainties coresponding to the `values`.
    """

    if title is not None:
        scatter_kde_params.title = title

    if xlabel is not None:
        scatter_kde_params.xlabel = xlabel

    if ylabel is not None:
        if isinstance(ylabel, list):
            if len(ylabel) == 2:
                scatter_kde_params.ylabel1 = ylabel[0]
                scatter_kde_params.ylabel2 = ylabel[1]
            elif len(ylabel) == 1:
                scatter_kde_params.ylabel1 = ylabel[0]
        else:
            scatter_kde_params.ylabel1 = ylabel

    if legend_labels is not None:
        scatter_kde_params.legend_labels = legend_labels

    sns.set(style="ticks")
    info_path.set_codefile()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   figsize=(scatter_kde_params.plot_width,
                                            scatter_kde_params.plot_height),
                                   gridspec_kw={'hspace': 0})

    # Make a scatter plot
    # ---------------

    marker_colors = cycle(scatter_kde_params.marker_colors)
    marker_edgecolors = cycle(scatter_kde_params.marker_edgecolors)
    errorbar_colors = cycle(scatter_kde_params.errorbar_colors)
    markers = cycle(scatter_kde_params.markers)

    for i, (values_list, uncertainties_list) in enumerate(zip(values, uncertainties)):
        marker_color = next(marker_colors)
        marker_edgecolor = next(marker_edgecolors)
        errorbar_color = next(errorbar_colors)
        marker = next(markers)

        ax1.errorbar(values_list, range(len(values_list)),
                     xerr=uncertainties_list, fmt='none',
                     ecolor=errorbar_color,
                     elinewidth=1,
                     zorder=1)

        value_label = '_nolegend_'

        if scatter_kde_params.legend_labels is not None:
            value_label = scatter_kde_params.legend_labels[i]

        ax1.scatter(values_list, range(len(values_list)),
                    marker=marker,
                    color=marker_color,
                    edgecolor=marker_edgecolor,
                    s=scatter_kde_params.markersize,
                    zorder=2,
                    label=value_label)

    if scatter_kde_params.ylabel1 is not None:
        ax1.set_ylabel(scatter_kde_params.ylabel1)

    if scatter_kde_params.legend_labels is not None:
        ax1.legend()

    # Make a density plot
    # -----------------

    xlims = ax1.get_xlim()
    pad = abs(xlims[0] - xlims[1]) / 10
    x = np.linspace(xlims[0] - pad, xlims[1] + pad, 1000)

    kde_facecolors = cycle(scatter_kde_params.kde_facecolors)
    kde_edgecolors = cycle(scatter_kde_params.kde_edgecolors)

    for values_list, uncertainties_list in zip(values, uncertainties):
        kde_facecolor = next(kde_facecolors)
        kde_edgecolor = next(kde_edgecolors)

        y = gaussian_kde(x, values_list, uncertainties_list)

        ax2.fill_between(x, y1=y,
                         edgecolor=None,
                         facecolor=kde_facecolor,
                         linewidth=0)

        ax2.plot(x, y, c=kde_edgecolor, linewidth=1)

    if scatter_kde_params.xlabel is not None:
        ax2.set_xlabel(scatter_kde_params.xlabel)

    if scatter_kde_params.ylabel2 is not None:
        ax2.set_ylabel(scatter_kde_params.ylabel2)

    if scatter_kde_params.title is not None:
        fig.suptitle(scatter_kde_params.title)

    info_path.base_name = info_path.base_name or "scatter_kde"
    info_path.extension = info_path.extension or 'pdf'
    plot_path = get_info_path(info_path)

    ax1.grid(color=scatter_kde_params.grid_color, linewidth=1,
             alpha=scatter_kde_params.grid_alpha)

    ax2.grid(color=scatter_kde_params.grid_color, linewidth=1,
             alpha=scatter_kde_params.grid_alpha)

    remove_ticks_labels(ax1, remove_x=False, remove_y=True)
    remove_ticks_labels(ax2, remove_x=False, remove_y=True)

    if scatter_kde_params.title is not None:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(plot_path, dpi=info_path.dpi)


def gaussian_kde(eval_points, values, uncertainties):
    """
    Computer Gaussian KDE (kernel density estimator) using values and their
    uncertianties by adding PDFs of Normal distributions at each value that
    have mean equal to the value and standard deviation equal to value's
    uncertainty.

    Parameters
    ----------

    eval_points: list or numpy.ndarray
        Value at which KDE will be evaluated.
    values: list or numpy.ndarray
        Values for computing KDE.
    uncertainties: list or numpy.ndarray
        Corresponding uncertainties for the `values`.

    Returns
    -------
    numpy.ndarray
        Values of the KDE corresponding to `eval_points`.
    """

    x = np.array(eval_points)
    mu = np.array(values)
    sigma = np.array(uncertainties)

    if mu.shape[0] == 0:
        return np.array([])

    if mu.shape[0] != sigma.shape[0]:
        raise ValueError('Values and uncertainties\
are lists of different lengths')

    guassian = -0.5 * ((x[:, np.newaxis] - mu[np.newaxis, :]) / sigma)**2
    gaussian = np.exp(guassian)
    gaussian /= np.sqrt(2 * np.pi) * sigma
    return gaussian.sum(axis=1) / len(values)
