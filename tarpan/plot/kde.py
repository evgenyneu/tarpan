from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.plot.utils import remove_ticks_labels


@dataclass
class ScatterKdeParams:
    title: str = None  # Plot's title
    xlabel: str = None
    ylabel1: str = None
    ylabel2: str = None
    marker_color: str = "#00a6ff66"
    marker_edgecolor: str = "#00a6ff"
    errorbar_color: str = "#00a6ff66"
    kde_facecolor: str = "#00a6ff66"
    kde_edgecolor: str = "#00a6ff"
    grid_color: str = "#aaaaaa"
    grid_alpha: float = 0.2
    markersize: float = 80


def save_scatter_and_kde(values,
                         uncertainties,
                         title=None,
                         xlabel=None,
                         ylabel=None,
                         info_path=InfoPath(),
                         scatter_kde_params=ScatterKdeParams()):
    """
    Create a scatter plot and a KDE plot under it.
    The KDE plot uses uncertainties of each individual observation.

    Parameters
    ----------
    values: list
        List of values to plot
    uncertainties: list
        Uncertainties coresponding to the numbers
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

    sns.set(style="ticks")
    info_path.set_codefile()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'hspace': 0})

    ax1.errorbar(values, range(len(values)),
                 xerr=uncertainties, fmt='none',
                 ecolor=scatter_kde_params.errorbar_color,
                 elinewidth=1,
                 zorder=1)

    ax1.scatter(values, range(len(values)),
                color=scatter_kde_params.marker_color,
                edgecolor=scatter_kde_params.marker_edgecolor,
                s=scatter_kde_params.markersize,
                zorder=2)

    if scatter_kde_params.ylabel1 is not None:
        ax1.set_ylabel(scatter_kde_params.ylabel1)

    margin = abs(min(values) - max(values)) / 5
    x = np.linspace(min(values) - margin, max(values) + margin, 1000)
    y = gaussian_kde(x, values, uncertainties)

    ax2.fill_between(x, y1=y,
                     edgecolor=None,
                     facecolor=scatter_kde_params.kde_facecolor,
                     linewidth=0)

    ax2.plot(x, y, c=scatter_kde_params.kde_edgecolor, linewidth=1)

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
