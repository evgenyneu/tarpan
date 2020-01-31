import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tarpan.shared.info_path import InfoPath, get_info_path


def save_scatter_and_kde(values,
                         uncertainties,
                         info_path=InfoPath(),
                         ylabel1=None,
                         ylabel2=None,
                         xlabel=None,
                         title=None):
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

    info_path.set_codefile()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'hspace': 0})

    ax1.errorbar(values, range(len(values)),
                 xerr=uncertainties, fmt='o')

    if ylabel1 is not None:
        ax1.set_ylabel(ylabel1)

    sns.kdeplot(values, ax=ax2)

    if xlabel is not None:
        ax2.set_xlabel(xlabel)

    if ylabel2 is not None:
        ax2.set_ylabel(ylabel2)

    if title is not None:
        fig.suptitle(title)

    info_path.base_name = info_path.base_name or "scatter_kde"
    info_path.extension = info_path.extension or 'pdf'
    plot_path = get_info_path(info_path)

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
