"""Make pair plot of parameters"""

from dataclasses import dataclass
import seaborn as sns
import matplotlib.pyplot as plt
from tarpan.shared.param_names import filter_param_names
from tarpan.shared.info_path import InfoPath, get_info_path
import math


@dataclass
class PairPlotParams:
    title: str = None  # Plot's title
    color: str = "#00a6ff"
    edgecolor: str = "#00a6ff55"
    diag_edge_color: str = "#ffffff"  # Edge color for histograms on diagonal
    alpha: float = 0.15  # Transparency of the marker color
    marker_size: float = 30
    max_params: int = 4  # Maximum number of parameter to show in the plot
    max_samples: int = 717  # Maximum number of samples to show in pair plot

    """Type of diahonal plots: 'auto’, ‘hist’, ‘kde’, None"""
    diag_kind: str = 'kde'


def save_pair_plot(samples, param_names=None,
                   info_path=InfoPath(),
                   pair_plot_params=PairPlotParams()):
    """
    Make histograms for the parameters from posterior destribution.

    Parameters
    -----------

    samples : Panda's DataFrame

        Each column contains samples from posterior distribution.

    param_names : list of str

        Names of the parameters for plotting. If None, all will be plotted.
    """

    info_path = InfoPath(**info_path.__dict__)
    info_path.set_codefile()

    g = make_pair_plot(
        samples, param_names=param_names,
        pair_plot_params=pair_plot_params)

    info_path.base_name = info_path.base_name or "pair_plot"
    info_path.extension = info_path.extension or 'pdf'
    plot_path = get_info_path(info_path)
    g.savefig(plot_path, dpi=info_path.dpi)


def make_pair_plot(samples, param_names=None,
                   pair_plot_params=PairPlotParams()):
    """
    Make a pair plot for the parameters from posterior destribution.

    Parameters
    -----------

    samples : Panda's DataFrame

        Each column contains samples from posterior distribution.

    param_names : list of str

        Names of the parameters for plotting. If None, all will be plotted.

    Returns
    -------
    Seaborn's PairGrid
    """

    param_names = filter_param_names(samples.columns, param_names)

    if len(param_names) > pair_plot_params.max_params:
        print((
            f'Showing only first {pair_plot_params.max_params} '
            f'parameters out of {len(param_names)} in pair plot.'
            'Consider limiting the parameter with "param_names".'))

        param_names = param_names[:pair_plot_params.max_params]

    samples = samples[param_names]

    # Show no more than `max_samples` markers
    keep_nth = math.ceil(samples.shape[0] / pair_plot_params.max_samples)
    samples = samples[::keep_nth]

    g = sns.PairGrid(samples)

    g = g.map_upper(sns.scatterplot, s=pair_plot_params.marker_size,
                    color=pair_plot_params.color,
                    edgecolor=pair_plot_params.edgecolor,
                    alpha=pair_plot_params.alpha)

    g = g.map_lower(sns.kdeplot, color=pair_plot_params.color)
    g = g.map_diag(plt.hist, color=pair_plot_params.color,
                   edgecolor=pair_plot_params.diag_edge_color)

    return g
