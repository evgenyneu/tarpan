"""Make pair plot of parameters"""

from dataclasses import dataclass
import seaborn as sns
from tarpan.shared.param_names import filter_param_names


@dataclass
class PairPlotParams:
    title: str = None  # Plot's title
    plot_color: str = "#00a6ff"
    plot_edge_color: str = "#00a6ff"
    plot_alpha: float = 0.03
    plot_line_width: float = 1
    plot_marker_size: float = 30
    diag_color: str = "#00a6ff"
    max_params: int = 7  # Maximum number of parameter to show in the plot

    """Type of diahonal plots: 'auto’, ‘hist’, ‘kde’, None"""
    diag_kind: str = 'kde'


def make_pair_plot(samples, param_names=None,
                   pair_plot_params=PairPlotParams()):
    """
    Make histograms for the parameters from posterior destribution.

    Parameters
    -----------

    samples : Panda's DataFrame

        Each column contains samples from posterior distribution.

    param_names : list of str

        Names of the parameters for plotting. If None, all will be plotted.

    Returns
    -------
    Matplotlib's figure:
        Plot's figure
    """

    param_names = filter_param_names(samples.columns, param_names)

    fig = sns.pairplot(samples,
                       diag_kind=pair_plot_params.diag_kind,
                       plot_kws=dict(
                            s=pair_plot_params.plot_marker_size,
                            color=pair_plot_params.plot_color,
                            edgecolor=pair_plot_params.plot_edge_color,
                            alpha=pair_plot_params.plot_alpha,
                            linewidth=pair_plot_params.plot_line_width
                       ),
                       diag_kws=dict(
                            color=pair_plot_params.diag_color
                       ))

    return fig
