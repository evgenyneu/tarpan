import numpy as np
from itertools import cycle
from dataclasses import dataclass, field
from typing import List
from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.plot.kde import scatter_and_kde, ScatterKdeParams


@dataclass
class PosteriorKdeParams:
    kde_plot_count: int = 50  # Number of posterior plots
    kde_values: int = 500  # Number of x and y values to calculate for KDE

    kde_edgecolors: List = field(
        default_factory=lambda:
            ["#0060ff15", '#ff002115', '#8888FF30', '#BBBB1130'])

    data_kde_show_shade: bool = False  # Show shade under data KDE plot
    data_kde_line_width: float = 2  # Line width of data KDE


def save_posterior_scatter_and_kde(
    fits,
    pdf,
    values,
    uncertainties,
    title=None,
    xlabel=None,
    ylabel=None,
    info_path=InfoPath(),
    scatter_kde_params=ScatterKdeParams(),
    posterior_kde_params=PosteriorKdeParams(),
    legend_labels=None,
    plot_fn=None):
    """
    Create a scatter plot for the data using `values` and `uncertainties`,
    show a KDE plot under it, and plot multiple posterior distributions
    on the KDE plot.

    Parameters
    ----------
    fits : list of cmdstanpy.stanfit.CmdStanMCMC
        Contains the samples from cmdstanpy.

    pdf : function(x, row)
        A function that will be called to evaluate the posterior
        distribution at x. The functinon should return the value
        of the posterior PDE corresponding to x.

        x : float
            Value at which the posterior is evaluated
        row : dict
            Contains single sample from posterior distribution.

    values: list of lists
        List of values to plot. Supply more than one list to
        see distributions shown with different colors and markers.
    uncertainties: list of lists
        Uncertainties coresponding to the `values`.
    plot_fn: [function(fig, axes, params), params]
        function:
            A function that can be used to add extra information to the
            plot before it is saved.

            Parameters
            ----------

            fig: Matplotlib's figure object
            axes: list of Matplotlib's axes objects
            params: custom parameters that are passed to the function
        params: custom parameters that will be passed to the function

    Returns
    --------
    fig : Matplotlib's figure object
    axes : list of Matplotlib's axes
    """

    scatter_kde_params.kde_show_shade = posterior_kde_params.data_kde_show_shade
    scatter_kde_params.kde_line_width = posterior_kde_params.data_kde_line_width

    fig, axes = scatter_and_kde(
                    values=values,
                    uncertainties=uncertainties,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend_labels=legend_labels,
                    scatter_kde_params=scatter_kde_params)

    kde_edge_colors = cycle(posterior_kde_params.kde_edgecolors)
    ax2 = axes[1]
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()

    for fit in fits:
        kde_edge_color = next(kde_edge_colors)
        post = fit.get_drawset()
        x = np.linspace(xlim[0], xlim[1], posterior_kde_params.kde_values)

        for chain_id in range(fit.chains):
            nplots = int(posterior_kde_params.kde_plot_count / fit.chains)

            for i in range(nplots):
                row = post.iloc[chain_id * fit.draws + i]
                y = pdf(x=x, row=row)
                ax2.plot(x, y, color=kde_edge_color, zorder=1)

    ax2.set_xlim(xlim)

    # Ensure the new ylim upper value is not too large
    ylim_new = ax2.get_ylim()
    ylim_max = ylim[1] * min(2, ylim_new[1] / ylim[1])
    ax2.set_ylim(ylim[0], ylim_max)

    if plot_fn is not None:
        plot_fn[0](fig, axes, params=plot_fn[1])

    info_path.set_codefile()
    info_path.base_name = info_path.base_name or "posterior_scatter_kde"
    info_path.extension = info_path.extension or 'pdf'
    plot_path = get_info_path(info_path)
    fig.savefig(plot_path, dpi=info_path.dpi)

    return fig, axes
