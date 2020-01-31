"""Make traceplots from Stan's output samples"""

from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.plot.utils import plot_kde_fallback_hist, remove_ticks_labels
from tarpan.shared.param_names import filter_param_names


@dataclass
class TraceplotParams:
    max_traceplot_pages = 4  # Maximum number of traceplot to generate.
    num_traceplot_rows = 8  # Number of rows (subplots) in a traceplot.

    # Maximum number of samples to show in one traceplot.
    show_max_samples = 300

    color = ['#00993399', '#FF992299', '#8888FF99',
             '#BBBB1199', '#3399AA99', '#33BB1199',
             '#33007799', '#BBAA2299', '#55FFBB99']

    color_samples = ['#00993350', '#FF992250', '#8888FF50',
                     '#BBBB1150', '#3399AA50', '#33BB1150',
                     '#33007750', '#BBAA2250', '#55FFBB50']


def save_traceplot(fit, param_names=None, info_path=InfoPath(),
                   traceplot_params=TraceplotParams()):
    """
    Saves traceplots form the fit.

    Parameters
    ----------

    fit : cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

    param_names : list of str

        Names of parameters to plot.
    """

    info_path.set_codefile()
    info_path = InfoPath(**info_path.__dict__)

    figures_and_axes = traceplot(
        fit, param_names=param_names, params=traceplot_params)

    base_name = info_path.base_name or "traceplot"
    info_path.extension = info_path.extension or 'pdf'

    for i, figure_and_axis in enumerate(figures_and_axes):
        info_path.base_name = f'{base_name}_{i + 1:02d}'
        plot_path = get_info_path(info_path)
        fig = figure_and_axis[0]
        fig.savefig(plot_path, dpi=info_path.dpi)
        plt.close(fig)


def make_single_traceplot(i_start, fit,
                          param_names,
                          params: TraceplotParams):
    """
    Show one page of traceplots.

    Parameters
    ----------

    param_names : list of str

        Names of parameters to plot.

    """

    nrows = fit.sample.shape[2] - i_start

    if nrows > params.num_traceplot_rows:
        nrows = params.num_traceplot_rows

    if nrows > len(param_names) - i_start:
        nrows = len(param_names) - i_start

    fig_height = 2 * nrows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2, figsize=(12, fig_height),
        squeeze=False)

    # If there are too many samples, plotting will be slow
    # Thus, we only use no more than `show_max_samples`
    # by taking only every `draw_every` sample
    draw_every = int(fit.draws / params.show_max_samples)

    if draw_every < 1:
        draw_every = 1

    for i_row in range(nrows):
        param_name = param_names[i_start + i_row]
        # Parameter index in sample array
        i_param = fit.column_names.index(param_name)

        for i_chain in range(fit.chains):
            samples_for_kde = fit.sample[:, i_chain, i_param]

            # Plot KDEs
            # ----------

            ax = axes[i_row, 0]

            # Exclude extreme outliers from the samples
            # to avoid the blow-up of the x-axis range

            inner_range = np.percentile(samples_for_kde, [0.5, 99.5])

            samples_for_kde = samples_for_kde[
                (samples_for_kde > inner_range[0])
                & (samples_for_kde < inner_range[1])]

            plot_kde_fallback_hist(samples_for_kde, ax=ax,
                                   color=params.color[i_chain])

            remove_ticks_labels(ax, remove_x=False, remove_y=True)
            ax.set_ylabel(param_name)

            # Plot samples
            # ----------

            samples = fit.sample[::draw_every, i_chain, i_param]
            ax = axes[i_row, 1]
            ax.plot(samples, color=params.color_samples[i_chain])
            remove_ticks_labels(ax)

    fig.tight_layout()

    return (fig, axes)


def traceplot(fit, param_names=None, params=TraceplotParams()):
    """
    Show traceplots, diagnostic plots of samples for all
    parameters for all chains.

    Parameters
    ----------

    param_names: list of str

        List of parameters to plot.  If None, plot all.

    """

    sns.set(style="ticks")

    # Make the list of columns to be shown in the plots
    param_names = filter_param_names(fit.column_names, param_names)
    param_names.insert(0, 'lp__')  # Always show traceplot of probability

    # Total number of plots
    n_plots = math.ceil(len(param_names) / params.num_traceplot_rows)

    if n_plots > params.max_traceplot_pages:
        print((
            f'Traceplot shows only first {params.max_traceplot_pages} '
            f'pages out of {n_plots}. Consider specifying "param_names".'))

        n_plots = params.max_traceplot_pages

    if n_plots < 1:
        n_plots = 1

    figures_and_axes = []

    # Make multople traceplots
    for i_plot in range(n_plots):
        fig, ax = make_single_traceplot(
            i_start=i_plot * params.num_traceplot_rows,
            fit=fit,
            param_names=param_names,
            params=params)

        figures_and_axes.append([fig, ax])

    return figures_and_axes
