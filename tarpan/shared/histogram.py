"""Make histograms of posterior parameter distributions"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.shared.summary import SummaryParams, sample_summary
from tarpan.shared.param_names import filter_param_names


@dataclass
class HistogramParams:
    title: str = None  # Plot's title
    max_plot_pages: int = 4  # Maximum number of plots to generate.
    num_plot_rows: int = 4  # Number of rows (subplots) in a plot.
    ncols: int = 2  # Number of columns in the plot.

    hist_color = "#00a6ff"  # Fill color for histogram bars
    hist_edge_color = "#FFFFFF"  # Edge color for the histogram bars

    # Colors and line styles for KDE lines of the error bars (HPDIs)
    # Sorted from largerst smallest HPDI values
    kde_colors = ['#FF9922', '#6666FF', '#44FF55']
    kde_line_styles = ['dotted', 'solid', '-.']


def save_histogram(samples, param_names=None,
                   info_path=InfoPath(),
                   histogram_params=HistogramParams(),
                   summary_params=SummaryParams()):
    """
    Make histograms for the parameters from posterior destribution.

    Parameters
    -----------

    samples : Panda's DataFrame

        Each column contains samples from posterior distribution.

    param_names : list of str

        Names of the parameters for plotting. If None, all will be plotted.
    """

    info_path.set_codefile()
    df_summary, table = sample_summary(df=samples)

    save_histogram_from_summary(samples, df_summary,
                                param_names=param_names,
                                info_path=info_path,
                                histogram_params=histogram_params,
                                summary_params=summary_params)


def save_histogram_from_summary(samples, summary, param_names=None,
                                info_path=InfoPath(),
                                histogram_params=HistogramParams(),
                                summary_params=SummaryParams()):
    """
    Make histograms for the parameters from posterior destribution.

    Parameters
    -----------

    samples : Panda's DataFrame

        Each column contains samples from posterior distribution.

    summary : Panda's DataFrame

        Summary information about each column.

    param_names : list of str

        Names of the parameters for plotting. If None, all will be plotted.
    """

    info_path = InfoPath(**info_path.__dict__)

    figures_and_axes = make_histograms(
        samples, summary, param_names=param_names,
        params=histogram_params,
        summary_params=summary_params)

    base_name = info_path.base_name or "histogram"
    info_path.extension = info_path.extension or 'pdf'

    for i, figure_and_axis in enumerate(figures_and_axes):
        info_path.base_name = f'{base_name}_{i + 1:02d}'
        plot_path = get_info_path(info_path)
        fig = figure_and_axis[0]
        fig.savefig(plot_path, dpi=info_path.dpi)
        plt.close(fig)


def make_histogram_one_page(i_start, samples, summary, param_names,
                            params: HistogramParams,
                            summary_params=SummaryParams()):
    """
    Make a single file with histograms for the parameters
    from posterior destribution.
    """

    nrows = math.ceil((len(param_names) - i_start) / params.ncols)

    if nrows > params.num_plot_rows:
        nrows = params.num_plot_rows

    ncols = params.ncols
    fig_height = 4 * nrows
    fig_width = 12

    # Special case: if there is just one parameter show a plot with one column
    if len(param_names) == 1:
        ncols = 1
        fig_width /= 2

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols, figsize=(fig_width, fig_height),
        squeeze=False)

    axes = ax.flatten()

    for i_axis, ax in enumerate(axes):
        i_param = i_start + i_axis

        if i_param >= len(param_names):
            break

        parameter = param_names[i_param]
        param_samples = samples[parameter]
        data = summary.loc[parameter]

        # Exclude extreme outliers from the samples
        # to avoid the blow-up of the x-axis range
        inner_range = np.percentile(param_samples, [0.5, 99.5])

        samples_for_kde = param_samples[(param_samples > inner_range[0])
                                        & (param_samples < inner_range[1])]

        sns.distplot(samples_for_kde, kde=False, norm_hist=True, ax=ax,
                     hist_kws={
                        "color": params.hist_color,
                        "zorder": 1,
                        "edgecolor": params.hist_edge_color,
                        "linewidth": 1,
                        "alpha": 1})

        # Show KDEs for the error bars (HPDIs)
        # -----------

        hpdis = sorted(summary_params.hpdi_percent(), reverse=True)

        for i, hpdi in enumerate(hpdis):
            start = f'{hpdi}CI-'
            end = f'{hpdi}CI+'

            # KDE plot
            sns.kdeplot(samples_for_kde, shade=False,
                        clip=[data[start], data[end]],
                        label=f'{hpdi}% HPDI', ax=ax, legend=None,
                        color=params.kde_colors[i],
                        linestyle=params.kde_line_styles[i],
                        linewidth=2)

            if i == len(hpdis) - 1:
                # Show shade under KDE for the last error bar
                sns.kdeplot(samples_for_kde, shade=True,
                            clip=[data[start], data[end]],
                            color="#000000",
                            label='_nolegend_', alpha=0.2,
                            zorder=10,
                            ax=ax, legend=None,
                            linewidth=2)

        ax.axvline(x=data['Mean'], label='Mean', linewidth=1.5,
                   linestyle='dashed', color='#33AA66')

        ax.axvline(x=data['Mode'], label='Mode', linewidth=1.5,
                   color='#FF66AA')

        ax.set_xlabel(parameter)

    # Do not draw the axes for non-existing plots
    for ax in axes[len(param_names):]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', mode='expand',
               ncol=len(labels))

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return (fig, ax)


def make_histograms(samples, summary, param_names=None,
                    params=HistogramParams(),
                    summary_params=SummaryParams()):
    """
    Make multiple files with
    histograms for the parameters from posterior destribution.

    Parameters
    -----------

    samples : Panda's DataFrame

        Each column contains samples from posterior distribution.

    summary : Panda's DataFrame

        Summary information about each column.

    param_names : list of str

        Names of the parameters for plotting. If None, all will be plotted.
    """
    param_names = filter_param_names(samples.columns, param_names)

    # Total number of plots
    n_plots = math.ceil(math.ceil(len(param_names) / params.ncols) /
                        params.num_plot_rows)

    if n_plots > params.max_plot_pages:
        print((
            f'Showing only first {params.max_plot_pages} '
            f'pages out of {n_plots} of histogram.'
            'Consider specifying "param_names".'))

        n_plots = params.max_plot_pages

    if n_plots < 1:
        n_plots = 1

    figures_and_axes = []

    # Make multiple traceplots
    for i_plot in range(n_plots):
        fig, ax = make_histogram_one_page(
            i_start=i_plot * params.num_plot_rows * params.ncols,
            samples=samples,
            summary=summary,
            param_names=param_names,
            params=params,
            summary_params=summary_params)

        figures_and_axes.append([fig, ax])

    return figures_and_axes
