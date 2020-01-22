import numpy as np
from tabulate import tabulate
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import math
import seaborn as sns
import re
import inspect
import os
from dataclasses import dataclass
from itertools import cycle


@dataclass
class InfoPath:
    # Used in `sub_dir_name` to indicate that sub-directory should
    # not be created, and the file is placed in parent directory instead.
    DO_NOT_CREATE: str = "do not create"

    # Path information for creating summaries
    path: str = None  # Full path to parent of info dir. Auto if None.
    dir_name: str = "model_info"  # Name of the info directory.
    sub_dir_name: str = None  # Name of the subdirectory. Auto if None.
    stack_depth: int = 3  # Used fo automatic `path` and `sub_dir_name`
    base_name: str = None  # Base name of the summary file or plot
    extension: str = None  # Extension of the summary or plot
    dpi: int = 300  # DPI setting using for plots


@dataclass
class TreePlotParams:
    """
    Parameters of the tree plot.
    """

    group_height: float = None
    error_bar_cap_size: float = None
    markersize: float = 50
    ylim: float = None
    markers = ['x', 'o', 'v', 'D', 'X', '1', '<', '*', '>', '|', 'd']
    marker_line_width: float = 1

    marker_colors = ['#FF9922', '#009933', '#8888FF',
                     '#BBBB11', '#3399AA', '#33BB11',
                     '#330077', '#BBAA22', '#00AA77',
                     '#EE8844']

    marker_edge_colors = ['#994400', '#003300', '#111177',
                          '#888800', '#003355', '#008800',
                          '#000022', '#776600', '#006622',
                          '#662200']

    error_bar_colors = ["#0033BB33", '#7755FF55', '#22FF9955']
    labels = None  # Labels for the plot legend. No legend if None
    figure_width: int = None
    figure_height: int = None
    title: str = None
    xlabel: str = None
    xlim: list = None


@dataclass
class PosteriorPlotParams:
    title: str = None  # Plot's title
    max_plot_pages: int = 4  # Maximum number of plots to generate.
    num_plot_rows: int = 4  # Number of rows (subplots) in a plot.
    ncols: int = 2  # Number of columns in the plot.

    # Colors and line styles for KDE lines of the error bars (HPDIs)
    # Sorted from largerst smallest HPDI values
    kde_colors = ['#FF9922', '#6666FF', '#44FF55']
    kde_line_styles = ['dotted', 'solid', '-.']


@dataclass
class SummaryParams:
    # List of probabilities for HPDIs (highest posterior density intervals)
    # to be shown in summary
    hpdis = [0.6827, 0.9545]

    def hpdi_percent(self):
        """
        Returns
        -------

        float : rounded HPDI percent value, i. e. 68 for 0.6827.
        """

        return [
            int(round(fraction, 2) * 100) for fraction in self.hpdis
        ]


def get_info_path(info_path=InfoPath()):
    """
    Get full path to the plot ro summary file.

    Parameters
    ----------

    info_path : InfoPath

        Path information for creating summaries.

    """
    info_path = InfoPath(**info_path.__dict__)
    frame = inspect.stack()[info_path.stack_depth]
    module = inspect.getmodule(frame[0])
    codefile = module.__file__

    if info_path.path is None:
        info_path.path = os.path.dirname(codefile)
    else:
        info_path.path = os.path.expanduser(info_path.path)

    full_path = os.path.join(info_path.path, info_path.dir_name)

    if info_path.sub_dir_name != info_path.DO_NOT_CREATE:
        if info_path.sub_dir_name is None:
            info_path.sub_dir_name = os.path.basename(codefile).rsplit('.', 1)[0]

        full_path = os.path.join(full_path, info_path.sub_dir_name)

    os.makedirs(full_path, exist_ok=True)
    filename = f'{info_path.base_name}.{info_path.extension}'
    return os.path.join(full_path, filename)


def save_summary(samples, info_path=InfoPath()):
    """
    Generates and saves statistical summary of the samples using mean, std, mode, hpdi.

    Parameters
    ----------

    samples : Panda's dataframe

        Each column contains samples for a parameter.

    info_path : InfoPath

        Path information for creating summaries.
    """

    df_summary, table = sample_summary(samples)
    return save_summary_to_disk(df_summary, table, info_path)


def save_summary_to_disk(df_summary, txt_summary, info_path=InfoPath()):
    """
    Saves statistical summary of the samples using mean, std, mode, hpdi.

    Parameters
    ----------

    df_summary : cmdstanpy.stanfit.CmdStanMCMC

        Panda's dataframe containing the summary for all parameters.

    txt_summary : list of str

        Text of the summary table.

    info_path : InfoPath

        Path information for creating summaries.


    Returns
    --------

    Dict:

        "df" : Dataframe containing summary.

        "table" : text version of the summary.

        "path_txt": Path to txt summary file.

        "path_csv": Path to csv summary file.

    """
    info_path = InfoPath(**info_path.__dict__)
    info_path.base_name = info_path.base_name or "summary"
    info_path.extension = 'txt'
    path_to_summary_txt = get_info_path(info_path)
    info_path.extension = 'csv'
    path_to_summary_csv = get_info_path(info_path)

    with open(path_to_summary_txt, "w") as text_file:
        print(txt_summary, file=text_file)

    df_summary.to_csv(path_to_summary_csv, index_label='Name')

    return {
        "df": df_summary,
        "table": txt_summary,
        "path_txt": path_to_summary_txt,
        "path_csv": path_to_summary_csv,
    }


def get_mode(values):
    """
    Calculates mode of the gaussian distribution
    (value at which it is maximum)
    """

    kernel = gaussian_kde(values)

    # Evaluate the kernel at no more than `max_values` for performance
    max_values = 100
    take_every = int(len(values) / max_values)

    if take_every < 1:
        take_every = 1

    values = values[0::take_every].tolist()
    imax = np.argmax(kernel(values))
    return values[imax]


def sample_summary(df, extra_values=None, params=SummaryParams()):
    """
    Prints table showing statistical summary from the sample parameters:
    mean, std, mode, hpdi.

    Parameters
    ------------

    df : Panda's dataframe

        Contains parameter sample values: each column is a parameter.

    extra_values : Panda's dataframe

        Additional values to be shown for parameters. Indexes are
        parameter names, and columns contain additional values to
        be shown in summary.

    Returns
    -------

    Tuple:
        df: Panda's dataframe containing the summary for all parameters.

        summary: text of the summary table
    """
    rows = []

    for column in df:
        values = df[column].to_numpy()
        mean = df[column].mean()
        std = df[column].std()
        mode = get_mode(df[column])

        summary_values = [column, mean, std, mode]

        for i, hpdi in enumerate(params.hpdis):
            hpdi_value = az.hpd(values, credible_interval=hpdi).tolist()

            if i == 0:
                # For the first interval, calculate upper and
                # lower uncertainties
                uncert_plus = hpdi_value[1] - mode
                uncert_minus = mode - hpdi_value[0]
                summary_values.append(uncert_plus)
                summary_values.append(uncert_minus)

            summary_values.append(hpdi_value[0])
            summary_values.append(hpdi_value[1])

        if extra_values is not None:
            # Add extra columns
            summary_values += extra_values.loc[column].values.tolist()

        rows.append(summary_values)

    headers = ['Name', 'Mean', 'Std', 'Mode', '+', '-']

    for hpdi_percent in params.hpdi_percent():
        headers.append(f'{hpdi_percent}CI-')
        headers.append(f'{hpdi_percent}CI+')


    if extra_values is not None:
        headers += extra_values.columns.values.tolist()

    formats = [".2f"] * len(headers)

    if 'N_Eff' in headers:
        formats[headers.index('N_Eff')] = ".0f"

    table = tabulate(rows, headers=headers, floatfmt=formats, tablefmt="pipe")

    df_summary = pd.DataFrame(rows, columns=headers, index=df.columns.values)
    df_summary.drop('Name', axis=1, inplace=True)
    return df_summary, table


def make_tree_plot(df_summary, param_names=None, info_path=InfoPath(),
                   tree_params: TreePlotParams = TreePlotParams()):
    """
    Make tree plot of parameters.
    """

    info_path = InfoPath(**info_path.__dict__)
    tree_plot_data = extract_tree_plot_data(
        df_summary, param_names=param_names)

    fig, ax = tree_plot(tree_plot_data, params=tree_params)
    info_path.base_name = info_path.base_name or 'summary'
    info_path.extension = info_path.extension or 'pdf'
    the_path = get_info_path(info_path)
    fig.savefig(the_path, dpi=info_path.dpi)
    plt.close(fig)


def summary_from_dict(data):
    """
    Converts values of parmeter to summary data frame, used for
    adding values to tree plot.


    Parameters
    ----------

    data: dict

        Mode values of the parameters, for example

        {
            "a": 12,
            "b": 32,
            "c": 48
        }
    Returns
    -------

    A summary data frame with one "Mode" column.
    """
    return pd.DataFrame.from_dict(data, orient="index", columns=["Mode"])


def make_comparative_tree_plot(
        summaries, param_names=None, info_path=InfoPath(),
        tree_params: TreePlotParams = TreePlotParams()):
    """
    Make tree plot that compares summaries of parameters
    """

    info_path = InfoPath(**info_path.__dict__)
    tree_plot_data = None

    for df_summary in summaries:
        tree_plot_data = extract_tree_plot_data(
            df_summary, param_names=param_names, groups=tree_plot_data)

    fig, ax = tree_plot(tree_plot_data, params=tree_params)
    info_path.base_name = info_path.base_name or 'summary'
    info_path.extension = info_path.extension or 'pdf'
    the_path = get_info_path(info_path)
    fig.savefig(the_path, dpi=info_path.dpi)
    plt.close(fig)


def tree_plot(groups, params: TreePlotParams = TreePlotParams()):
    """
    Make a tree plot of parameters.

    Parameters
    -----------

    groups : list of dict

        Data for plotting. The data structure is described in the
        "Returns" section of the `extract_tree_plot_data` function.

    group_height : float

        The height of each group, a unmber between 0 and 1.
    """

    sns.set(style="ticks")

    if params.error_bar_cap_size is None:
        error_bar_cap_size = len(groups) / 100

    fig, ax = plt.subplots()
    total_markers = 0

    for i_group, group in enumerate(groups):
        elements_in_group = len(group['values'])

        this_group_height = params.group_height

        if this_group_height is None:
            # Determine group height automatically
            x = len(group['values']) - 1
            this_group_height = 1 - math.exp(-x / 5)

        # Vertical separation between values in one group
        if len(group['values']) == 1:
            y_increment = 0
            this_group_height = 0
        else:
            y_increment = 1 / (elements_in_group - 1) * this_group_height

        markers = cycle(params.markers)
        marker_colors = cycle(params.marker_colors)
        marker_edge_colors = cycle(params.marker_edge_colors)

        # Plot the values for the group
        for i_value, value in enumerate(group['values']):
            total_markers += 1
            y_coord = i_group + this_group_height/2 - y_increment * i_value

            value_label = '_nolegend_'

            if params.labels is not None and i_group == 0:
                value_label = params.labels[i_value]

            marker = next(markers)
            marker_color = next(marker_colors)
            marker_edge_color = next(marker_edge_colors)

            ax.scatter(value["value"], y_coord,
                       marker=marker, s=params.markersize,
                       facecolor=marker_color,
                       edgecolor=marker_edge_color,
                       linewidth=params.marker_line_width, zorder=5,
                       label=value_label)

            # Plot error bars
            n_error_bars = len(value["error_bars"])
            error_bar_colors = cycle(params.error_bar_colors)

            for i_error_bar, error_bar in enumerate(value["error_bars"]):
                color = next(error_bar_colors)

                if i_error_bar == n_error_bars - 1:
                    color = marker_color

                ax.plot(error_bar, [y_coord, y_coord], zorder=i_error_bar + 1,
                        color=color)

                # Plot caps
                ax.plot([error_bar[0], error_bar[0]],
                        [y_coord - error_bar_cap_size/2,
                         y_coord + error_bar_cap_size/2],
                        zorder=i_error_bar + 1,
                        color=color)

                ax.plot([error_bar[1], error_bar[1]],
                        [y_coord - error_bar_cap_size/2,
                         y_coord + error_bar_cap_size/2],
                        zorder=i_error_bar + 1,
                        color=color)

    group_names = [group['name'] for group in groups]

    # Set figure size
    # ---------

    if params.figure_width is None or params.figure_height is None:
        # Set width and height automatically based
        params.figure_width = 6

        # Make figure taller if there are more markers to make them spread out
        params.figure_height = 3.5 + total_markers * 0.1

    fig.set_figwidth(params.figure_width)
    fig.set_figheight(params.figure_height)

    # Plot vertical line around zero if neeeded
    if ax.get_ylim()[0] < 0 and ax.get_ylim()[1] > 1:
        ax.axvline(x=0, linestyle='dashed')

    if params.labels is not None:
        ax.legend()

    if params.xlim is not None:
        ax.set_xlim(params.xlim)

    if params.xlabel is not None:
        ax.set_xlabel(params.xlabel)

    if params.title is not None:
        ax.set_title(params.title)

    ax.set_yticks(range(0, len(groups)))
    ax.set_yticklabels(group_names)
    ax.grid(axis='x', zorder=-10)

    fig.tight_layout()

    if params.ylim is not None:
        ax.set_ylim(params.ylim)

    return (fig, ax)


def extract_tree_plot_data(df, param_names=None, groups=None,
                           summary_params=SummaryParams()):
    """
    Extract data used to for tree plot function from a dataframe.

    Parameters
    -----------

    param_names: list of str

        List of parameters to plot. If None, plot all.

    df : Panda's data frame
        Data frame containing summary

    groups : list
        Tree plot data. If passed, the data frame's data will be added to it.

    Returns
    -------

    Array of dictionaries that will be used to make tree plot.

    For example, here we plot values of two variables "temperature"
    and "pressure" from two observations.
    Each value has multiple error bars, the 95% and 68% bars, for example.

    [
        {
            "name": "temperature"
            "values": [
                {
                    value: 10,
                    error_bars: [[6, 16], [9, 11]]
                },
                {
                    value: 40,
                    error_bars: [[10, 80], [38, 42]]
                },
            ]
        },
        {
            "name": "pressure"
            "values": [
                {
                    value: 1.1,
                    error_bars: [[0.1, 2.3], [0.9, 1.2]]
                },
                {
                    value: 1.6,
                    error_bars: [[0.6, 2.7], [1.1, 1.9]]
                }
            ]
        }
    ]
    """

    if groups is None:
        groups = []

    for column_name, row in df.iterrows():
        if param_names is not None:
            # If param_names contains 'a', we will also plot
            # parameters named 'a.1', 'a.2' etc.
            if (column_name not in param_names
               and re.sub(r'\.[0-9]+\Z', '', column_name) not in param_names):
                continue

        column_summary = row
        group_data = None

        # Check if `groups` already has the column
        for group in groups:
            if group['name'] == column_name:
                group_data = group
                break

        # Have not found group data - create one
        if group_data is None:
            group_data = {
                'name': column_name,
                'values': []
            }

            groups.append(group_data)

        # Add new value

        value = {}
        value['value'] = column_summary["Mode"]
        group_data['values'].append(value)

        # Add error bars from the HPDI values
        # ---------

        error_bars = []
        hpdis = sorted(summary_params.hpdi_percent(), reverse=True)

        for hpdi in hpdis:
            start = f'{hpdi}CI-'
            end = f'{hpdi}CI+'

            if start in column_summary:
                error_bars.append([
                    column_summary[start],
                    column_summary[end]])

        value['error_bars'] = error_bars

    return groups


def save_posterior_plot(samples, summary, param_names=None,
                        info_path=InfoPath(),
                        posterior_plot_params=PosteriorPlotParams()):
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

    figures_and_axes = plot_posterior(
        samples, summary, param_names=param_names,
        params=posterior_plot_params)

    base_name = info_path.base_name or "posterior"
    info_path.extension = info_path.extension or 'pdf'

    for i, figure_and_axis in enumerate(figures_and_axes):
        info_path.base_name = f'{base_name}_{i + 1:02d}'
        plot_path = get_info_path(info_path)
        fig = figure_and_axis[0]
        fig.savefig(plot_path, dpi=info_path.dpi)
        plt.close(fig)


def make_single_posterior_plot(i_start, samples, summary, param_names,
                               params: PosteriorPlotParams,
                               summary_params=SummaryParams()):

    nrows = math.ceil((len(param_names) - i_start) / params.ncols)

    if nrows > params.num_plot_rows:
        nrows = params.num_plot_rows

    fig_height = 4 * nrows

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=params.ncols, figsize=(12, fig_height),
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

        sns.distplot(samples_for_kde, kde=False, norm_hist=True, ax=ax)

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
                            color=params.kde_colors[i],
                            label='_nolegend_', alpha=0.2,
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

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return (fig, ax)


def plot_posterior(samples, summary, param_names=None,
                   params=PosteriorPlotParams()):
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
    param_filtered = samples.columns

    if param_names is not None:
        # If param_names contains 'a', we will also plot
        # parameters named 'a.1', 'a.2' etc.
        param_filtered = [
            a for a in param_filtered
            if a in param_names
            or (re.sub(r'\.[0-9]+\Z', '', a) in param_names)
        ]

    param_names = param_filtered

    # Total number of plots
    n_plots = math.ceil(math.ceil(len(param_names) / params.ncols) / \
        params.num_plot_rows)

    if n_plots > params.max_plot_pages:
        print((
            f'Showing only first {params.max_plot_pages} '
            f'pages out of {n_plots} of posterior plots.'
            'Consider specifying "param_names".'))

        n_plots = params.max_plot_pages

    if n_plots < 1:
        n_plots = 1

    figures_and_axes = []

    # Make multople traceplots
    for i_plot in range(n_plots):
        fig, ax = make_single_posterior_plot(
            i_start=i_plot * params.num_plot_rows * params.ncols,
            samples=samples,
            summary=summary,
            param_names=param_names,
            params=params)

        figures_and_axes.append([fig, ax])

    return figures_and_axes
