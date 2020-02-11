"""Makes a tree plot showing summary of distributions of parameters"""

from dataclasses import dataclass
from tarpan.shared.info_path import InfoPath, get_info_path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from itertools import cycle
import re
from tarpan.shared.summary import SummaryParams, sample_summary
from tarpan.shared.param_names import filter_param_names


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


def make_tree_plot(df_summary, param_names=None, info_path=InfoPath(),
                   tree_params: TreePlotParams = TreePlotParams(),
                   summary_params=SummaryParams()):
    """
    Make tree plot of parameters.
    """

    info_path = InfoPath(**info_path.__dict__)
    tree_plot_data = extract_tree_plot_data(
        df_summary, param_names=param_names, summary_params=summary_params)

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
    Panda's DataFrame
        Summary dataframe with a single "Mode" column.
    """

    return pd.DataFrame.from_dict(data, orient="index", columns=["Mode"])


def save_tree_plot(models,
                   extra_values=[],
                   param_names=None,
                   info_path=InfoPath(),
                   summary_params=SummaryParams(),
                   tree_params=TreePlotParams()):
    """
    Save a tree plot that summarises parameter distributions.
    Can compare summaries from multiple models, when multiple samples are
    supplied. One can also supply additional markers
    to be compared with using `extra_values` parameter.

    Parameters
    ----------

    models : list Panda's data frames

        List of data frames for each model, containg sample values for
        multiple parameters (one parameter is one data frame column).
        Supply multiple data frames to see their distribution summaries
        compared on the tree plot.

    extra_values : list of dict
        Additional markers to be shown on tree plot, without error bars:

        [
            {
                "mu": 2.3,
                "sigma": 3.3
            }
        ]

    param_names : list of str

        Names of parameters. Include all if None.

    info_path : InfoPath

        Path information for creating summaries.

    """

    info_path.set_codefile()
    summaries = []

    for samples in models:
        column_names = list(samples)
        param_names = filter_param_names(column_names, param_names)
        summary, _ = sample_summary(samples, params=summary_params)
        summaries.append(summary)

    for values in extra_values:
        summaries.append(summary_from_dict(values))

    make_comparative_tree_plot(
        summaries, info_path=info_path,
        tree_params=tree_params)


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
