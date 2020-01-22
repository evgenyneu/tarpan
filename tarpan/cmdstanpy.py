import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from dataclasses import dataclass

from .shared import (
    sample_summary, save_summary_to_disk,
    InfoPath, get_info_path, make_tree_plot,
    save_posterior_plot)


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


stan_technical_columns = [
    'lp__',
    'accept_stat__',
    'stepsize__',
    'treedepth__',
    'n_leapfrog__',
    'divergent__',
    'energy__']


def save_summary(fit, param_names=None, info_path=InfoPath()):
    """
    Saves statistical summary of the samples using mean, std, mode, hpdi.

    Parameters
    ----------

    fit : cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

    param_names : list of str

        Names of parameters to be included in the summar. Include all if None.

    info_path : InfoPath

        Path information for creating summaries.

    """

    info_path = InfoPath(**info_path.__dict__)
    info_path.stack_depth += 1

    df_summary, summary, samples = make_summary(
        fit, param_names=param_names)

    output = save_summary_to_disk(df_summary, summary, info_path=info_path)

    return {
        "df": df_summary,
        "table": summary,
        "samples": samples,
        "path_txt": output["path_txt"],
        "path_csv": output["path_csv"]
    }


def make_summary(fit, param_names):
    """
    Returns statistical summary table for parameters:
    mean, std, mode, hpdi.

    Parameters
    ----------

    fit : cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

    param_names : list of str

        Names of parameters to be included in the summar. Include all if None.
    """

    # Make the list of columns for the summary
    # ---------------

    # Exclude Stan's diagnistic columns
    param_filtered = [
        a for a in fit.column_names if a not in stan_technical_columns
    ]

    if param_names is not None:
        # If param_names contains 'a', we will also plot
        # parameters named 'a.1', 'a.2' etc.
        param_filtered = [
            a for a in param_filtered
            if a in param_names
            or (re.sub(r'\.[0-9]+\Z', '', a) in param_names)
        ]

    param_names = param_filtered

    samples = fit.get_drawset(params=param_names)

    # Get R_hat values from the summary
    # --------

    df_summary = fit.summary()

    df_summary.rename(
        index=(lambda name: name.replace('[', '.').replace(']', '')),
        inplace=True)

    df_summary = df_summary[['N_Eff', 'R_hat']]
    df_summary['N_Eff'] = np.round(df_summary['N_Eff'])
    df_summary['N_Eff'] = df_summary['N_Eff'].astype(int)

    # Get the summary
    df_summary, table = sample_summary(df=samples, extra_values=df_summary)

    return df_summary, table, samples


def remove_ticks_labels(ax, remove_x=True, remove_y=True):
    if remove_x:
        ax.set_xticklabels([])
        ax.set_xticks([])

    if remove_y:
        ax.set_yticklabels([])
        ax.set_yticks([])


def plot_kde_fallback_hist(samples, **kwargs):
    """
    Plot KDE of the sample and fall back to histogram if KDE fails
    """

    try:
        sns.kdeplot(samples, **kwargs)
    except np.linalg.LinAlgError:
        sns.distplot(samples, norm_hist=True, kde=False, **kwargs)


def save_diagnostic(fit, info_path=InfoPath()):
    """
    Save diagnostic information from the fit into a text file.
    """

    info_path = InfoPath(**info_path.__dict__)
    info_path.base_name = info_path.base_name or 'diagnostic'
    info_path.extension = 'txt'
    file_path = get_info_path(info_path)

    with open(file_path, "w") as text_file:
        print(fit.diagnose(), file=text_file)


def make_traceplot(fit, param_names=None, info_path=InfoPath(),
                             traceplot_params=TraceplotParams()):
    """
    Make traceplots form the fit.

    Parameters
    ----------

    fit : cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

    param_names : list of str

        Names of parameters to plot.
    """

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

            samples_for_kde = samples_for_kde[(samples_for_kde > inner_range[0])
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
    # ---------------

    # Exclude Stan's diagnostic columns
    param_filtered = [
        a for a in fit.column_names if a not in stan_technical_columns]

    if param_names is not None:
        # If param_names contains 'a', we will also plot
        # parameters named 'a.1', 'a.2' etc.
        param_filtered = [
            a for a in param_filtered
            if a in param_names
            or (re.sub(r'\.[0-9]+\Z', '', a) in param_names)
        ]

    param_filtered.insert(0, 'lp__')  # Always show traceplot of probability
    param_names = param_filtered

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


def analyse(fit, param_names=None, info_path=InfoPath()):
    """
    Save diagnostic, summary informatino, trace and posterior.

    Parameters
    -----------

    fit : cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

    param_names : list of str

        Names of parameters to plot.
    """

    save_diagnostic(fit, info_path=info_path)

    summary = save_summary(
        fit, param_names=param_names, info_path=info_path)

    make_tree_plot(summary['df'], param_names=param_names, info_path=info_path)
    make_traceplot(fit, param_names=param_names, info_path=info_path)

    save_posterior_plot(
        summary['samples'], summary['df'], param_names=param_names,
        info_path=info_path)
