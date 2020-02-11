"""Contains main function that does all analysis"""

from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.shared.summary import SummaryParams
from tarpan.shared.tree_plot import make_tree_plot
from tarpan.shared.histogram import save_histogram_from_summary
from tarpan.shared.pair_plot import save_pair_plot
from tarpan.cmdstanpy.summary import save_summary
from tarpan.cmdstanpy.traceplot import save_traceplot


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


def save_analysis(fit, param_names=None, info_path=InfoPath(),
                  summary_params=SummaryParams()):
    """
    Create all analysis files at once: diagnostic, summary, trace and posterior.

    Parameters
    -----------

    fit : cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

    param_names : list of str

        Names of parameters to plot. Plot all parameters if None.
    """

    info_path.set_codefile()
    save_diagnostic(fit, info_path=info_path)

    summary = save_summary(
        fit, param_names=param_names, info_path=info_path,
        summary_params=summary_params)

    make_tree_plot(summary['df'], param_names=param_names, info_path=info_path,
                   summary_params=summary_params)

    save_traceplot(fit, param_names=param_names, info_path=info_path)

    save_histogram_from_summary(
        summary['samples'], summary['df'], param_names=param_names,
        info_path=info_path, summary_params=summary_params)

    save_pair_plot(
        summary['samples'], param_names=param_names,
        info_path=info_path)
