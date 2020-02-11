"""Contains main function that does all analysis"""

from tarpan.shared.info_path import InfoPath
from tarpan.shared.summary import SummaryParams
from tarpan.shared.tree_plot import make_tree_plot
from tarpan.shared.histogram import save_histogram_from_summary
from tarpan.shared.pair_plot import save_pair_plot
from tarpan.shared.summary import save_summary


def save_analysis(samples, param_names=None, info_path=InfoPath(),
                  summary_params=SummaryParams()):
    """
    Creates all analysis files at once: summary, trace and posterior.

    Parameters
    -----------

    samples : Panda's DataFrame

        Each column contains samples from posterior distribution.

    param_names : list of str

        Names of parameters to plot. Plot all parameters if None.
    """

    info_path.set_codefile()

    summary = save_summary(
        samples, param_names=param_names, info_path=info_path,
        summary_params=summary_params)

    make_tree_plot(summary['df'], param_names=param_names, info_path=info_path,
                   summary_params=summary_params)

    save_histogram_from_summary(
        samples, summary['df'], param_names=param_names,
        info_path=info_path, summary_params=summary_params)

    save_pair_plot(samples, param_names=param_names, info_path=info_path)
