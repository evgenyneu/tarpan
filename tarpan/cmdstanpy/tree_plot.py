"""Makes a tree plot showing summary of distributions of parameters"""

from tarpan.shared.info_path import InfoPath
from tarpan.shared.tree_plot import TreePlotParams, summary_from_dict

from tarpan.shared.summary import (
    SummaryParams, sample_summary, make_comparative_tree_plot)


def save_comparative_tree_plot(fits,
                               values_no_error_bars=[],
                               param_names=None,
                               info_path=InfoPath(),
                               summary_params=SummaryParams(),
                               tree_params=TreePlotParams()):
    """
    Save a tree plot that compares parameter summaries form
    different distributions. One can also supply additional markers
    to be compared with using `values_no_error_bars` parameter.

    Parameters
    ----------

    fits : list of cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

    values_no_error_bars : list of dict
        Additional markers to be shown on tree plot, without error bars:

        [
            {
                "mu": 2.3,
                "sigma": 3.3
            }
        ]

    param_names : list of str

        Names of parameters to be included in the summar. Include all if None.

    info_path : InfoPath

        Path information for creating summaries.

    """

    summaries = []

    for fit in fits:
        samples = fit.get_drawset(params=param_names)
        summary, _ = sample_summary(samples, params=summary_params)
        summaries.append(summaries)

    for values in values_no_error_bars:
        summaries.append(summary_from_dict(values))

    make_comparative_tree_plot(
        summaries, info_path=info_path,
        tree_params=tree_params)
