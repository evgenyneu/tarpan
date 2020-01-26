"""Makes a tree plot showing summary of distributions of parameters"""

from tarpan.shared.info_path import InfoPath
from tarpan.shared.tree_plot import (
    TreePlotParams, summary_from_dict, make_comparative_tree_plot)

from tarpan.shared.summary import (
    SummaryParams, sample_summary)

from tarpan.shared.param_names import filter_param_names


def save_tree_plot(fits,
                   extra_values=[],
                   param_names=None,
                   info_path=InfoPath(),
                   summary_params=SummaryParams(),
                   tree_params=TreePlotParams()):
    """
    Save a tree plot that summarises parameter distributions.
    Can compare summaries from multiple models, when multiple fits are
    supplied. One can also supply additional markers
    to be compared with using `extra_values` parameter.

    Parameters
    ----------

    fits : list of cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

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

    for fit in fits:
        param_names = filter_param_names(fit.column_names, param_names)
        samples = fit.get_drawset(params=param_names)
        summary, _ = sample_summary(samples, params=summary_params)
        summaries.append(summary)

    for values in extra_values:
        summaries.append(summary_from_dict(values))

    make_comparative_tree_plot(
        summaries, info_path=info_path,
        tree_params=tree_params)
