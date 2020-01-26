"""Make pair plot of parameters"""

from tarpan.shared.info_path import InfoPath
from tarpan.shared.param_names import filter_param_names
from tarpan.shared.pair_plot import (
    PairPlotParams, save_pair_plot as shared_save_pair_plot)


def save_pair_plot(fit,
                   param_names=None,
                   info_path=InfoPath(),
                   pair_plot_params=PairPlotParams()):
    """
    Save a pair plot of distributions of parameters. It helps
    to see correlations between parameters and spot funnel
    shaped distributions that can result in sampling problems.

    Parameters
    ----------

    fit : cmdstanpy.stanfit.CmdStanMCMC
        Samples from cmdstanpy.

    param_names : list of str
        Names of parameters. Include all if None.

    info_path : InfoPath
        Path information for creating summaries.

    """

    info_path.set_codefile()
    param_names = filter_param_names(fit.column_names, param_names)
    samples = fit.get_drawset(params=param_names)

    shared_save_pair_plot(samples, param_names=param_names,
                          info_path=info_path,
                          pair_plot_params=pair_plot_params)
