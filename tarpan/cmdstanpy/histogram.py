"""Make histograms of parameter distributions"""

from tarpan.shared.info_path import InfoPath

from tarpan.shared.histogram import (
    HistogramParams, save_histogram_from_summary)

from tarpan.cmdstanpy.summary import make_summary
from tarpan.shared.summary import SummaryParams


def save_histogram(fit,
                   param_names=None,
                   info_path=InfoPath(),
                   summary_params=SummaryParams(),
                   histogram_params=HistogramParams()):
    """
    Make histograms of parameter distributions.

    Parameters
    ----------

    fit : cmdstanpy.stanfit.CmdStanMCMC
        Samples from cmdstanpy.

    param_names : list of str
        Names of parameters to be included in the summar. Include all if None.

    info_path : InfoPath
        Path information for creating summaries.

    """

    info_path.set_codefile()

    df_summary, summary, samples = make_summary(
        fit, param_names=param_names, summary_params=summary_params)

    save_histogram_from_summary(
        samples, df_summary, param_names=param_names,
        info_path=info_path, summary_params=summary_params,
        histogram_params=histogram_params)
