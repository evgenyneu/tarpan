"""Makes statistical summary of parameter destibutions: means, std etc."""

import numpy as np
from tarpan.shared.info_path import InfoPath

from tarpan.shared.summary import (
    SummaryParams, sample_summary, save_summary_to_disk)

from tarpan.shared.param_names import filter_param_names


def save_summary(fit, param_names=None, info_path=InfoPath(),
                 summary_params=SummaryParams()):
    """
    Saves statistical summary of the samples using mean, std, mode, hpdi.

    Parameters
    ----------

    fit : cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

    param_names : list of str

        Names of parameters to be included in the summary. Include all if None.

    info_path : InfoPath

        Path information for creating summaries.

    Returns
    -------
    dict:
        df:
            Panda's data frame containing the summary
        table: str
            Summary table in text format.
        samples: Panda's data frame
            Combined samples from all chains
        path_txt: str
            Path to the text summary
        path_csv: str
            Path to summary in CSV format
    """

    info_path.set_codefile()
    info_path = InfoPath(**info_path.__dict__)

    df_summary, summary, samples = make_summary(
        fit, param_names=param_names, summary_params=summary_params)

    output = save_summary_to_disk(df_summary, summary, info_path=info_path)

    return {
        "df": df_summary,
        "table": summary,
        "samples": samples,
        "path_txt": output["path_txt"],
        "path_csv": output["path_csv"]
    }


def make_summary(fit, param_names, summary_params=SummaryParams()):
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

    param_names = filter_param_names(fit.column_names, param_names)
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
    df_summary, table = sample_summary(df=samples, extra_values=df_summary,
                                       params=summary_params)

    return df_summary, table, samples
