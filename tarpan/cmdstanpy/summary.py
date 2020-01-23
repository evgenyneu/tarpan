"""Makes statistical summary of parameter destibutions: means, std etc."""

import numpy as np
import re
from tarpan.shared.info_path import InfoPath

from tarpan.shared.summary import (
    SummaryParams, sample_summary, save_summary_to_disk)

from tarpan.cmdstanpy.stan import STAN_TECHNICAL_COLUMNS


def save_summary(fit, param_names=None, info_path=InfoPath(),
                 summary_params=SummaryParams()):
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

    # Make the list of columns for the summary
    # ---------------

    # Exclude Stan's diagnistic columns
    param_filtered = [
        a for a in fit.column_names if a not in STAN_TECHNICAL_COLUMNS
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
    df_summary, table = sample_summary(df=samples, extra_values=df_summary,
                                       params=summary_params)

    return df_summary, table, samples
