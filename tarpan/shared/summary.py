"""Create summary of parameter distributions: mean, std, mode etc."""

from dataclasses import dataclass, field
import arviz as az
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import gaussian_kde
from typing import List
from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.shared.param_names import filter_param_names


@dataclass
class SummaryParams:
    # List of probabilities for HPDIs (highest posterior density intervals)
    # to be shown in summary
    hpdis: List = field(default_factory=lambda: [0.6827, 0.9545])

    def hpdi_percent(self):
        """
        Returns
        -------

        float : rounded HPDI percent value, i. e. 68 for 0.6827.
        """

        return [
            int(round(fraction, 2) * 100) for fraction in self.hpdis
        ]


def save_summary(samples, param_names=None, info_path=InfoPath(),
                 summary_params=SummaryParams()):
    """
    Generates and saves statistical summary of the samples using mean, std, mode, hpdi.

    Parameters
    ----------

    samples : Panda's dataframe

        Each column contains samples for a parameter.

    param_names : list of str

        Names of parameters to be included in the summary. Include all if None.

    info_path : InfoPath

        Path information for creating summaries.
    """

    info_path.set_codefile()
    column_names = list(samples)
    param_names = filter_param_names(column_names, param_names)
    samples = samples[param_names]  # Filter by column names
    df_summary, table = sample_summary(samples, params=summary_params)
    return save_summary_to_disk(df_summary, table, info_path)


def save_summary_to_disk(df_summary, txt_summary, info_path=InfoPath()):
    """
    Saves statistical summary of the samples using mean, std, mode, hpdi.

    Parameters
    ----------

    df_summary : cmdstanpy.stanfit.CmdStanMCMC

        Panda's dataframe containing the summary for all parameters.

    txt_summary : list of str

        Text of the summary table.

    info_path : InfoPath

        Path information for creating summaries.


    Returns
    --------

    Dict:

        "df" : Dataframe containing summary.

        "table" : text version of the summary.

        "path_txt": Path to txt summary file.

        "path_csv": Path to csv summary file.

    """
    info_path = InfoPath(**info_path.__dict__)
    info_path.base_name = info_path.base_name or "summary"
    info_path.extension = 'txt'
    path_to_summary_txt = get_info_path(info_path)
    info_path.extension = 'csv'
    path_to_summary_csv = get_info_path(info_path)

    with open(path_to_summary_txt, "w") as text_file:
        print(txt_summary, file=text_file)

    df_summary.to_csv(path_to_summary_csv, index_label='Name')

    return {
        "df": df_summary,
        "table": txt_summary,
        "path_txt": path_to_summary_txt,
        "path_csv": path_to_summary_csv,
    }


def get_mode(values):
    """
    Calculates mode of the gaussian distribution
    (value at which it is maximum)
    """

    kernel = gaussian_kde(values)

    # Evaluate the kernel at no more than `max_values` for performance
    max_values = 100
    take_every = int(len(values) / max_values)

    if take_every < 1:
        take_every = 1

    values = values[0::take_every].tolist()
    imax = np.argmax(kernel(values))
    return values[imax]


def sample_summary(df, extra_values=None, params=SummaryParams()):
    """
    Returns table showing statistical summary from the sample parameters:
    mean, std, mode, hpdi.

    Parameters
    ------------

    df : Panda's dataframe

        Contains parameter sample values: each column is a parameter.

    extra_values : Panda's dataframe

        Additional values to be shown for parameters. Indexes are
        parameter names, and columns contain additional values to
        be shown in summary.

    Returns
    -------
    Panda's dataframe
        Panda's dataframe containing the summary for all parameters.
    str
        text of the summary table
    """
    rows = []

    for column in df:
        values = df[column].to_numpy()
        mean = df[column].mean()
        std = df[column].std()
        mode = get_mode(df[column])

        summary_values = [column, mean, std, mode]

        for i, hpdi in enumerate(params.hpdis):
            hpdi_value = az.hpd(values, credible_interval=hpdi).tolist()

            if i == 0:
                # For the first interval, calculate upper and
                # lower uncertainties
                uncert_plus = hpdi_value[1] - mode
                uncert_minus = mode - hpdi_value[0]
                summary_values.append(uncert_plus)
                summary_values.append(uncert_minus)

            summary_values.append(hpdi_value[0])
            summary_values.append(hpdi_value[1])

        if extra_values is not None:
            # Add extra columns
            summary_values += extra_values.loc[column].values.tolist()

        rows.append(summary_values)

    headers = ['Name', 'Mean', 'Std', 'Mode', '+', '-']

    for hpdi_percent in params.hpdi_percent():
        headers.append(f'{hpdi_percent}CI-')
        headers.append(f'{hpdi_percent}CI+')

    if extra_values is not None:
        headers += extra_values.columns.values.tolist()

    formats = [".2f"] * len(headers)

    if 'N_Eff' in headers:
        formats[headers.index('N_Eff')] = ".0f"

    table = tabulate(rows, headers=headers, floatfmt=formats, tablefmt="pipe")

    df_summary = pd.DataFrame(rows, columns=headers, index=df.columns.values)
    df_summary.drop('Name', axis=1, inplace=True)
    return df_summary, table
