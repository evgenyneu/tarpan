"""Save a text summary comparing parameters from different models"""

import pandas as pd
from tabulate import tabulate
from enum import Enum
from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.shared.summary import SummaryParams, sample_summary
from tarpan.shared.param_names import filter_param_names


class CompareParametersType(Enum):
    """Style of presenting values"""

    """Value and uncertainty is presented as 1.23 (+0.02, -0.03)"""
    TEXT = 1

    """
    Value and uncertainty is presented in gitlab latex format:
        $`1.23^{0.02}_{0.03}`$
    """
    GITLAB_LATEX = 2


def format_value(value, type: CompareParametersType):
    """
    Converts value to text

    Parameters
    ----------

    value: float
        A value to be converte to text.

    type:
        Style of prepresenting a number in text
    """

    if type == CompareParametersType.TEXT:
        return f"{value:.2f}"
    else:
        value_txt = f"{value:.2f}"
        return f"$`{value_txt}`$"


def format_parameter(data, type: CompareParametersType):
    """
    Converts a parameter value to text

    Parameters
    ----------

    data:
        a dictionary or Panda's row containing parameter values

    type:
        Style of prepresenting a number in text
    """

    if type == CompareParametersType.TEXT:
        return f"{data['Mode']:.2f} (+{data['+']:.2f}, -{data['-']:.2f})"
    else:
        value = f"{data['Mode']:.2f}"
        upper = f"{data['+']:.2f}"
        lower = f"{data['-']:.2f}"

        return f"$`{value}^{{+{upper}}}_{{-{lower}}}`$"


def compare_parameters(
        models,
        labels,
        extra_values=[],
        type: CompareParametersType = CompareParametersType.TEXT,
        param_names=None,
        summary_params=SummaryParams()):
    """
    Create model parameters

    Parameters
    ----------

    models : list Panda's data frames
        List of data frames for each model, containg sample values for
        multiple parameters (one parameter is one data frame column).
        Supply multiple data frames to compare parameter distributions.

    labels : list of str
        Names of the models in `models` list.

    extra_values : list of dict
        Additional values to be shown in the table:

        [
            {
                "mu": 2.3,
                "sigma": 3.3
            }
        ]

    type : CompareParametersType
        Format of values in the text table.

    param_names : list of str
        Names of parameters. Include all if None.

    Returns
    --------
    df: Panda's data frame
        Table in Panda's format
    txt : str
        Table in text format
    """

    if len(models) == 0:
        raise ValueError('Models list is empty')
        return

    if (len(models) + len(extra_values)) != len(labels):
        raise ValueError('Models list length is different from labels')
        return

    samples = models[0]
    column_names = list(samples)
    param_names = filter_param_names(column_names, param_names)

    df = pd.DataFrame(index=labels, columns=param_names)
    param_names_filtered = None

    for samples, label in zip(models, labels):
        column_names = list(samples)

        if param_names_filtered is None:
            param_names_filtered = filter_param_names(column_names, param_names)

        samples = samples[param_names_filtered]
        df_summary, _ = sample_summary(samples, params=summary_params)

        values = [
            format_parameter(df_summary.loc[name], type)
            for name in param_names_filtered
        ]

        df.loc[label] = values

    # Add extra values
    # ---------------

    extra_labels = labels[len(models):]

    for data, label in zip(extra_values, extra_labels):
        column_names = list(data.keys())

        values = [
            format_value(data[name], type)
            for name in param_names_filtered
        ]

        df.loc[label] = values

    table = tabulate(df, headers=param_names_filtered, tablefmt="pipe",
                     stralign="right")

    return df, table


def save_compare_parameters(
        models,
        labels,
        extra_values=[],
        type: CompareParametersType = CompareParametersType.TEXT,
        param_names=None,
        info_path=InfoPath(),
        summary_params=SummaryParams()):
    """
    Saves a text table that compares model parameters

    Parameters
    ----------

    models : list Panda's data frames
        List of data frames for each model, containg sample values for
        multiple parameters (one parameter is one data frame column).
        Supply multiple data frames to compare parameter distributions.

    labels : list of str
        Names of the models in `models` list.

    extra_values : list of dict
        Additional values to be shown in the table:

        [
            {
                "mu": 2.3,
                "sigma": 3.3
            }
        ]

    type : CompareParametersType
        Format of values in the text table.

    param_names : list of str
        Names of parameters. Include all if None.

    info_path : InfoPath
        Path information for creating summaries.
    """

    info_path.set_codefile()

    df, table = compare_parameters(models=models,
                                   labels=labels,
                                   extra_values=extra_values,
                                   type=type,
                                   param_names=param_names,
                                   summary_params=summary_params)

    info_path = InfoPath(**info_path.__dict__)
    info_path.base_name = info_path.base_name or "parameters_compared"
    info_path.extension = 'txt'
    path_to_txt = get_info_path(info_path)

    with open(path_to_txt, "w") as text_file:
        print(table, file=text_file)
