import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from tabulate import tabulate
from scipy.special import logsumexp
from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.shared.tree_plot import tree_plot, TreePlotParams
from tarpan.shared.compare import model_weights


"""
Prefix of the columns in Stan's output that contain log
probability density value for each observation. For example,
if lpd_column_name='possum', when output is expected to have
columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.
"""
LPD_COLUMN_NAME_DEFAULT = "lpd_pointwise"


# Results of WAIC calculations
@dataclass
class WaicData:
    """
    WAIC value (Widely Aplicable Information Criterion).
    It provides an estimate of models accuracy (out-of-the-sample deviance)
    and is used for comparing models. Smaller WAIC values are better.
    """
    waic: float

    """WAIC for each individual observation point."""
    waic_pointwise: List[float]

    """Standard error of WAIC (approximate)."""
    waic_std_err: float

    """
    LPPD (log-pointwise-predictive-density) value.
    LPPD is a measure of the accuracy of the model,
    larger is better. It describes how well the model corresponds to the
    observations.
    """
    lppd: float

    """LPPD for each individual observation point."""
    lppd_pointwise: List[float]

    """
    Penalty term, aka "effective number of parameters", which tells
    how much probabilities of observations vary across samples. The penalty
    The penalty is added to WAIC number. The purpose of this is
    to combat overfitting, so models with too many parameters will have larger
    penlties and larger WAIC.
    """
    penalty: float

    """Penatiles of all individual observations."""
    penalty_pointwise: List[float]


@dataclass
class WaicModelCompared:
    """Model name"""
    name: str

    """Results of WAIC calculations for the model"""
    waic_data: WaicData

    """
    waic value difference between this model and the model with smallest waic.
    """
    waic_difference_best: float = None

    """
    Estimate of the standard error of the `waic_difference_best`.
    """
    waic_difference_best_std_err: float = None

    """
    Approximate measure of the relevance of the model, higher numbers
    correspond to models that are more compatible with the data.
    """
    weight: float = None


def waic(fit, lpd_column_name=LPD_COLUMN_NAME_DEFAULT) -> WaicData:
    """
    Compute WAIC (Widely Aplicable Information Criterion).
    It provides an estimate of models accuracy (out-of-the-sample deviance)
    and is used for comparing models. Smaller WAIC values are better,
    but they only make sense when compared.

    Parameters
    ----------

    fit : cmdstanpy.stanfit.CmdStanMCMC
        Contains the samples from cmdstanpy.

    lpd_column_name : str
        Prefix of the columns in Stan's output that contain log
        probability density value for each observation. For example,
        if lpd_column_name='possum', when output is expected to have
        columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.

    Returns
    -------

    WaicData:
        Waic calculation result.
    """

    # Get log probability density of each observation.
    # The point_lpd is a 2D array with indexes:
    #   1. Sample
    #   2. Observation
    # ---------

    lpd_pointwise = fit.get_drawset(params=[lpd_column_name])
    n_samples = lpd_pointwise.shape[0]  # Number of samples
    n_observations = lpd_pointwise.shape[1]  # Number of observations

    # Compute LPPD (log-pointwise-predictive-density) values for each
    # observation. LPPD is a measure of the accuracy of the model,
    # larger is better. It describes how well the model corresponds to the
    # observations.
    # ----------

    lppd_pointwise = [
        (logsumexp(lpd_pointwise.iloc[:, i]) - math.log(n_samples))
        for i in range(n_observations)
    ]

    # Compute penalty term, aka "effective number of parameters"
    # Variable `penalty_pointwise` is an array, each item corresponds
    # to one observation point. Each item contains variance of log probability
    # density values from all samples for a single observation.

    penalty_pointwise = [
        lpd_pointwise.iloc[:, i].var()
        for i in range(n_observations)
    ]

    # Compute WAIC
    # -------

    penalty = sum(penalty_pointwise)
    lppd = sum(lppd_pointwise)
    waic = -2*(lppd - penalty)

    # Approximate standard error of WAIC using the central limit theorem
    # -------

    waic_pointwise = -2*(np.array(lppd_pointwise) -
                         np.array(penalty_pointwise))

    waic_std_err = math.sqrt(n_observations * waic_pointwise.var())

    result = WaicData(
        waic=waic,
        waic_pointwise=waic_pointwise,
        waic_std_err=waic_std_err,
        lppd=lppd,
        lppd_pointwise=lppd_pointwise,
        penalty=penalty,
        penalty_pointwise=penalty_pointwise,
    )

    return result


def compare_waic(models, lpd_column_name=LPD_COLUMN_NAME_DEFAULT) \
        -> List[WaicModelCompared]:
    """
    Compare models using WAIC (Widely Aplicable Information Criterion)
    to see which models are more compatible with the data.

    Parameters
    ----------

    models : dict
        key: str
            Model name
        value: cmdstanpy.stanfit.CmdStanMCMC
            Contains the samples from cmdstanpy to compare

    lpd_column_name : str
        Prefix of the columns in Stan's output that contain log
        probability density value for each observation. For example,
        if lpd_column_name='possum', when output is expected to have
        columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.

    Returns
    -------

    list of WaicModelCompared:
        List of WAIC comparisons. The list is sorted: models with
        lower WAIC falues (more compatible with data) come first.
    """

    waic_results = [
        WaicModelCompared(
            name=name,
            waic_data=waic(fit=fit, lpd_column_name=lpd_column_name)
        ) for name, fit in models.items()
    ]

    # Ensure all models have same number of observations
    # --------

    deviances_lengths = [
        len(result.waic_data.waic_pointwise) for result in waic_results
    ]

    if len(set(deviances_lengths)) > 1:
        raise AttributeError("Models have different number of data points")

    # Sort by WAIC, lower (better) first
    waic_results = sorted(waic_results, key=lambda x: x.waic_data.waic)

    best_model = waic_results[0]  # Model with smallest WAIC

    # Calculate model weights
    deviances = [result.waic_data.waic_pointwise for result in waic_results]
    weights = model_weights(deviances)

    # Calculate WAIC difference between models
    for i, model_result in enumerate(waic_results):
        n_points = len(model_result.waic_data.waic_pointwise)
        model_result.weight = weights[i]

        if i == 0:
            continue

        model_result.waic_difference_best = model_result.waic_data.waic - \
            best_model.waic_data.waic

        # Calculate standard error of the waic difference
        # ------------

        waic_difference_pointwise = \
            np.array(model_result.waic_data.waic_pointwise) - \
            np.array(best_model.waic_data.waic_pointwise)

        # Using the central limit theorem
        std_err = math.sqrt(n_points * waic_difference_pointwise.var())

        model_result.waic_difference_best_std_err = std_err

    return waic_results


def waic_compared_to_df(compared: List[WaicModelCompared]):
    """
    Convert waic comparison to Pandas data frame.

    Parameters
    ----------
    compared: List[WaicModelCompared]
        Results of comparing WAIC between multiple models.

    Returns
    -------
    Pandas' DataFrame:
        WAIC comparison results.
    """

    column_names = ["WAIC", "SE", "dWAIC", "dSE", "pWAIC", "Weight"]
    model_names = [item.name for item in compared]
    df = pd.DataFrame(index=model_names, columns=column_names)

    for item in compared:
        waic = item.waic_data

        df.loc[item.name] = [
            waic.waic,
            waic.waic_std_err,
            item.waic_difference_best,
            item.waic_difference_best_std_err,
            waic.penalty,
            item.weight
        ]

    return df


def save_compare_waic_csv(models,
                          lpd_column_name=LPD_COLUMN_NAME_DEFAULT,
                          info_path=InfoPath()):
    """
    Compare models using WAIC (Widely Aplicable Information Criterion)
    to see which models are more compatible with the data. The result
    is saved in a CSV file.

    Parameters
    ----------

    models : dict
        key: str
            Model name.
        value: cmdstanpy.stanfit.CmdStanMCMC
            Contains the samples from cmdstanpy to compare.

    lpd_column_name : str
        Prefix of the columns in Stan's output that contain log
        probability density value for each observation. For example,
        if lpd_column_name='possum', when output is expected to have
        columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.

    info_path : InfoPath
        Determines the location of the output file.
    """

    info_path.set_codefile()
    compared = compare_waic(models=models, lpd_column_name=lpd_column_name)
    save_compare_waic_csv_from_compared(compared=compared, info_path=info_path)


def save_compare_waic_csv_from_compared(compared, info_path=InfoPath()):
    """
    Compare models using WAIC (Widely Aplicable Information Criterion)
    to see which models are more compatible with the data. The result
    is saved in a CSV file.

    Parameters
    ----------

    compared : list of WaicModelCompared
        List of compared models.
    info_path : InfoPath
        Determines the location of the output file.
    """

    info_path.set_codefile()
    info_path = InfoPath(**info_path.__dict__)
    info_path.base_name = info_path.base_name or "compare_waic"
    info_path.extension = 'csv'
    df = waic_compared_to_df(compared)
    path = get_info_path(info_path)
    df.to_csv(path, index_label='Name')


def save_compare_waic_txt(models,
                          lpd_column_name=LPD_COLUMN_NAME_DEFAULT,
                          info_path=InfoPath()):
    """
    Compare models using WAIC (Widely Aplicable Information Criterion)
    to see which models are more compatible with the data. The result
    is saved in a text file.

    Parameters
    ----------

    models : dict
        key: str
            Model name.
        value: cmdstanpy.stanfit.CmdStanMCMC
            Contains the samples from cmdstanpy to compare.

    lpd_column_name : str
        Prefix of the columns in Stan's output that contain log
        probability density value for each observation. For example,
        if lpd_column_name='possum', when output is expected to have
        columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.

    info_path : InfoPath
        Determines the location of the output file.
    """

    info_path.set_codefile()
    compared = compare_waic(models=models, lpd_column_name=lpd_column_name)
    save_compare_waic_txt_from_compared(compared=compared, info_path=info_path)


def save_compare_waic_txt_from_compared(compared, info_path=InfoPath()):
    """
    Compare models using WAIC (Widely Aplicable Information Criterion)
    to see which models are more compatible with the data. The result
    is saved in a text file.

    Parameters
    ----------

    compared : list of WaicModelCompared
        List of compared models.

    info_path : InfoPath
        Determines the location of the output file.
    """

    info_path.set_codefile()
    info_path = InfoPath(**info_path.__dict__)
    info_path.base_name = info_path.base_name or "compare_waic"
    info_path.extension = 'txt'
    df = waic_compared_to_df(compared)
    table = tabulate(df, headers=list(df), floatfmt=".2f", tablefmt="pipe")
    path = get_info_path(info_path)

    with open(path, "w") as text_file:
        print(table, file=text_file)


def compare_waic_tree_plot(models, lpd_column_name=LPD_COLUMN_NAME_DEFAULT,
                           tree_plot_params: TreePlotParams = TreePlotParams()):
    """
    Make a plot that compares models using WAIC
    (Widely Aplicable Information Criterion).

    Parameters
    ----------

    models : dict
        key: str
            Model name.
        value: cmdstanpy.stanfit.CmdStanMCMC
            Contains the samples from cmdstanpy to compare.

    lpd_column_name : str
        Prefix of the columns in Stan's output that contain log
        probability density value for each observation. For example,
        if lpd_column_name='possum', when output is expected to have
        columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.

    Returns
    -------
    (fig, ax):
        fig: Matplotlib's figure
        ax: Matplotlib's axis
    """

    compared = compare_waic(models=models, lpd_column_name=lpd_column_name)

    return compare_waic_tree_plot_from_compared(
            compared=compared, tree_plot_params=tree_plot_params)


def compare_waic_tree_plot_from_compared(
        compared, tree_plot_params: TreePlotParams = TreePlotParams()):
    """
    Make a plot that compares models using WAIC
    (Widely Aplicable Information Criterion).

    Parameters
    ----------

    compared : list of WaicModelCompared
        List of compared models.

    Returns
    -------
    (fig, ax):
        fig: Matplotlib's figure
        ax: Matplotlib's axis
    """

    plot_groups = []
    tree_plot_params = TreePlotParams(**tree_plot_params.__dict__)

    if tree_plot_params.labels is None:
        tree_plot_params.labels = ["dWAIC", "WAIC"]

    if tree_plot_params.xlabel is None:
        tree_plot_params.xlabel = "WAIC (deviance)"

    if tree_plot_params.title is None:
        tree_plot_params.title = "Model comparison (smaller is better)"

    for model in reversed(compared):
        values = []
        waic = model.waic_data
        group = dict(name=model.name, values=values)
        plot_groups.append(group)

        # WAIC difference
        # --------

        value = dict(value=waic.waic, error_bars=[])

        if model.waic_difference_best_std_err is not None:
            error_bars = [
                waic.waic - model.waic_difference_best_std_err,
                waic.waic + model.waic_difference_best_std_err
            ]

            value = dict(value=waic.waic, error_bars=[error_bars])

        values.append(value)

        # WAIC value
        # --------

        error_bars = [
            waic.waic - waic.waic_std_err,
            waic.waic + waic.waic_std_err
        ]

        value = dict(value=waic.waic, error_bars=[error_bars])

        values.append(value)

    tree_plot_params.draw_zero_line_if_in_range = False
    fig, ax = tree_plot(groups=plot_groups, params=tree_plot_params)

    # Draw a vertical line through the best model
    # ----------

    model = compared[0]

    ax.axvline(x=model.waic_data.waic, linestyle='dashed',
               color=tree_plot_params.marker_edge_colors[0])

    return fig, ax


def save_compare_waic_tree_plot(
        models, lpd_column_name=LPD_COLUMN_NAME_DEFAULT,
        tree_plot_params: TreePlotParams = TreePlotParams(),
        info_path=InfoPath()):
    """
    Make a plot that compares models using WAIC
    (Widely Aplicable Information Criterion) and save it to a file.

    Parameters
    ----------

    models : list of dict
        List of model samples from cmdstanpy to compare.

        The dictionary has keys:
            name: str
                Model name
            fit: cmdstanpy.stanfit.CmdStanMCMC
                Contains the samples from cmdstanpy.

    lpd_column_name : str
        Prefix of the columns in Stan's output that contain log
        probability density value for each observation. For example,
        if lpd_column_name='possum', when output is expected to have
        columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.


    info_path : InfoPath
        Determines the location of the output file.
    """

    info_path.set_codefile()
    compared = compare_waic(models=models, lpd_column_name=lpd_column_name)

    save_compare_waic_tree_plot_from_compared(
        compared=compared,
        tree_plot_params=tree_plot_params,
        info_path=info_path)


def save_compare_waic_tree_plot_from_compared(
        compared,
        tree_plot_params: TreePlotParams = TreePlotParams(),
        info_path=InfoPath()):
    """
    Make a plot that compares models using WAIC
    (Widely Aplicable Information Criterion) and save it to a file.

    Parameters
    ----------

    compared : list of WaicModelCompared
        List of compared models.

    lpd_column_name : str
        Prefix of the columns in Stan's output that contain log
        probability density value for each observation. For example,
        if lpd_column_name='possum', when output is expected to have
        columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.


    info_path : InfoPath
        Determines the location of the output file.
    """

    info_path.set_codefile()
    info_path = InfoPath(**info_path.__dict__)

    fig, ax = compare_waic_tree_plot_from_compared(
        compared=compared,
        tree_plot_params=tree_plot_params)

    info_path.base_name = info_path.base_name or 'compare_waic'
    info_path.extension = info_path.extension or 'pdf'
    the_path = get_info_path(info_path)
    fig.savefig(the_path, dpi=info_path.dpi)
    plt.close(fig)
