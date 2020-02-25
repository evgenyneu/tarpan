import math
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.special import logsumexp

"""
Prefix of the columns in Stan's output that contain log
probability density value for each observation. For example,
if lpd_column_name='possum', when output is expected to have
columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.
"""
LPD_COLUMN_NAME_DEFAULT = "log_probability_density_pointwise"


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
    The penlaty is added to WAIC number. The purpose of this is
    to combat overfitting, so models with too many parameters will have larger
    penlaties and larger WAIC.
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


def compare_waic(models, lpd_column_name=LPD_COLUMN_NAME_DEFAULT):
    """
    Compare models using WAIC (Widely Aplicable Information Criterion)
    to see which models are more compatible with the data.

    Parameters
    ----------

    models : list of dict
        List of model samples from cmdstanpy to compare.

        The dictionary has keys:
            name: str
                Model name
            fit: cmdstanpy.stanfit.CmdStanMCMC
                Contains the samples from cmdstanpy.

    Returns
    -------

    list of WaicModelCompared:
        List of WAIC comparisons. The list is sorted: models with
        lower WAIC falues (more compatible with data) come first.
    """

    waic_results = [
        WaicModelCompared(
            name=model["name"],
            waic_data=waic(fit=model["fit"], lpd_column_name=lpd_column_name)
        ) for model in models
    ]

    # Sort by WAIC, lower (better) first
    waic_results = sorted(waic_results, key=lambda x: x.waic_data.waic)

    best_model = waic_results[0]  # Model with smallest WAIC

    # Calculate WAIC difference between models
    for i, model_result in enumerate(waic_results):
        if i == 0:
            continue

        model_result.waic_difference_best = model_result.waic_data.waic - \
            best_model.waic_data.waic

        # Calculate standard error of the waic difference
        # ------------

        waic_difference_pointwise = \
            np.array(model_result.waic_data.waic_pointwise) - \
            np.array(best_model.waic_data.waic_pointwise)

        n_points = len(model_result.waic_data.waic_pointwise)
        n_points_preious = len(best_model.waic_data.waic_pointwise)

        if n_points != n_points_preious:
            raise AttributeError("Models have different number of data points")

        # Using the central limit theorem
        std_err = math.sqrt(n_points * waic_difference_pointwise.var())

        model_result.waic_difference_best_std_err = std_err

    return waic_results
