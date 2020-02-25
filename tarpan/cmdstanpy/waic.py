import math
import numpy as np
from dataclasses import dataclass, field
from typing import List
from scipy.special import logsumexp


@dataclass
class WaicData:
    """
    WAIC value (Widely Aplicable Information Criterion).
    It provides an estimate of models accuracy (out-of-the-sample deviance)
    and is used for comparing models. Smaller WAIC values are better.
    """
    waic: float

    """
    WAIC for each individual observation point.
    """
    waic_pointwise: List[float]

    """
    Standard error of WAIC (approximate).
    """
    waic_std_err: float

    """
    LPPD (log-pointwise-predictive-density) value.
    LPPD is a measure of the accuracy of the model,
    larger is better. It describes how well the model corresponds to the
    observations.
    """
    lppd: float

    """
    LPPD for each individual observation point.
    """
    lppd_pointwise: List[float]

    """
    Penalty term, aka "effective number of parameters", which tells
    how much probabilities of observations vary across samples. The penalty
    The penlaty is added to WAIC number. The purpose of this is
    to combat overfitting, so models with too many parameters will have larger
    penlaties and larger WAIC.
    """
    penalty: float

    """
    Penatiles of all individual observations.
    """
    penalty_pointwise: List[float]


def waic(fit, lpd_column_name="log_probability_density_pointwise") -> WaicData:
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


def compare_waic():
    pass
