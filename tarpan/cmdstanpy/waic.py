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
    LPPD (log-pointwise-predictive-density) value.
    LPPD is a measure of the accuracy of the model,
    larger is better. It describes how well the model corresponds to the
    observations.
    """
    lppd: float

    """
    LPPD for each individual observations point.
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


def waic(fit) -> WaicData:
    """
    Compute WAIC (Widely Aplicable Information Criterion).
    It provides an estimate of models accuracy (out-of-the-sample deviance)
    and is used for comparing models. Smaller WAIC values are better.
    """

    # Get log probability density of each observation.
    # The point_lpd is a 2D array with indexes:
    #   1. Sample
    #   2. Observation
    # ---------

    lpd_pointwise = fit.get_drawset(params=["log_probability_density_pointwise"])
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
    print(f'waic={waic}')
    print(f'lppd={lppd}')
    print(f'penalty={penalty}')

    # Approximate standard error of WAIC using the central limit theorem
    # -------

    waic_pointwise = -2*(np.array(lppd) - np.array(penalty_pointwise))
    waic_std_err = math.sqrt(n_observations * waic_pointwise.var())
    print(f'waic_std_err={waic_std_err}')

    result = WaicData(
        waic=waic,
        lppd=lppd,
        lppd_pointwise=lppd_pointwise,
        penalty=penalty
    )

    return result
