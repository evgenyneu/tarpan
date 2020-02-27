from dataclasses import dataclass
from typing import List
import math
import pandas as pd
import numpy as np
from scipy.special import logsumexp
from tarpan.cmdstanpy.waic import LPD_COLUMN_NAME_DEFAULT
from tarpan.cmdstanpy.psis_from_arviz import _psislw


# Results of PSIS calculations
@dataclass
class PsisData:
    """
    Compute PSIS (Pareto-smoothed importance sampling).
    It provides an estimate of models accuracy (out-of-the-sample deviance)
    and is used for comparing models. Smaller PSIS values are better.
    """
    psis: float

    """PSIS for each individual observation point."""
    psis_pointwise: List[float]

    """Standard error of PSIS (approximate)."""
    psis_std_err: float

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
    The penlaty is added to PSIS number. The purpose of this is
    to combat overfitting, so models with too many parameters will have larger
    penlaties and larger PSIS.
    """
    penalty: float

    """Penatiles of all individual observations."""
    penalty_pointwise: List[float]

    """
    Pareto k values for each observation. Used as to diagnose the
    reliability of the estimation of the model accurace.
    If k > 0.7, then the observation is a possible outlier that influences
    the posterior distribution.
    """
    pareto_k: List[float]


@dataclass
class PsisModelCompared:
    """Model name"""
    name: str

    """Results of PSIS calculations for the model"""
    psis_data: PsisData

    """
    PSIS value difference between this model and the model with smallest psis.
    """
    psis_difference_best: float = None

    """
    Estimate of the standard error of the `psis_difference_best`.
    """
    psis_difference_best_std_err: float = None

    """
    Largest value of pareto k. Values above 0.7 indicate possible outliers.
    """
    largest_pareto_k: float = None


def compare_psis(models, lpd_column_name=LPD_COLUMN_NAME_DEFAULT) \
        -> List[PsisModelCompared]:
    """
    Compare models using PSIS (Pareto-smoothed importance sampling)
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

    lpd_column_name : str
        Prefix of the columns in Stan's output that contain log
        probability density value for each observation. For example,
        if lpd_column_name='possum', when output is expected to have
        columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.

    Returns
    -------

    list of PsisModelCompared:
        List of PSIS comparisons. The list is sorted: models with
        lower PSIS falues (more compatible with data) come first.
    """

    psis_results = [
        PsisModelCompared(
            name=model["name"],
            psis_data=psis(fit=model["fit"], lpd_column_name=lpd_column_name)
        ) for model in models
    ]

    # Sort by PSIS, lower (better) first
    psis_results = sorted(psis_results, key=lambda x: x.psis_data.psis)

    best_model = psis_results[0]  # Model with smallest PSIS

    # Calculate PSIS difference between models
    for i, model_result in enumerate(psis_results):
        model_result.largest_pareto_k = max(model_result.psis_data.pareto_k)

        if i == 0:
            continue

        model_result.psis_difference_best = model_result.psis_data.psis - \
            best_model.psis_data.psis

        # Calculate standard error of the psis difference
        # ------------

        n_points = len(model_result.psis_data.psis_pointwise)
        n_points_preious = len(best_model.psis_data.psis_pointwise)

        if n_points != n_points_preious:
            raise AttributeError("Models have different number of data points")

        psis_difference_pointwise = \
            np.array(model_result.psis_data.psis_pointwise) - \
            np.array(best_model.psis_data.psis_pointwise)

        # Using the central limit theorem
        std_err = math.sqrt(n_points * psis_difference_pointwise.var())

        model_result.psis_difference_best_std_err = std_err

    return psis_results


def calculate_reff(fit):
    """
    Calculate number of effective samples divided by
    the number of actual samples.
    """
    df_summary = fit.summary()
    neff = float(df_summary[['N_Eff']].astype(int).mean())
    return neff / (fit.chains * fit.draws)


def psis(fit, lpd_column_name=LPD_COLUMN_NAME_DEFAULT) -> PsisData:
    """
    Compute PSIS (Pareto-smoothed importance sampling).
    It provides an estimate of models accuracy (out-of-the-sample deviance)
    and is used for comparing models. Smaller PSIS values are better,
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

    PsisData:
        PSIS calculation result.
    """

    # Get log probability density of each observation.
    # The point_lpd is a 2D array with indexes:
    #   1. Sample
    #   2. Observation
    # ---------

    log_likelihood = fit.get_drawset(params=[lpd_column_name]).to_numpy()
    n_samples = log_likelihood.shape[0]  # Number of samples
    n_observations = log_likelihood.shape[1]  # Number of observations

    # Compute LPPD (log-pointwise-predictive-density) values for each
    # observation. LPPD is a measure of the accuracy of the model,
    # larger is better. It describes how well the model corresponds to the
    # observations.
    # ----------

    lppd_pointwise = [
        (logsumexp(log_likelihood[:, i]) - math.log(n_samples))
        for i in range(n_observations)
    ]

    lppd = sum(lppd_pointwise)  # Total LPPD

    # Calculed smoothed log_weights using PSIS
    # --------

    reff = calculate_reff(fit)
    log_likelihood_transposed = log_likelihood.transpose()
    log_weights, pareto_k = psislw(-log_likelihood_transposed, reff)
    log_weights += log_likelihood_transposed
    log_weights = log_weights.transpose()

    # Calculate PSIS for all observations
    # -------

    psis_pointwise = np.apply_along_axis(logsumexp, axis=0, arr=log_weights)

    # Compute penalty term, aka "effective number of parameters"
    # Variable `penalty_pointwise` is an array, each item corresponds
    # to one observation point.
    # -------

    penalty_pointwise = lppd_pointwise - psis_pointwise
    penalty = sum(penalty_pointwise)  # Total penalty

    # Convert PSIS from LPPD score to deviance
    psis_pointwise *= -2

    # Calculate total PSIS value
    psis = sum(psis_pointwise)

    # Approximate standard error of PSIS using the central limit theorem
    psis_std_err = math.sqrt(n_observations * psis_pointwise.var())

    result = PsisData(
        psis=psis,
        psis_pointwise=psis_pointwise,
        psis_std_err=psis_std_err,
        lppd=lppd,
        lppd_pointwise=lppd_pointwise,
        penalty=penalty,
        penalty_pointwise=penalty_pointwise,
        pareto_k=pareto_k
    )

    return result


def psislw(log_weights, reff=1.0):
    """
    Pareto smoothed importance sampling (PSIS).

    The function is taken from arviz library `psislw` function (stats.py)
    (https://github.com/arviz-devs/arviz), version 0.6.1,
    distributed under Apache License Version 2.0.

    Original code comes from Aki Vehtari, Tuomas Sivula:
      https://github.com/avehtari/PSIS

    Theory: https://arxiv.org/abs/1507.02646v5

    The original arviz function was modified by removing xarray functionality.


    Parameters
    ----------
    log_weights : array
        Array of size (n_observations, n_samples)
    reff : float
        relative MCMC efficiency, `ess / n`

    Returns
    -------
    lw_out : array
        Smoothed log weights
    kss : array
        Pareto tail indices (Pareto k values)
    """

    n_samples = log_weights.shape[-1]

    # precalculate constants
    cutoff_ind = -int(np.ceil(min(n_samples / 5.0, 3 * (n_samples / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)
    k_min = 1.0 / 3

    result = np.apply_along_axis(_psislw, axis=1, arr=log_weights,
                                 cutoff_ind=cutoff_ind, cutoffmin=cutoffmin,
                                 k_min=k_min)

    log_weights = np.stack(result[:, 0], axis=0)
    pareto_k = result[:, 1]

    return log_weights, pareto_k


def psis_compared_to_df(compared: List[PsisModelCompared]):
    """
    Convert PSIS comparison to Pandas data frame.

    Parameters
    ----------
    compared: List[PsisModelCompared]
        Results of comparing WAIC between multiple models.

    Returns
    -------
    Pandas' DataFrame:
        PSIS comparison results.
    """

    column_names = ["PSIS", "SE", "dPSIS", "dSE", "pPSIS", "Max K"]
    model_names = [item.name for item in compared]
    df = pd.DataFrame(index=model_names, columns=column_names)

    for item in compared:
        psis = item.psis_data

        df.loc[item.name] = [
            psis.psis,
            psis.psis_std_err,
            item.psis_difference_best,
            item.psis_difference_best_std_err,
            psis.penalty,
            item.largest_pareto_k
        ]

    return df
