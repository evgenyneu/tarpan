from dataclasses import dataclass
from typing import List
import math
import numpy as np
import arviz as az
from arviz.stats.stats import loo
from tarpan.cmdstanpy.waic import LPD_COLUMN_NAME_DEFAULT


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
    Penalty term, aka "effective number of parameters", which tells
    how much probabilities of observations vary across samples. The penalty
    The penlaty is added to PSIS number. The purpose of this is
    to combat overfitting, so models with too many parameters will have larger
    penlaties and larger PSIS.
    """
    penalty: float

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

    cmdstanpy_data = az.from_cmdstanpy(
        posterior=fit,
        log_likelihood=lpd_column_name
    )

    result = loo(cmdstanpy_data, pointwise=True)

    psis = float(result.loc["loo"])
    psis_pointwise = result.loc["loo_i"].values.tolist()
    psis_std_err = float(result.loc["loo_se"])
    penalty = float(result.loc["p_loo"])
    pareto_k = result.loc["pareto_k"].values.tolist()

    result = PsisData(
        psis=psis,
        psis_pointwise=psis_pointwise,
        psis_std_err=psis_std_err,
        penalty=penalty,
        pareto_k=pareto_k
    )

    return result


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
