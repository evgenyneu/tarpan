from dataclasses import dataclass
from typing import List
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
    If k > 0.7, then the observation is an outlier that influences
    the posterior distribution.
    """
    pareto_k: List[float]


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
