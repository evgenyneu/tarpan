import math
from scipy.special import logsumexp


def waic(fit):
    """
    Compute WAIC (Widely Aplicable Information Criterion).
    It provides an estimate of models accuracy (out-of-the-sample deviance)
    and is used for comparing models. Smaller WAIC values are better.
    """

    # Get log probability density of each observation.
    # The point_lpd is a 2D array with indexes:
    #   * sample
    #   * observation
    point_lpd = fit.get_drawset(params=["point_log_probability_density"])
    n_samples = point_lpd.shape[0]  # Number of samples
    n_observations = point_lpd.shape[1]  # Number of observations

    # Compute LPPD (log-pointwise-predictive-density) values for each
    # observation. LPPD measure of the accuracy of the model
    # (larger is better)
    # ----------

    lppd = [
        (logsumexp(point_lpd.iloc[:, i]) - math.log(n_samples))
        for i in range(n_observations)
    ]

    # Compute penalty term pWAIC, aka "effective number of parameters"
    # pWAIC is an array, each item corresponds to one observation point.
    # Each item contains varianace of sample log probability density values
    # from all samples.
    pWAIC = [
        point_lpd.iloc[:, i].var()
        for i in range(n_observations)
    ]

    print(pWAIC)

    # scipy.misc.logsumexp

    return 32
