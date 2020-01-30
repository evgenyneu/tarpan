import numpy as np


def kde(eval_points, values, uncertainties):
    """
    Computer Gaussian KDE (kernel density estimator) using values and their
    uncertianties by adding PDFs of Normal distributions at each value that
    have mean equal to the value and standard deviation equal to value's
    uncertainty.

    Parameters
    ----------

    eval_points: list or numpy.ndarray
        Value at which KDE will be evaluated.
    values: list or numpy.ndarray
        Values for computing KDE.
    uncertainties: list or numpy.ndarray
        Corresponding uncertainties for the `values`.

    Returns
    -------
    numpy.ndarray
        Values of the KDE corresponding to `eval_points`.
    """

    x = np.array(eval_points)
    mu = np.array(values)
    sigma = np.array(uncertainties)

    if mu.shape[0] == 0:
        return np.array([])

    if mu.shape[0] != sigma.shape[0]:
        raise ValueError('Values and uncertainties\
are lists of different lengths')

    guassian = -0.5 * ((x[:, np.newaxis] - mu[np.newaxis, :]) / sigma)**2
    gaussian = np.exp(guassian)
    gaussian /= np.sqrt(2 * np.pi) * sigma
    return gaussian.sum(axis=1) / len(values)
