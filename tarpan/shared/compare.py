import numpy as np
import scipy.stats as st


def model_weights(deviances, b_samples=1000):
    """
    Calculate weights of the model that can be used to compare them.

    Pseudo-Bayesian Model averaging using Akaike-type
    weighting. The weights are stabilized using the Bayesian bootstrap.

    The code is taken from `compare` function of stats.py file
    of arviz library (https://github.com/arviz-devs/arviz), version 0.6.1,
    distributed under Apache License Version 2.0.

    Parameters
    ----------
    deviances: 2D array
        An array containing WAIC or PSIS values for each model and observations
        Rows are models and columns are observations.

    b_samples: int
        Number of samples taken by the Bayesian bootstrap estimation.

    Returns
    -------
    list:
        Weight of each of the model. Higher value means the model
        is more compatible with the data.
    """

    ic_i_val = np.array(deviances).transpose()
    rows = ic_i_val.shape[0]
    cols = ic_i_val.shape[1]
    ic_i_val = ic_i_val * rows

    b_weighting = st.dirichlet.rvs(alpha=[1] * rows, size=b_samples,
                                   random_state=1)

    weights = np.zeros((b_samples, cols))
    z_bs = np.zeros_like(weights)

    for i in range(b_samples):
        z_b = np.dot(b_weighting[i], ic_i_val)
        u_weights = np.exp((z_b - np.min(z_b)) / -2)
        z_bs[i] = z_b
        weights[i] = u_weights / np.sum(u_weights)

    return weights.mean(axis=0).tolist()
