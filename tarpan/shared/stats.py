import numpy as np


def hpdi(values, probability):
    """
    Computes HPDI (Highest Posterior Density Interval),
    which is the inteval of minimum width that includes the given
    `probability` (or proportion of the numbers)

    The following code is based on from `hpd` function from
    arviz library (https://github.com/arviz-devs/arviz), version 0.6.1,
    distributed under Apache License Version 2.0.

    Changes made to arviz code:
     * Kept only code for single, non-circular interval.

    Parameters
    ----------

    values: list of float
        List of numbers.
    probability: float
        Probability (or compatibility/credible interval), or fraction
        of numbers.

    Returns
    -------
    Tuple containing lower and upper boundaries of the interval.
    """

    values = np.sort(values)
    n = len(values)
    interval_idx_inc = int(np.floor(probability * n))
    n_intervals = n - interval_idx_inc
    interval_width = values[interval_idx_inc:] - values[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation. ")

    min_idx = np.argmin(interval_width)
    hdi_min = values[min_idx]
    hdi_max = values[min_idx + interval_idx_inc]

    return (hdi_min, hdi_max)
