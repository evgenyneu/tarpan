#
# The following code is taken from arviz library
# (https://github.com/arviz-devs/arviz), version 0.6.1,
# distributed under Apache License Version 2.0.
#
# Changes made to arviz code:
#  * Replaced _logsumexp with logsumexp from scipy.special
#
# Original code comes from Aki Vehtari, Tuomas Sivula:
#   https://github.com/avehtari/PSIS
#
# Theory: https://arxiv.org/abs/1507.02646v5
#

import numpy as np
from scipy.special import logsumexp


def _psislw(log_weights, cutoff_ind, cutoffmin, k_min=1.0 / 3):
    """
    Pareto smoothed importance sampling (PSIS) for a 1D vector.

    Parameters
    ----------
    log_weights : array
        Array of length n_observations
    cutoff_ind : int
    cutoffmin : float
    k_min : float

    Returns
    -------
    lw_out : array
        Smoothed log weights
    kss : float
        Pareto tail index
    """
    x = np.asarray(log_weights)

    # improve numerical accuracy
    x -= np.max(x)
    # sort the array
    x_sort_ind = np.argsort(x)
    # divide log weights into body and right tail
    xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)

    expxcutoff = np.exp(xcutoff)
    (tailinds,) = np.where(x > xcutoff)  # pylint: disable=unbalanced-tuple-unpacking
    x_tail = x[tailinds]
    tail_len = len(x_tail)
    if tail_len <= 4:
        # not enough tail samples for gpdfit
        k = np.inf
    else:
        # order of tail samples
        x_tail_si = np.argsort(x_tail)
        # fit generalized Pareto distribution to the right tail samples
        x_tail = np.exp(x_tail) - expxcutoff
        k, sigma = _gpdfit(x_tail[x_tail_si])

        if k >= k_min:
            # no smoothing if short tail or GPD fit failed
            # compute ordered statistic for the fit
            sti = np.arange(0.5, tail_len) / tail_len
            smoothed_tail = _gpinv(sti, k, sigma)
            smoothed_tail = np.log(  # pylint: disable=assignment-from-no-return
                smoothed_tail + expxcutoff
            )
            # place the smoothed tail into the output array
            x[tailinds[x_tail_si]] = smoothed_tail
            # truncate smoothed values to the largest raw weight 0
            x[x > 0] = 0
    # renormalize weights
    x -= logsumexp(x)

    return x, k


def _gpdfit(ary):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).

    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.

    Parameters
    ----------
    ary : array
        sorted 1D data array

    Returns
    -------
    k : float
        estimated shape parameter
    sigma : float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(ary)
    m_est = 30 + int(n ** 0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]

    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)  # pylint: disable=no-member
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    # remove negligible weights
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    # normalise weights
    weights /= weights.sum()

    # posterior mean for b
    b_post = np.sum(b_ary * weights)
    # estimate for k
    k_post = np.log1p(-b_post * ary).mean()  # pylint: disable=invalid-unary-operand-type,no-member
    # add prior for k_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    sigma = -k_post / b_post

    return k_post, sigma


def _gpinv(probs, kappa, sigma):
    """Inverse Generalized Pareto distribution function."""
    # pylint: disable=unsupported-assignment-operation, invalid-unary-operand-type
    x = np.full_like(probs, np.nan)
    if sigma <= 0:
        return x
    ok = (probs > 0) & (probs < 1)
    if np.all(ok):
        if np.abs(kappa) < np.finfo(float).eps:
            x = -np.log1p(-probs)
        else:
            x = np.expm1(-kappa * np.log1p(-probs)) / kappa
        x *= sigma
    else:
        if np.abs(kappa) < np.finfo(float).eps:
            x[ok] = -np.log1p(-probs[ok])
        else:
            x[ok] = np.expm1(-kappa * np.log1p(-probs[ok])) / kappa
        x *= sigma
        x[probs == 0] = 0
        if kappa >= 0:
            x[probs == 1] = np.inf
        else:
            x[probs == 1] = -sigma / kappa
    return x
