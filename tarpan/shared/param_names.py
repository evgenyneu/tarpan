"""Manipulate names of model parameters"""

import re


"""Non-parameter columns returned by Stan"""
STAN_TECHNICAL_COLUMNS = [
    'lp__',
    'accept_stat__',
    'stepsize__',
    'treedepth__',
    'n_leapfrog__',
    'divergent__',
    'energy__']


def filter_param_names(stan_params, param_names=None):
    """
    Filter names of parmeters from `stan_params`
    based on names in `param_names`. The purpose of this function
    is to return numbered names like 'a.1', 'a.2', 'a.3' by just
    supplying 'a' in `param_names`.
    In addition, removes Stan's technical columns like 'stepsize__'.

    Parameters
    ---------

    stan_params: list of str
        Parameter names from stan.
        This can include names like 'a.1', 'a.2' etc.
    param_names: list of str
        Parameter names we want. It can include both numbered names
        ('a.1', 'a.2', 'a.3') and non-numbered ('a', 'b'). Return everything
        is None.

    Returns
    -------
    list of str:
        Filtered parameter names
    """

    # Remove technical columns
    stan_params = [
        a for a in stan_params if a not in STAN_TECHNICAL_COLUMNS]

    if param_names is not None:
        # If param_names contains 'a', we want both 'a' and
        # numbered parameter names 'a.1', 'a.2' etc.
        stan_params = [
            a for a in stan_params
            if a in param_names
            or (re.sub(r'\.[0-9]+\Z', '', a) in param_names)
        ]

    return stan_params
