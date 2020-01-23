"""Code related to cmdstan"""

"""Non-parameter columns returned by Stan"""
STAN_TECHNICAL_COLUMNS = [
    'lp__',
    'accept_stat__',
    'stepsize__',
    'treedepth__',
    'n_leapfrog__',
    'divergent__',
    'energy__']
