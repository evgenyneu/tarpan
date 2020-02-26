import warnings

from tarpan.cmdstanpy.psis import (
    psis)

from tarpan.testutils.a05_divorse.divorse import (
    get_fit1_divorse_age)


def test_psis():
    warnings.simplefilter("ignore")
    fit = get_fit1_divorse_age()

    result = psis(fit)

    
