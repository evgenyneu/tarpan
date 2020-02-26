import warnings
from pytest import approx

from tarpan.cmdstanpy.psis import (
    psis)

from tarpan.testutils.a05_divorse.divorse import (
    get_fit1_divorse_age)


def test_psis():
    warnings.simplefilter("ignore")
    fit = get_fit1_divorse_age()

    result = psis(fit)

    assert result.psis == approx(126.20, rel=1e-4)
    assert result.psis_std_err == approx(12.867, rel=1e-4)
    assert result.penalty == approx(3.8525, rel=1e-4)

    assert len(result.psis_pointwise) == 50
    assert result.psis_pointwise[0] == approx(4.1269, rel=1e-4)
    assert result.psis_pointwise[49] == approx(1.9442, rel=1e-4)

    assert len(result.pareto_k) == 50
    assert result.pareto_k[0] == approx(0.0048670, rel=1e-4)
    assert result.pareto_k[12] == approx(0.80011, rel=1e-4)
    assert result.pareto_k[49] == approx(0.28713, rel=1e-4)
