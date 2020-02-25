from pytest import approx
from tarpan.testutils.a03_cars.cars import get_fit
from tarpan.cmdstanpy.waic import waic, compare_waic

from tarpan.testutils.a04_height.height import (
    get_fit1_intercept, get_fit2_fungus_treatment, get_fit3_treatment)


def test_waic():
    fit = get_fit()
    result = waic(fit)

    assert result.waic == approx(421.5135196466395, rel=1e-15)

    assert len(result.waic_pointwise) == 50
    assert result.waic_pointwise[0] == approx(7.284060083431996, rel=1e-15)
    assert result.waic_pointwise[49] == approx(7.324510608904949, rel=1e-15)

    assert result.waic_std_err == approx(16.327468671341204, rel=1e-15)
    assert result.lppd == approx(-206.5875738029627, rel=1e-15)

    assert len(result.lppd_pointwise) == 50
    assert result.lppd_pointwise[0] == approx(-3.6203241203579615, rel=1e-15)
    assert result.lppd_pointwise[49] == approx(-3.641419673133626, rel=1e-15)

    assert result.penalty == approx(4.169186020357044, rel=1e-15)

    assert len(result.penalty_pointwise) == 50

    assert result.penalty_pointwise[0] == approx(0.021705921358036437,
                                                 rel=1e-15)

    assert result.penalty_pointwise[49] == approx(0.020835631318848448,
                                                  rel=1e-15)


def test_compare_waic():
    fit1_intercept = get_fit1_intercept()
    fit2_fungus_treatment = get_fit2_fungus_treatment()
    fit3_treatment = get_fit3_treatment()
    compare_waic()
