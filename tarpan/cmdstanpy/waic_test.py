import pytest
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

    models = [
        dict(name="Itercept", fit=fit1_intercept),
        dict(name="Fungus+treatment", fit=fit2_fungus_treatment),
        dict(name="Treatment", fit=fit3_treatment)
    ]

    result = compare_waic(models=models)

    assert [model.name for model in result] == ['Fungus+treatment',
                                                'Treatment',
                                                'Itercept']

    assert [round(model.waic_data.waic, 2) for model in result] == \
        [361.45, 402.71, 405.93]

    assert [round(model.waic_data.waic_std_err, 2) for model in result] == \
        [13.34, 10.78, 11.29]

    difference = [
        None if model.waic_difference_best is None
        else round(model.waic_difference_best, 2)
        for model in result
    ]

    assert difference == [None, 41.27, 44.48]

    std_err = [
        None if model.waic_difference_best_std_err is None
        else round(model.waic_difference_best_std_err, 2)
        for model in result
    ]

    assert std_err == [None, 9.82, 11.55]

    assert [round(model.waic_data.penalty, 1) for model in result] == \
        [3.4, 2.6, 1.6]


def test_compare_waic__model_with_different_data_points():
    cars_fit = get_fit()
    plants_fit = get_fit1_intercept()

    models = [
        dict(name="Cars", fit=cars_fit),
        dict(name="Plants", fit=plants_fit)
    ]

    with pytest.raises(AttributeError,
                       match=r"different number of data points"):
        compare_waic(models=models)
