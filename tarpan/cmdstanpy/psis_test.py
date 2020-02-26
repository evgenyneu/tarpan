import warnings
from pytest import approx

from tarpan.cmdstanpy.psis import (
    psis, compare_psis)

from tarpan.testutils.a05_divorse.divorse import (
    get_fit1_divorse_age, get_fit2_divorse_marriage,
    get_fit3_divorse_age_marriage)


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


def test_compare_waic():
    warnings.simplefilter("ignore")
    fit1_divorse_age = get_fit1_divorse_age()
    fit2_divorse_marriage = get_fit2_divorse_marriage()
    fit3_divorse_age_marriage = get_fit3_divorse_age_marriage()

    models = [
        dict(name="Divorse vs Age", fit=fit1_divorse_age),
        dict(name="Divorse vs Marriage", fit=fit2_divorse_marriage),
        dict(name="Divorse vs Age+Marriage", fit=fit3_divorse_age_marriage)
    ]

    result = compare_psis(models=models)

    assert [model.name for model in result] == ['Divorse vs Age',
                                                'Divorse vs Age+Marriage',
                                                'Divorse vs Marriage']

    assert [round(model.psis_data.psis, 2) for model in result] == \
        [126.2, 127.07, 139.24]

    assert [round(model.psis_data.psis_std_err, 2) for model in result] == \
        [12.87, 12.39, 9.79]

    difference = [
        None if model.psis_difference_best is None
        else round(model.psis_difference_best, 2)
        for model in result
    ]

    assert difference == [None, 0.87, 13.04]

    std_err = [
        None if model.psis_difference_best_std_err is None
        else round(model.psis_difference_best_std_err, 2)
        for model in result
    ]

    assert std_err == [None, 1.08, 9.48]

    assert [round(model.psis_data.penalty, 1) for model in result] == \
        [3.9, 4.5, 2.9]

    actual_largest_k = [
        round(model.largest_pareto_k, 2)
        for model in result
    ]

    assert actual_largest_k == [0.8, 0.56, 0.34]
