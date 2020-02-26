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
    fit1_divorse_age = get_fit1_divorse_age()
    fit2_divorse_marriage = get_fit2_divorse_marriage()
    fit3_divorse_age_marriage = get_fit3_divorse_age_marriage()

    models = [
        dict(name="Divorse vs Age", fit=fit1_divorse_age),
        dict(name="Divorse vs Marriage", fit=fit2_divorse_marriage),
        dict(name="Divorse vs Age+Marriage", fit=fit3_divorse_age_marriage)
    ]

    result = compare_psis(models=models)
    #
    # assert [model.name for model in result] == ['Fungus+treatment',
    #                                             'Treatment',
    #                                             'Itercept']
    #
    # assert [round(model.waic_data.waic, 2) for model in result] == \
    #     [361.45, 402.71, 405.93]
    #
    # assert [round(model.waic_data.waic_std_err, 2) for model in result] == \
    #     [13.34, 10.78, 11.29]
    #
    # difference = [
    #     None if model.waic_difference_best is None
    #     else round(model.waic_difference_best, 2)
    #     for model in result
    # ]
    #
    # assert difference == [None, 41.27, 44.48]
    #
    # std_err = [
    #     None if model.waic_difference_best_std_err is None
    #     else round(model.waic_difference_best_std_err, 2)
    #     for model in result
    # ]
    #
    # assert std_err == [None, 9.82, 11.55]
    #
    # assert [round(model.waic_data.penalty, 1) for model in result] == \
    #     [3.4, 2.6, 1.6]
