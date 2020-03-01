from pytest import approx
import os
import shutil
import pandas as pd
import numpy as np
import pytest

from tarpan.cmdstanpy.psis import (
    psis, compare_psis, PsisData, PsisModelCompared, psis_compared_to_df,
    save_compare_psis_csv, save_compare_psis_txt,
    compare_psis_tree_plot, save_compare_psis_tree_plot,
    psis_pareto_k_plot, save_psis_pareto_k_plot,
    save_psis_pareto_k_plot_from_compared)

from tarpan.testutils.a03_cars.cars import get_fit

from tarpan.testutils.a04_height.height import get_fit1_intercept

from tarpan.testutils.a05_divorse.divorse import (
    get_fit1_divorse_age, get_fit2_divorse_marriage,
    get_fit3_divorse_age_marriage)


def test_psis():
    fit = get_fit1_divorse_age()

    result = psis(fit)

    assert result.psis == approx(126.26, rel=1e-4)
    assert result.psis_std_err == approx(12.981, rel=1e-4)
    assert result.penalty == approx(3.8938, rel=1e-4)

    assert len(result.psis_pointwise) == 50
    assert result.psis_pointwise[0] == approx(4.1495, rel=1e-4)
    assert result.psis_pointwise[49] == approx(1.9286, rel=1e-4)

    assert len(result.pareto_k) == 50
    assert result.pareto_k[0] == approx(0.093890, rel=1e-4)
    assert result.pareto_k[12] == approx(0.74203, rel=1e-4)
    assert result.pareto_k[49] == approx(0.13764, rel=1e-4)


def test_compare_psis():
    fit1_divorse_age = get_fit1_divorse_age()
    fit2_divorse_marriage = get_fit2_divorse_marriage()
    fit3_divorse_age_marriage = get_fit3_divorse_age_marriage()

    models = {
        "Divorse vs Age": fit1_divorse_age,
        "Divorse vs Marriage": fit2_divorse_marriage,
        "Divorse vs Age+Marriage": fit3_divorse_age_marriage
    }

    result = compare_psis(models=models)

    assert [model.name for model in result] == ['Divorse vs Age',
                                                'Divorse vs Age+Marriage',
                                                'Divorse vs Marriage']

    assert [round(model.psis_data.psis, 2) for model in result] == \
        [126.26, 127.07, 139.23]

    assert [round(model.psis_data.psis_std_err, 2) for model in result] == \
        [12.98, 12.38, 9.79]

    difference = [
        None if model.psis_difference_best is None
        else round(model.psis_difference_best, 2)
        for model in result
    ]

    assert difference == [None, 0.81, 12.98]

    std_err = [
        None if model.psis_difference_best_std_err is None
        else round(model.psis_difference_best_std_err, 2)
        for model in result
    ]

    assert std_err == [None, 1.14, 9.53]

    assert [round(model.psis_data.penalty, 1) for model in result] == \
        [3.9, 4.5, 2.9]

    actual_largest_k = [
        round(model.largest_pareto_k, 2)
        for model in result
    ]

    assert actual_largest_k == [0.74, 0.56, 0.39]

    actual_weight = [
        round(model.weight, 2)
        for model in result
    ]

    assert actual_weight == [0.57, 0.36, 0.08]


def test_compare_psis__model_with_different_data_points():
    cars_fit = get_fit()
    plant_fit = get_fit1_intercept()

    models = {
        "Cars": cars_fit,
        "Plants": plant_fit
    }

    with pytest.raises(AttributeError,
                       match=r"different number of data points"):
        compare_psis(models=models)


def test_psis_compared_to_df():
    compared = []

    for i in range(1, 4):
        psis = PsisData(
            psis=i,
            psis_pointwise=[i] * 3,
            psis_std_err=i * 1.1,
            lppd=i * 1.2,
            lppd_pointwise=[i * 1.2] * 3,
            penalty=i * 0.3,
            penalty_pointwise=[i * 0.3] * 3,
            pareto_k=[i * 1.5] * 3
        )

        compared_element = PsisModelCompared(
            name=f"Model {i}",
            psis_data=psis,
            psis_difference_best=i * 1.3,
            psis_difference_best_std_err=i * 1.4,
            largest_pareto_k=i * 1.6,
            weight=i * 1.7
        )

        compared.append(compared_element)

    result = psis_compared_to_df(compared=compared)

    assert len(result) == 3

    row = result.loc["Model 1"]
    assert row["PSIS"] == 1
    assert row["SE"] == 1.1
    assert row["dPSIS"] == 1.3
    assert row["dSE"] == 1.4
    assert row["pPSIS"] == 0.3
    assert row["MaxK"] == 1.6
    assert row["Weight"] == 1.7

    row = result.loc["Model 2"]
    assert row["PSIS"] == 2
    assert row["SE"] == 2.2
    assert row["dPSIS"] == 2.6
    assert row["dSE"] == 2.8
    assert row["pPSIS"] == 0.6
    assert row["MaxK"] == 3.2
    assert row["Weight"] == 3.4


def test_save_compare_psis_csv():
    fit1_divorse_age = get_fit1_divorse_age()
    fit2_divorse_marriage = get_fit2_divorse_marriage()
    fit3_divorse_age_marriage = get_fit3_divorse_age_marriage()

    models = {
        "Divorse vs Age": fit1_divorse_age,
        "Divorse vs Marriage": fit2_divorse_marriage,
        "Divorse vs Age+Marriage": fit3_divorse_age_marriage
    }

    outdir = "tarpan/cmdstanpy/model_info/psis_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare_psis_csv(models=models)

    assert os.path.isfile(os.path.join(outdir, "compare_psis.csv"))

    df = pd.read_csv(os.path.join(outdir, "compare_psis.csv"),
                     index_col="Name")

    assert len(df) == 3

    row = df.loc["Divorse vs Age"]
    assert row["PSIS"] == approx(126.25, rel=1e-3)
    assert row["SE"] == approx(12.981, rel=1e-3)
    assert np.isnan(row["dPSIS"])
    assert np.isnan(row["dSE"])
    assert row["pPSIS"] == approx(3.8938, rel=1e-3)
    assert row["MaxK"] == approx(0.74203, rel=1e-3)
    assert row["Weight"] == approx(0.5665, rel=1e-3)

    row = df.loc["Divorse vs Age+Marriage"]
    assert row["PSIS"] == approx(127.06, rel=1e-3)
    assert row["SE"] == approx(12.3822, rel=1e-3)
    assert row["dPSIS"] == approx(0.80989, rel=1e-3)
    assert row["dSE"] == approx(1.14009, rel=1e-3)
    assert row["pPSIS"] == approx(4.45447, rel=1e-3)
    assert row["MaxK"] == approx(0.56233, rel=1e-3)
    assert row["Weight"] == approx(0.3552, rel=1e-3)


def test_save_compare_psis_txt():
    fit1_divorse_age = get_fit1_divorse_age()
    fit2_divorse_marriage = get_fit2_divorse_marriage()
    fit3_divorse_age_marriage = get_fit3_divorse_age_marriage()

    models = {
        "Divorse vs Age": fit1_divorse_age,
        "Divorse vs Marriage": fit2_divorse_marriage,
        "Divorse vs Age+Marriage": fit3_divorse_age_marriage
    }

    outdir = "tarpan/cmdstanpy/model_info/psis_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare_psis_txt(models=models)

    assert os.path.isfile(os.path.join(outdir, "compare_psis.txt"))

    with open(os.path.join(outdir, "compare_psis.txt"), 'r') as file:
        data = file.read()
        assert "dPSIS" in data
        assert "Divorse vs Age" in data
        assert "12.98" in data


def test_compare_psis_tree_plot():
    fit1_divorse_age = get_fit1_divorse_age()
    fit2_divorse_marriage = get_fit2_divorse_marriage()
    fit3_divorse_age_marriage = get_fit3_divorse_age_marriage()

    models = {
        "Divorse vs Age": fit1_divorse_age,
        "Divorse vs Marriage": fit2_divorse_marriage,
        "Divorse vs Age+Marriage": fit3_divorse_age_marriage
    }

    fig, ax = compare_psis_tree_plot(models=models)

    assert ax.get_xlabel() == "PSIS (deviance)"


def test_save_compare_psis_tree_plot():
    fit1_divorse_age = get_fit1_divorse_age()
    fit2_divorse_marriage = get_fit2_divorse_marriage()
    fit3_divorse_age_marriage = get_fit3_divorse_age_marriage()

    models = {
        "Divorse vs Age": fit1_divorse_age,
        "Divorse vs Marriage": fit2_divorse_marriage,
        "Divorse vs Age+Marriage": fit3_divorse_age_marriage
    }

    outdir = "tarpan/cmdstanpy/model_info/psis_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare_psis_tree_plot(models=models)

    assert os.path.isfile(os.path.join(outdir, "compare_psis.pdf"))


def test_psis_pareto_k_plot():
    fit_divorse_age = get_fit1_divorse_age()

    fig, ax = psis_pareto_k_plot(fit=fit_divorse_age, name="Divorse vs age")

    assert ax.get_xlabel() == "Point index"
    assert ax.get_title() == "'Divorse vs age' Pareto K values"


def test_save_psis_pareto_k_plot():
    fit_divorse_age = get_fit1_divorse_age()

    outdir = "tarpan/cmdstanpy/model_info/psis_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_psis_pareto_k_plot(fit=fit_divorse_age, name="Divorse vs age")

    assert os.path.isfile(os.path.join(outdir, "pareto_k_divorse_vs_age.pdf"))


def test_save_psis_pareto_k_plot_from_compared():
    outdir = "tarpan/cmdstanpy/model_info/psis_test"

    compared = []

    for i in range(1, 4):
        psis = PsisData(
            psis=i,
            psis_pointwise=[i] * 3,
            psis_std_err=i * 1.1,
            lppd=i * 1.2,
            lppd_pointwise=[i * 1.2] * 3,
            penalty=i * 0.3,
            penalty_pointwise=[i * 0.3] * 3,
            pareto_k=[i * 1.5] * 3
        )

        compared_element = PsisModelCompared(
            name=f"Model {i}",
            psis_data=psis,
            psis_difference_best=i * 1.3,
            psis_difference_best_std_err=i * 1.4,
            largest_pareto_k=i * 1.6,
            weight=i * 1.7
        )

        compared.append(compared_element)

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_psis_pareto_k_plot_from_compared(compared=compared)

    assert os.path.isfile(os.path.join(outdir, "pareto_k_model_1.pdf"))
    assert os.path.isfile(os.path.join(outdir, "pareto_k_model_2.pdf"))
    assert os.path.isfile(os.path.join(outdir, "pareto_k_model_3.pdf"))
