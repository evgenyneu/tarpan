import pytest
from pytest import approx
import os
import shutil
import numpy as np
import pandas as pd
from tarpan.testutils.a03_cars.cars import get_fit

from tarpan.cmdstanpy.waic import (
    waic, compare_waic, save_compare_waic_csv, save_compare_waic_txt,
    waic_compared_to_df,
    WaicData, WaicModelCompared,
    compare_waic_tree_plot, save_compare_waic_tree_plot)

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

    models = {
        "Itercept": fit1_intercept,
        "Fungus+treatment": fit2_fungus_treatment,
        "Treatment": fit3_treatment
    }

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

    actual_weight = [
        round(model.weight, 5)
        for model in result
    ]

    assert actual_weight == [0.99986, 2e-05, 0.00012]


def test_compare_waic__model_with_different_data_points():
    cars_fit = get_fit()
    plants_fit = get_fit1_intercept()

    models = {
        "Cars": cars_fit,
        "Plants": plants_fit
    }

    with pytest.raises(AttributeError,
                       match=r"different number of data points"):
        compare_waic(models=models)


def test_waic_compared_to_df():
    compared = []

    for i in range(1, 4):
        waic = WaicData(
            waic=i,
            waic_pointwise=[i] * 3,
            waic_std_err=i * 1.1,
            lppd=i * 1.2,
            lppd_pointwise=[i * 1.2] * 3,
            penalty=i * 0.3,
            penalty_pointwise=[i * 0.3] * 3,
        )

        compared_element = WaicModelCompared(
            name=f"Model {i}",
            waic_data=waic,
            waic_difference_best=i * 1.3,
            waic_difference_best_std_err=i * 1.4,
            weight=i*1.7
        )

        compared.append(compared_element)

    result = waic_compared_to_df(compared=compared)

    assert len(result) == 3

    row = result.loc["Model 1"]
    assert row["WAIC"] == 1
    assert row["SE"] == 1.1
    assert row["dWAIC"] == 1.3
    assert row["dSE"] == 1.4
    assert row["pWAIC"] == 0.3
    assert row["Weight"] == 1.7

    row = result.loc["Model 2"]
    assert row["WAIC"] == 2
    assert row["SE"] == 2.2
    assert row["dWAIC"] == 2.6
    assert row["dSE"] == 2.8
    assert row["pWAIC"] == 0.6
    assert row["Weight"] == 3.4


def test_save_compare_waic_csv():
    fit1_intercept = get_fit1_intercept()
    fit2_fungus_treatment = get_fit2_fungus_treatment()
    fit3_treatment = get_fit3_treatment()

    models = {
        "Itercept": fit1_intercept,
        "Fungus+treatment": fit2_fungus_treatment,
        "Treatment": fit3_treatment
    }

    outdir = "tarpan/cmdstanpy/model_info/waic_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare_waic_csv(models=models)

    assert os.path.isfile(os.path.join(outdir, "compare_waic.csv"))

    df = pd.read_csv(os.path.join(outdir, "compare_waic.csv"),
                     index_col="Name")

    assert len(df) == 3

    row = df.loc["Fungus+treatment"]
    assert row["WAIC"] == approx(361.44, rel=1e-3)
    assert row["SE"] == approx(13.33, rel=1e-3)
    assert np.isnan(row["dWAIC"])
    assert np.isnan(row["dSE"])
    assert row["pWAIC"] == approx(3.4388, rel=1e-3)
    assert row["Weight"] == approx(0.99985, rel=1e-3)

    row = df.loc["Itercept"]
    assert row["WAIC"] == approx(405.93, rel=1e-3)
    assert row["SE"] == approx(11.292, rel=1e-3)
    assert row["dWAIC"] == approx(44.48, rel=1e-3)
    assert row["dSE"] == approx(11.55, rel=1e-3)
    assert row["pWAIC"] == approx(1.5745, rel=1e-3)
    assert row["Weight"] == approx(0.00012332, rel=1e-3)


def test_save_compare_waic_txt():
    fit1_intercept = get_fit1_intercept()
    fit2_fungus_treatment = get_fit2_fungus_treatment()
    fit3_treatment = get_fit3_treatment()

    models = {
        "Itercept": fit1_intercept,
        "Fungus+treatment": fit2_fungus_treatment,
        "Treatment": fit3_treatment
    }

    outdir = "tarpan/cmdstanpy/model_info/waic_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare_waic_txt(models=models)

    assert os.path.isfile(os.path.join(outdir, "compare_waic.txt"))

    with open(os.path.join(outdir, "compare_waic.txt"), 'r') as file:
        data = file.read()
        assert "dWAIC" in data
        assert "Treatment" in data
        assert "402.71" in data


def test_compare_waic_tree_plot():
    fit1_intercept = get_fit1_intercept()
    fit2_fungus_treatment = get_fit2_fungus_treatment()
    fit3_treatment = get_fit3_treatment()

    models = {
        "Itercept": fit1_intercept,
        "Fungus+treatment": fit2_fungus_treatment,
        "Treatment": fit3_treatment
    }

    fig, ax = compare_waic_tree_plot(models=models)

    assert ax.get_xlabel() == "WAIC (deviance)"


def test_save_compare_waic_tree_plot():
    fit1_intercept = get_fit1_intercept()
    fit2_fungus_treatment = get_fit2_fungus_treatment()
    fit3_treatment = get_fit3_treatment()

    models = {
        "Itercept": fit1_intercept,
        "Fungus+treatment": fit2_fungus_treatment,
        "Treatment": fit3_treatment
    }

    outdir = "tarpan/cmdstanpy/model_info/waic_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare_waic_tree_plot(models=models)

    assert os.path.isfile(os.path.join(outdir, "compare_waic.pdf"))
