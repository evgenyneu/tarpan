import shutil
import os

from tarpan.testutils.a01_eight_schools.eight_schools import (
    get_fit, get_fit_larger_uncertainties)

from tarpan.cmdstanpy.compare_parameters import save_compare_parameters


def test_save_compare_parameters():
    fit = get_fit()
    fit2 = get_fit_larger_uncertainties()
    fits = [fit, fit2]

    outdir = "tarpan/cmdstanpy/model_info/compare_parameters_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare_parameters(fits, labels=['One', 'Two'],
                            param_names=["mu", "tau", "eta.1"])

    assert os.path.isfile(os.path.join(outdir, "parameters_compared.txt"))

    with open(os.path.join(outdir, "parameters_compared.txt"), 'r') as file:
        data = file.read()
        assert "2.36 (+5.41, -2.35)" in data


def test_save_compare_parameters__extra_values():
    fit = get_fit()
    fit2 = get_fit_larger_uncertainties()
    fits = [fit, fit2]

    outdir = "tarpan/cmdstanpy/model_info/compare_parameters_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    extra_values = [
        {
            "mu": 1.1,
            "tau": 2.1,
            "eta.1": 3.1,
            "eta.2": 5.2
        },
        {
            "mu": 10.1,
            "tau": 20.1,
            "eta.1": 30.1,
            "eta.2": 35.3
        }
    ]

    save_compare_parameters(fits, labels=['One', 'Two', 'Extra 1', 'Extra 2'],
                            extra_values=extra_values,
                            param_names=["mu", "tau", "eta.1"])

    assert os.path.isfile(os.path.join(outdir, "parameters_compared.txt"))

    with open(os.path.join(outdir, "parameters_compared.txt"), 'r') as file:
        data = file.read()
        assert "2.36 (+5.41, -2.35)" in data
        assert "Extra 1" in data
        assert "Extra 2" in data
        assert "30.1" in data
