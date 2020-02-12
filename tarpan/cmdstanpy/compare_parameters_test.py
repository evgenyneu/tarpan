import shutil
import os

from tarpan.testutils.a01_eight_schools.eight_schools import (
    get_fit, get_fit_larger_uncertainties)

from tarpan.cmdstanpy.compare_parameters import save_compare_parameters


def test_save_tree_plot():
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
        assert "2.09 (+5.54, -2.09)" in data
