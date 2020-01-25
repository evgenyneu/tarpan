import shutil
import os

from tarpan.testutils.a01_eight_schools.eight_schools import (
    get_fit, get_fit_larger_uncertainties)

from tarpan.cmdstanpy.tree_plot import save_tree_plot
from tarpan.shared.tree_plot import TreePlotParams


def test_save_tree_plot():
    fit = get_fit()
    fit2 = get_fit_larger_uncertainties()
    fits = [fit, fit2]

    values_no_error_bars = [
        {
            "mu": 1.1,
            "tau": 3.1,
        }
    ]

    tree_params = TreePlotParams()

    tree_params.labels = [
        "Normal",
        "Larger uncertainties",
        "Exact",
    ]

    outdir = "tarpan/cmdstanpy/model_info/tree_plot_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_tree_plot(
        fits, extra_values=values_no_error_bars, param_names=["mu", "tau"],
        tree_params=tree_params)

    assert os.path.isfile(os.path.join(outdir, "summary.pdf"))
