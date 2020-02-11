import shutil
import os
import pandas as pd
from tarpan.shared.tree_plot import TreePlotParams, save_tree_plot


def test_save_tree_plot():
    data = {
        "x": [1, 2, 3, 4, 5, 6],
        "y": [-1, -2, -3, -4, -5, -6]
    }

    df = pd.DataFrame(data)

    values_no_error_bars = [
        {
            "x": 1.1,
            "y": -3.1,
        }
    ]

    tree_params = TreePlotParams()

    tree_params.labels = [
        "Normal",
        "Exact",
    ]

    outdir = "tarpan/shared/model_info/tree_plot_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_tree_plot(
        [df], extra_values=values_no_error_bars, param_names=["mu", "tau"],
        tree_params=tree_params)

    assert os.path.isfile(os.path.join(outdir, "summary.pdf"))
