import shutil
import os
import pandas as pd
from tarpan.shared.pair_plot import save_pair_plot


def test_save_tree_plot():
    data = {
        "x": [1, 2, 2.1, 4, 5, 6, 3.2, 3, 4.4, 2],
        "y": [-1, -2.5, -2, -2.7, -5, -6, -2, -6, -3, -2.5]
    }

    df = pd.DataFrame(data)

    outdir = "tarpan/shared/model_info/pair_plot_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_pair_plot(df, param_names=["x", "y"])

    assert os.path.isfile(os.path.join(outdir, "pair_plot.pdf"))
