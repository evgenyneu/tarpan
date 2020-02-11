import shutil
import os
import pandas as pd
from tarpan.shared.analyse import save_analysis


def test_save_analysis():
    data = {
        "x": [1, 2, 3, 4, 5, 6, 2, 3, 4, 4, 1, 3, 4, 5, 6, 2],
        "y": [-1, -2, -3, -4, -5, -6, -2, -2, -1, -1, -1, -3, -3, -2, -1, -4]
    }

    df = pd.DataFrame(data)

    outdir = "tarpan/shared/model_info/analyse_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_analysis(df)

    # Check summary files
    # ---------------

    assert os.path.isfile(os.path.join(outdir, "summary.txt"))
    assert os.path.isfile(os.path.join(outdir, "summary.csv"))

    # Tree plot
    assert os.path.isfile(os.path.join(outdir, "summary.pdf"))

    # Histogram of posterior distributions
    # ----------

    assert os.path.isfile(os.path.join(outdir, "histogram_01.pdf"))

    # Pair plot
    assert os.path.isfile(os.path.join(outdir, "pair_plot.pdf"))
