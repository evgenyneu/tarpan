import shutil
import os
import pandas as pd
from tarpan.shared.histogram import save_histogram


def test_save_tree_plot():
    data = {
        "x": [1, 2, 3, 4, 5, 6, 2, 3, 4, 4, 1, 3, 4, 5, 6, 2],
        "y": [-1, -2, -3, -4, -5, -6, -2, -2, -1, -1, -1, -3, -3, -2, -1, -4]
    }

    df = pd.DataFrame(data)

    outdir = "tarpan/shared/model_info/histogram_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_histogram(df, param_names=["x", "y"])

    assert os.path.isfile(os.path.join(outdir, "histogram_01.pdf"))
