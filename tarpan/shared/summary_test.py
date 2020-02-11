import shutil
import os
import pandas as pd
from tarpan.shared.summary import save_summary


def test_save_tree_plot():
    data = {
        "x": [1, 2, 3, 4, 5, 6],
        "y": [-1, -2, -3, -4, -5, -6]
    }

    df = pd.DataFrame(data)

    outdir = "tarpan/shared/model_info/summary_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    result = save_summary(df, param_names=["x"])

    assert os.path.isfile(os.path.join(outdir, "summary.txt"))
    assert os.path.isfile(os.path.join(outdir, "summary.csv"))

    assert result["df"].shape == (1, 9)
    assert "1.87" in result["table"]
    assert os.path.join(outdir, "summary.txt") in result["path_txt"]
    assert os.path.join(outdir, "summary.csv") in result["path_csv"]
