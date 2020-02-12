import shutil
import os
import pandas as pd

from tarpan.shared.compare_parameters import (
    save_compare_parameters, format_parameter, CompareParametersType)


def test_format_parameter__text():
    data = {
        "Mode": 1.23,
        "+": 0.02,
        "-": 0.06
    }

    result = format_parameter(data, CompareParametersType.TEXT)

    assert result == "1.23 (+0.02, -0.06)"


def test_format_parameter__gitlab_latex():
    data = {
        "Mode": 1.23,
        "+": 0.02,
        "-": 0.06
    }

    result = format_parameter(data, CompareParametersType.GITLAB_LATEX)

    assert result == "$`1.23^{+0.02}_{-0.06}`$"


def test_save_tree_plot():
    data1 = {
        "x": [1, 2, 3, 4, 5, 6],
        "y": [-1, -2, -3, -4, -5, -6],
        "z": [40, 21, 32, 41, 11, 31]
    }

    df1 = pd.DataFrame(data1)

    data2 = {
        "x": [2, 3, 1, 1, 3, 4],
        "y": [-2.1, -2, -2, -3, -1, -4],
        "z": [23, 19, 21, 13, 29, 10]
    }

    df2 = pd.DataFrame(data2)

    outdir = "tarpan/shared/model_info/compare_parameters_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare_parameters([df1, df2], labels=['One', 'Two'],
                            param_names=["x", "y"])

    assert os.path.isfile(os.path.join(outdir, "parameters_compared.txt"))

    with open(os.path.join(outdir, "parameters_compared.txt"), 'r') as file:
        data = file.read()
        assert "-2.10 (+0.10, -1.90)" in data


def test_save_tree_plot__gitlab_latex():
    data1 = {
        "x": [1, 2, 3, 4, 5, 6],
        "y": [-1, -2, -3, -4, -5, -6],
        "z": [40, 21, 32, 41, 11, 31]
    }

    df1 = pd.DataFrame(data1)

    data2 = {
        "x": [2, 3, 1, 1, 3, 4],
        "y": [-2.1, -2, -2, -3, -1, -4],
        "z": [23, 19, 21, 13, 29, 10]
    }

    df2 = pd.DataFrame(data2)

    outdir = "tarpan/shared/model_info/compare_parameters_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare_parameters([df1, df2], labels=['One', 'Two'],
                            param_names=["x", "y"],
                            type=CompareParametersType.GITLAB_LATEX)

    assert os.path.isfile(os.path.join(outdir, "parameters_compared.txt"))

    with open(os.path.join(outdir, "parameters_compared.txt"), 'r') as file:
        data = file.read()
        assert "$`-2.10^{+0.10}_{-1.90}`$" in data
