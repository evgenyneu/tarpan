import pandas as pd

from tarpan.shared.compare_parameters import (
    save_compare_parameters, CompareParametersType)


def run_model():
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

    save_compare_parameters([df1, df2],
                            labels=['Model 1', 'Model 2'],
                            type=CompareParametersType.TEXT,  # or GITLAB_LATEX
                            param_names=['x', 'y'])


if __name__ == '__main__':
    run_model()
    print('We are done')
