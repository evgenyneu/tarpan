from tarpan.shared.histogram import save_histogram
import pandas as pd


def run_model():
    data = {
        "x": [1, 2, 3, 4, 5, 6, 2, 3, 4, 4, 1, 3, 4, 5, 6, 2],
        "y": [-1, -2, -3, -4, -5, -6, -2, -2, -1, -1, -1, -3, -3, -2, -1, -4]
    }

    df = pd.DataFrame(data)

    save_histogram(df, param_names=['x', 'y'])


if __name__ == '__main__':
    run_model()
    print('We are done')
