from tarpan.shared.pair_plot import save_pair_plot
import pandas as pd


def run_model():

    data = {
        "x": [1, 2, 2.1, 4, 5, 6, 3.2, 3, 4.4, 2],
        "y": [-1, -2.5, -2, -2.7, -5, -6, -2, -6, -3, -2.5]
    }

    df = pd.DataFrame(data)

    save_pair_plot(df, param_names=['x', 'y'])


if __name__ == '__main__':
    run_model()
    print('We are done')
