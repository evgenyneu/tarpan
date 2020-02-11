from tarpan.shared.summary import save_summary
import pandas as pd


def run_model():

    data = {
        "x": [1, 2, 3, 4, 5, 6],
        "y": [-1, -2, -3, -4, -5, -6]
    }

    df = pd.DataFrame(data)

    save_summary(df, param_names=['x', 'y'])


if __name__ == '__main__':
    run_model()
    print('We are done')
