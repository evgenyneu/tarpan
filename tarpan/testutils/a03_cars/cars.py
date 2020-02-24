# Generate sampling output, to avoid running in unit tests

from cmdstanpy import CmdStanModel
import os
import pandas as pd
from tarpan.cmdstanpy.cache import run
from tarpan.shared.info_path import InfoPath


def get_data():
    csv_path = "tarpan/testutils/a03_cars/cars.csv"
    df = pd.read_csv(csv_path)

    return {
        "n": len(df.index),
        "dist": df["dist"].tolist(),
        "speed": df["speed"].tolist()
    }


def run_model(data, output_dir):
    """
    Runs Stan model and saves fit to disk
    """

    model_path = "tarpan/testutils/a03_cars/stan_model/cars.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(data=data, chains=1, cores=1,
                        sampling_iters=1000, warmup_iters=500,
                        output_dir=output_dir,
                        seed=1)


def get_fit():
    """
    Returns fit file for unit tests.
    """

    info_path = InfoPath(
                    path='temp_data',
                    dir_name="a03_cars",
                    sub_dir_name=InfoPath.DO_NOT_CREATE
                )

    data = get_data()
    return run(info_path=info_path, func=run_model, data=data)
