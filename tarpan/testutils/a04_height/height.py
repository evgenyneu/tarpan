# Generate sampling output, to avoid running in unit tests

from cmdstanpy import CmdStanModel
import pandas as pd
from tarpan.cmdstanpy.cache import run
from tarpan.shared.info_path import InfoPath


def get_data1_intercept():
    # The data contains:
    #   * h0: initial height of plant
    #   * h1: final height
    #   * fungus: 0, 1 presence of fungus
    #   * treatment: 0, 1 presence of anti-fungal treatment
    # The data is borrowed from Statistical Rethinking textbook
    # by Richard McElreath. All rights belong to the author
    csv_path = "tarpan/testutils/a04_height/height.csv"
    df = pd.read_csv(csv_path)

    return {
        "n": len(df.index),
        "h0": df["h0"].tolist(),
        "h1": df["h1"].tolist()
    }


def run_model1_intecept(data, output_dir, sampling_iters, warmup_iters):
    """
    Runs Stan model and saves fit to disk
    """

    model_path = "tarpan/testutils/a04_height/stan_model/height1_intercept.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(data=data, chains=1, cores=1,
                        sampling_iters=sampling_iters,
                        warmup_iters=warmup_iters,
                        output_dir=output_dir,
                        seed=1)


def get_iters():
    return {"sampling_iters": 1000, "warmup_iters": 500}


def get_fit1_intercept():
    """
    Returns fit file for unit tests.
    """

    info_path = InfoPath(
                    path='temp_data',
                    dir_name="a04_height1_intercept",
                    sub_dir_name=InfoPath.DO_NOT_CREATE
                )

    iters = get_iters()
    data = get_data1_intercept()

    return run(info_path=info_path, func=run_model1_intecept, data=data,
               sampling_iters=iters["sampling_iters"],
               warmup_iters=iters["warmup_iters"])
