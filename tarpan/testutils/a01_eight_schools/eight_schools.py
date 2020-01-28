# Generate sampling output, to avoid running in unit tests

from cmdstanpy import CmdStanModel
import os
from tarpan.cmdstanpy.cache import run
from tarpan.shared.info_path import InfoPath


def get_data():
    return {
        "J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]
    }


def run_model(data, output_dir):
    """
    Runs Stan model and saves fit to disk
    """

    model_path = "tarpan/testutils/a01_eight_schools/eight_schools.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(data=data, chains=4, cores=4,
                        sampling_iters=1000, warmup_iters=1000,
                        output_dir=output_dir)


def get_fit():
    """
    Returns fit file for unit tests.
    """

    info_path = InfoPath(
                    path='temp_data',
                    dir_name="a01_eight_schools",
                    sub_dir_name=InfoPath.DO_NOT_CREATE
                )

    return run(info_path=info_path, func=run_model, data=get_data())


def get_fit_larger_uncertainties():
    """
    Returns fit file for unit tests, uses data with larger uncertainties.
    """

    info_path = InfoPath(
                    path='temp_data',
                    dir_name="a01_eight_schools_large_uncert",
                    sub_dir_name=InfoPath.DO_NOT_CREATE
                )

    run(info_path=info_path, func=run_model, data=get_data())

    # Use data with increased uncertainties
    data = get_data()
    data["sigma"] = [u * 2 for u in data["sigma"]]

    return run(info_path=info_path, func=run_model, data=data)
