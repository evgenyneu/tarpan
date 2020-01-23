# Generate sampling output, to avoid running in unit tests

from cmdstanpy import CmdStanModel
import os
import pickle


def run_model(data_dir, path_to_fit):
    """
    Runs Stan model and saves fit to disk
    """

    model_path = "tarpan/testutils/a01_eight_schools/eight_schools.stan"
    model = CmdStanModel(stan_file=model_path)

    data = {
        "J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]
    }

    os.makedirs(data_dir, exist_ok=True)

    fit = model.sample(data=data, chains=4, cores=4,
                       sampling_iters=1000, warmup_iters=1000,
                       output_dir=f'./{data_dir}')

    with open(path_to_fit, 'wb') as file:
        pickle.dump(fit, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_fit():
    """
    Returns fit file for unit tests.
    """

    data_dir = "temp_data/a01_eight_schools"
    path_to_fit = os.path.join(data_dir, "fit.pkl")

    if not os.path.exists(path_to_fit):
        run_model(data_dir=data_dir, path_to_fit=path_to_fit)

    with open(path_to_fit, 'rb') as input:
        return pickle.load(input)
