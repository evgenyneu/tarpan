# Generate sampling output, to avoid running in unit tests

from cmdstanpy import CmdStanModel
from tarpan.cmdstanpy.cache import run
from tarpan.shared.info_path import InfoPath


def get_data1():
    values1 = [-0.99, -1.37, -1.38, -1.51, -1.29, -1.34, -1.50, -0.93, -0.83,
               -1.46, -1.07, -1.28, -0.73]

    uncertainties1 = [0.12, 0.05, 0.11, 0.18, 0.03, 0.19, 0.18, 0.12, 0.19,
                      0.09, 0.11, 0.16, 0.08]

    return {
        "y": values1,
        "uncertainties": uncertainties1,
        "N": len(values1)
    }


def get_data2():
    values2 = [
            -1.22, -1.15, -0.97, -0.68, -0.37, -0.48, -0.73, -0.61, -1.32,
            -0.62, -1.13, -0.65, -0.90, -1.29, -1.19, -0.54, -0.64, -0.45,
            -1.21, -0.75, -0.66, -0.71, -0.61, -0.59, -1.07, -0.65, -0.59]

    uncertainties2 = [
         0.13, 0.14, 0.17, 0.07, 0.11, 0.12, 0.23, 0.05, 0.04,
         0.30, 0.11, 0.13, 0.16, 0.03, 0.18, 0.20, 0.16, 0.16,
         0.11, 0.09, 0.20, 0.10, 0.08, 0.04, 0.04, 0.23, 0.19]

    return {
        "y": values2,
        "uncertainties": uncertainties2,
        "N": len(values2)
    }


def run_model(data, output_dir):
    """
    Runs Stan model and saves fit to disk
    """

    model_path = "tarpan/testutils/a02_gaussian_mixture/stan_model/gaussian_mixture.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(
        data=data, seed=333,
        adapt_delta=0.90, max_treedepth=5,
        sampling_iters=1000, warmup_iters=1000,
        chains=4, cores=4,
        output_dir=output_dir)


def get_fit1():
    """
    Returns fit file for unit tests, uses data with larger uncertainties.
    """

    info_path = InfoPath(
                    path='temp_data',
                    dir_name="a02_gaussian_mixture1",
                    sub_dir_name=InfoPath.DO_NOT_CREATE
                )

    return run(info_path=info_path, func=run_model, data=get_data1())


def get_fit2():
    """
    Returns fit file for unit tests, uses data with larger uncertainties.
    """

    info_path = InfoPath(
                    path='temp_data',
                    dir_name="a02_gaussian_mixture2",
                    sub_dir_name=InfoPath.DO_NOT_CREATE
                )

    return run(info_path=info_path, func=run_model, data=get_data2())
