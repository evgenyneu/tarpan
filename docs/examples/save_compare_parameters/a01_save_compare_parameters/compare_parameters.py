"""
Create a tree plot that compares parameters from multiple models.
"""

from cmdstanpy import CmdStanModel
from tarpan.cmdstanpy.compare_parameters import save_compare_parameters
from tarpan.shared.compare_parameters import CompareParametersType


def run_model():
    model = CmdStanModel(stan_file="eight_schools.stan")

    data = {
        "J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]
    }

    fit1 = model.sample(data=data, chains=4, cores=4, seed=1,
                        sampling_iters=1000, warmup_iters=1000)

    # Increase the uncertainties
    data["sigma"] = [i * 2 for i in data["sigma"]]

    fit2 = model.sample(data=data, chains=4, cores=4, seed=1,
                        sampling_iters=1000, warmup_iters=1000)

    save_compare_parameters([fit1, fit2],
                            labels=['Original', 'Larger uncertainties'],
                            type=CompareParametersType.TEXT,  # or GITLAB_LATEX
                            param_names=['mu', 'tau'])


if __name__ == '__main__':
    run_model()
    print('We are done')
