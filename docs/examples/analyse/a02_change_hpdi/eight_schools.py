"""
Example of using `analyze` function with custom
HPDIs (highest posterior density intervals).

Usage
-----

    python eight_schools.py

"""

from cmdstanpy import CmdStanModel
from tarpan.cmdstanpy import analyse
from tarpan.shared import SummaryParams


def run_model():
    model = CmdStanModel(stan_file="eight_schools.stan")

    data = {
        "J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]
    }

    fit = model.sample(data=data, chains=4, cores=4,
                       sampling_iters=1000, warmup_iters=1000)

    # Change the default HPDIs (highest posterior density intervals)
    summary_params = SummaryParams()
    summary_params.hpdis = [0.05, 0.3]
    analyse(fit, summary_params=summary_params)
    pass


if __name__ == '__main__':
    run_model()
    print('We are done')
