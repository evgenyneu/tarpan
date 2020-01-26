# Example of setting custom widths of HPD intervals

from cmdstanpy import CmdStanModel
from tarpan.cmdstanpy.summary import save_summary
from tarpan.shared.summary import SummaryParams


def run_model():
    model = CmdStanModel(stan_file="eight_schools.stan")

    data = {
        "J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]
    }

    fit = model.sample(data=data, chains=4, cores=4, seed=1,
                       sampling_iters=1000, warmup_iters=1000)

    # Specify custom HPDI values
    summary_params = SummaryParams()
    summary_params.hpdis = [0.05, 0.99]

    save_summary(fit, param_names=['mu', 'tau', 'eta.1'],
                 summary_params=summary_params)


if __name__ == '__main__':
    run_model()
    print('We are done')
