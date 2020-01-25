"""
Create a tree plot that compares parameters from multiple models.
"""

from cmdstanpy import CmdStanModel
from tarpan.cmdstanpy.tree_plot import save_tree_plot
from tarpan.shared.tree_plot import TreePlotParams


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

    # Show extra markers in tree plot for comparison (optional)
    extra_values = [
        {
            "mu": 2.2,
            "tau": 1.3,
        }
    ]

    # Supply legend labels (optional)
    tree_params = TreePlotParams()
    tree_params.labels = ["Model 1", "Model 2", "Exact"]

    save_tree_plot(fits=[fit1, fit2], extra_values=extra_values,
                   param_names=['mu', 'tau'],
                   tree_params=tree_params)


if __name__ == '__main__':
    run_model()
    print('We are done')
