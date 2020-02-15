from cmdstanpy import CmdStanModel
import scipy.stats as stats
from tarpan.plot.posterior import save_posterior_scatter_and_kde


def run_stan(observed_values, uncertainties):
    data = {
        "y": observed_values,
        "uncertainties": uncertainties,
        "N": len(observed_values)
    }

    model = CmdStanModel(stan_file="stan_model/gaussian_mixture.stan")

    return model.sample(
        data=data, seed=333,
        adapt_delta=0.90, max_treedepth=5,
        sampling_iters=500, warmup_iters=500,
        chains=4, cores=4)


def model_pdf(x, row):
    mu1 = row['mu.1']
    mu2 = row['mu.2']
    sigma = row['sigma']
    r = row['r']

    return (1 - r) * stats.norm.pdf(x, mu1, sigma) + \
        r * stats.norm.pdf(x, mu2, sigma)


def run_model():
    values1 = [-0.99, -1.37, -1.38, -1.51, -1.29, -1.34, -1.50, -0.93, -0.83,
               -1.46, -1.07, -1.28, -0.73]

    uncertainties1 = [0.12, 0.05, 0.11, 0.18, 0.03, 0.19, 0.18, 0.12, 0.19,
                      0.09, 0.11, 0.16, 0.08]

    values2 = [
            -1.22, -1.15, -0.97, -0.68, -0.37, -0.48, -0.73, -0.61, -1.32,
            -0.62, -1.13, -0.65, -0.90, -1.29, -1.19, -0.54, -0.64, -0.45,
            -1.21, -0.75, -0.66, -0.71, -0.61, -0.59, -1.07, -0.65, -0.59]

    uncertainties2 = [
         0.13, 0.14, 0.17, 0.07, 0.11, 0.12, 0.23, 0.05, 0.04,
         0.30, 0.11, 0.13, 0.16, 0.03, 0.18, 0.20, 0.16, 0.16,
         0.11, 0.09, 0.20, 0.10, 0.08, 0.04, 0.04, 0.23, 0.19]

    fit1 = run_stan(values1, uncertainties1)
    fit2 = run_stan(values2, uncertainties2)

    fig, axes = save_posterior_scatter_and_kde(
        fits=[fit1, fit2],
        pdf=model_pdf,
        values=[values1, values2],
        uncertainties=[uncertainties1, uncertainties2],
        title="Sodium abundances in RGB stars of NGC 288",
        xlabel="Sodium abundance [Na/H]",
        ylabel=["Star number", "Probability density"],
        legend_labels=["AGB", "RGB"])


if __name__ == '__main__':
    run_model()
    print('We are done')
