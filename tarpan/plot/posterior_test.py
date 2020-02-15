import os
import scipy.stats as stats
import shutil
from tarpan.plot.posterior import save_posterior_scatter_and_kde

from tarpan.testutils.a02_gaussian_mixture.gaussian_mixture import (
    get_fit1, get_fit2, get_data1, get_data2)


def model_pdf(x, row):
    mu1 = row['mu.1']
    mu2 = row['mu.2']
    sigma = row['sigma']
    r = row['r']

    return (1 - r) * stats.norm.pdf(x, mu1, sigma) + \
        r * stats.norm.pdf(x, mu2, sigma)


def test_save_posterior_scatter_and_kde():
    fit1 = get_fit1()
    fit2 = get_fit2()
    fits = [fit1, fit2]

    outdir = "tarpan/plot/model_info/posterior_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    data1 = get_data1()
    data2 = get_data2()

    fig, axes = save_posterior_scatter_and_kde(
        fits=fits,
        pdf=model_pdf,
        values=[data1["y"], data2["y"]],
        uncertainties=[data1["uncertainties"], data2["uncertainties"]],
        title="Sodium abundances in RGB stars of NGC 288",
        xlabel="Sodium abundance [Na/H]",
        ylabel=["Star number", "Probability density"],
        legend_labels=["AGB", "RGB"])

    assert os.path.isfile(os.path.join(outdir, "posterior_scatter_kde.pdf"))
    assert len(axes) == 2
