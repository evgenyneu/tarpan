import os
import shutil
import pytest
from pytest import approx

from tarpan.plot.kde import (
    gaussian_kde, save_scatter_and_kde)


def test_kde_with_uncerts():
    result = gaussian_kde([-10.1, 9.8], [-10, 10], [1.4, 1.8])

    assert result.shape[0] == 2
    assert result[0] == approx(0.14211638124953146, rel=1e-15)
    assert result[1] == approx(0.11013534965419111, rel=1e-15)


def test_kde_with_uncerts_unequal_data():
    with pytest.raises(ValueError):
        gaussian_kde([-10.1, 9.8], [-10, 10], [1.4, 1.8, 4.3])


def test_kde_with_uncerts_empty_arrays():
    result = gaussian_kde([-10.1, 9.8], [], [])

    assert result.shape[0] == 0


# save_scatter_and_kde
# ----------------------

def test_save_scatter_and_kde_single():
    outdir = "tarpan/plot/model_info/kde_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    values1 = [
            -1.22, -1.15, -0.97, -0.68, -0.37, -0.48, -0.73, -0.61, -1.32,
            -0.62, -1.13, -0.65, -0.90, -1.29, -1.19, -0.54, -0.64, -0.45,
            -1.21, -0.75, -0.66, -0.71, -0.61, -0.59, -1.07, -0.65, -0.59]

    uncertainties1 = [
         0.13, 0.14, 0.17, 0.07, 0.11, 0.12, 0.23, 0.05, 0.04,
         0.30, 0.11, 0.13, 0.16, 0.03, 0.18, 0.20, 0.16, 0.16,
         0.11, 0.09, 0.20, 0.10, 0.08, 0.04, 0.04, 0.23, 0.19]

    save_scatter_and_kde(values=[values1], uncertainties=[uncertainties1],
                         title="Sodium abundances in RGB stars of NGC 288",
                         xlabel="Sodium abundance [Na/H]",
                         ylabel=["Star number", "Probability density"])

    assert os.path.isfile(os.path.join(outdir, "scatter_kde.pdf"))


def test_save_scatter_and_kde_multiple():
    outdir = "tarpan/plot/model_info/kde_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

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

    save_scatter_and_kde(values=[values1, values2],
                         uncertainties=[uncertainties1, uncertainties2],
                         title="Sodium abundances in RGB stars of NGC 288",
                         xlabel="Sodium abundance [Na/H]",
                         ylabel=["Star number", "Probability density"],
                         legend_labels=["AGB", "RGB"])

    assert os.path.isfile(os.path.join(outdir, "scatter_kde.pdf"))
