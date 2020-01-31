import os
import shutil
import pytest
from pytest import approx
from tarpan.plot.kde import gaussian_kde, save_scatter_and_kde


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

def test_save_scatter_and_kde():
    outdir = "tarpan/plot/model_info/kde_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_scatter_and_kde(values=[1, 1.3, 1.5, 7, 4.9],
                         uncertainties=[0.1, 0.6, 0.35, 0.41, 0.03])

    assert os.path.isfile(os.path.join(outdir, "scatter_kde.pdf"))
