from data.a01_eight_schools.eight_schools import get_fit
from tarpan.cmdstanpy_utils import analyse
import shutil
import os


def test_analyse():
    fit = get_fit()

    outdir = "tarpan/model_info"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    analyse(fit)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/diagnostic.txt")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/posterior_01.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/posterior_02.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/posterior_03.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/summary.csv")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/summary.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/summary.txt")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/traceplot_01.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/traceplot_02.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/traceplot_03.pdf")
    assert os.path.isfile(file_path)
