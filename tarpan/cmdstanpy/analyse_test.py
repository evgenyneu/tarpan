from tarpan.testutils.a01_eight_schools.eight_schools import get_fit
from tarpan.cmdstanpy.analyse import save_analysis
import shutil
import os


def test_save_analysis():
    fit = get_fit()

    outdir = "tarpan/cmdstanpy/model_info/analyse_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_analysis(fit)

    file_path = os.path.join(outdir, "diagnostic.txt")
    assert os.path.isfile(file_path)

    # Check summary files
    # ---------------

    file_path = os.path.join(outdir, "summary.csv")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "cmdstanpy_utils_test/summary.txt")
    assert os.path.isfile(file_path)

    # Tree plot
    # ---------

    file_path = os.path.join(outdir, "summary.pdf")
    assert os.path.isfile(file_path)

    # file_path = os.path.join(outdir, "cmdstanpy_utils_test/posterior_01.pdf")
    # assert os.path.isfile(file_path)
    #
    # file_path = os.path.join(outdir, "cmdstanpy_utils_test/posterior_02.pdf")
    # assert os.path.isfile(file_path)
    #
    # file_path = os.path.join(outdir, "cmdstanpy_utils_test/posterior_03.pdf")
    # assert os.path.isfile(file_path)
    #
    #
    #
    # file_path = os.path.join(outdir, "cmdstanpy_utils_test/traceplot_01.pdf")
    # assert os.path.isfile(file_path)
    #
    # file_path = os.path.join(outdir, "cmdstanpy_utils_test/traceplot_02.pdf")
    # assert os.path.isfile(file_path)
    #
    # file_path = os.path.join(outdir, "cmdstanpy_utils_test/traceplot_03.pdf")
    # assert os.path.isfile(file_path)
