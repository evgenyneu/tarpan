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

    file_path = os.path.join(outdir, "summary.txt")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "summary.csv")
    assert os.path.isfile(file_path)

    # Tree plot
    # ---------

    file_path = os.path.join(outdir, "summary.pdf")
    assert os.path.isfile(file_path)

    # Trace plots
    # ----------

    file_path = os.path.join(outdir, "traceplot_01.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "traceplot_02.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "traceplot_03.pdf")
    assert os.path.isfile(file_path)

    # Histograms of posterior distributions
    # ----------

    file_path = os.path.join(outdir, "posterior_01.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "posterior_02.pdf")
    assert os.path.isfile(file_path)

    file_path = os.path.join(outdir, "posterior_03.pdf")
    assert os.path.isfile(file_path)
