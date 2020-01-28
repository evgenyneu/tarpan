import shutil
import os
from tarpan.testutils.a01_eight_schools.eight_schools import get_fit
from tarpan.cmdstanpy.analyse import save_analysis


def test_save_analysis():
    fit = get_fit()

    outdir = "tarpan/cmdstanpy/model_info/analyse_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_analysis(fit)

    # Diagnostic
    assert os.path.isfile(os.path.join(outdir, "diagnostic.txt"))

    # Check summary files
    # ---------------

    assert os.path.isfile(os.path.join(outdir, "summary.txt"))
    assert os.path.isfile(os.path.join(outdir, "summary.csv"))

    # Tree plot
    assert os.path.isfile(os.path.join(outdir, "summary.pdf"))

    # Trace plots
    # ----------

    assert os.path.isfile(os.path.join(outdir, "traceplot_01.pdf"))
    assert os.path.isfile(os.path.join(outdir, "traceplot_02.pdf"))
    assert os.path.isfile(os.path.join(outdir, "traceplot_03.pdf"))

    # Histograms of posterior distributions
    # ----------

    assert os.path.isfile(os.path.join(outdir, "histogram_01.pdf"))
    assert os.path.isfile(os.path.join(outdir, "histogram_02.pdf"))
    assert os.path.isfile(os.path.join(outdir, "histogram_03.pdf"))

    # Pair plot
    assert os.path.isfile(os.path.join(outdir, "pair_plot.pdf"))
