import os
import shutil
from tarpan.testutils.a01_eight_schools.eight_schools import get_fit
from tarpan.cmdstanpy.traceplot import save_traceplot


def test_save_traceplot():
    fit = get_fit()
    outdir = "tarpan/cmdstanpy/model_info/traceplot_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_traceplot(fit)

    assert os.path.isfile(os.path.join(outdir, "traceplot_01.pdf"))
    assert os.path.isfile(os.path.join(outdir, "traceplot_02.pdf"))
    assert os.path.isfile(os.path.join(outdir, "traceplot_03.pdf"))
