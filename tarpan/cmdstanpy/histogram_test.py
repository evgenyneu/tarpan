import os
import shutil
from tarpan.testutils.a01_eight_schools.eight_schools import get_fit
from tarpan.cmdstanpy.histogram import save_histogram


def test_save_traceplot():
    fit = get_fit()
    outdir = "tarpan/cmdstanpy/model_info/histogram_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_histogram(fit)

    assert os.path.isfile(os.path.join(outdir, "histogram_01.pdf"))
    assert os.path.isfile(os.path.join(outdir, "histogram_01.pdf"))
    assert os.path.isfile(os.path.join(outdir, "histogram_01.pdf"))
