import os
import shutil
from tarpan.cmdstanpy.summary import save_summary
from tarpan.testutils.a01_eight_schools.eight_schools import get_fit
from tarpan.shared.summary import SummaryParams


def test_save_summary():
    fit = get_fit()

    outdir = "tarpan/cmdstanpy/model_info/summary_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_summary(fit, param_names=["mu", "tau", 'eta.1'])

    assert os.path.isfile(os.path.join(outdir, "summary.txt"))
    assert os.path.isfile(os.path.join(outdir, "summary.csv"))


def test_save_summary_specify_hpdi():
    fit = get_fit()

    outdir = "tarpan/cmdstanpy/model_info/summary_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_summary(fit, param_names=["mu", "tau", 'eta.1'],
                 summary_params=SummaryParams(hpdis=[0.05, 0.99]))

    assert os.path.isfile(os.path.join(outdir, "summary.txt"))
    assert os.path.isfile(os.path.join(outdir, "summary.csv"))
