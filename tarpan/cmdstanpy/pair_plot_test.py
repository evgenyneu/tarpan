import os
import shutil
from tarpan.cmdstanpy.pair_plot import save_pair_plot
from tarpan.testutils.a01_eight_schools.eight_schools import get_fit


def test_save_pair_plot():
    fit = get_fit()

    outdir = "tarpan/cmdstanpy/model_info/pair_plot_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_pair_plot(fit, param_names=["mu", "tau", 'eta.1'])

    assert os.path.isfile(os.path.join(outdir, "pair_plot.pdf"))
