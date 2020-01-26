from tarpan.cmdstanpy.pair_plot import save_pair_plot
from tarpan.testutils.a01_eight_schools.eight_schools import get_fit


def test_save_pair_plot():
    fit = get_fit()

    save_pair_plot(fit, param_names=["mu", "tau", 'eta.1'])
