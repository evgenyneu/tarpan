from pytest import approx
from tarpan.testutils.a03_cars.cars import get_fit
from tarpan.cmdstanpy.waic import waic


def test_hello():
    fit = get_fit()
    result = waic(fit)

    assert result.waic == approx(421.5135196466395, rel=1e-15)
    assert result.lppd == approx(-206.5875738029627, rel=1e-15)

    assert len(result.lppd_pointwise) == 50
    assert result.lppd_pointwise[0] == approx(-3.6203241203579615, rel=1e-15)
    assert result.lppd_pointwise[49] == approx(-3.641419673133626, rel=1e-15)

    assert result.penalty == approx(4.169186020357044, rel=1e-15)
