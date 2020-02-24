from tarpan.testutils.a03_cars.cars import get_fit
from tarpan.cmdstanpy.waic import waic


def test_hello():
    fit = get_fit()
    point_lpd = fit.get_drawset(params=["a"])
    print(point_lpd)
    return

    result = waic(fit)
    # print(result)
