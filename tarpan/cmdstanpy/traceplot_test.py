from tarpan.testutils.a01_eight_schools.eight_schools import get_fit
from tarpan.cmdstanpy.traceplot import save_traceplot
from tarpan.shared.info_path import InfoPath


def test_save_traceplot():
    fit = get_fit()

    path = InfoPath()
    save_traceplot(fit, info_path=path)

    print(f'path.codefile_path={path.codefile_path}')

    assert 2 == 2
