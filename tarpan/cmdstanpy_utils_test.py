from data.a01_eight_schools.eight_schools import get_fit
from tarpan.cmdstanpy_utils import analyse


def test_hello():
    fit = get_fit()
    analyse(fit)

# hithere()
