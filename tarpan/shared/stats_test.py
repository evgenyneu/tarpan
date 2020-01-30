import pytest
from pytest import approx
from tarpan.shared.stats import kde


def test_kde_with_uncerts():
    result = kde([-10.1, 9.8], [-10, 10], [1.4, 1.8])

    assert result.shape[0] == 2
    assert result[0] == approx(0.14211638124953146, rel=1e-15)
    assert result[1] == approx(0.11013534965419111, rel=1e-15)


def test_kde_with_uncerts_unequal_data():
    with pytest.raises(ValueError):
        kde([-10.1, 9.8], [-10, 10], [1.4, 1.8, 4.3])


def test_kde_with_uncerts_empty_arrays():
    result = kde([-10.1, 9.8], [], [])

    assert result.shape[0] == 0
