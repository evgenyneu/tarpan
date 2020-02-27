from tarpan.shared.stats import hpdi


def test_hpdi():
    values = [1, 1.1, 1.1, 1.2, 1.8, 1.9, 8]
    result = hpdi(values, probability=0.68)

    assert result == (1.1, 1.9)
