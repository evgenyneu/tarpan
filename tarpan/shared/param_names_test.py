from tarpan.shared.param_names import filter_param_names


def test_filter_param_names():
    result = filter_param_names(['a', 'b', 'c'], ['a', 'b'])

    assert result == ['a', 'b']


def test_filter_param_names__numbered():
    result = filter_param_names(['a.1', 'a.2', 'a', 'b', 'c'], ['a', 'b'])

    assert result == ['a.1', 'a.2', 'a', 'b']


def test_filter_param_names__no_filter():
    result = filter_param_names(['a', 'b', 'c'])

    assert result == ['a', 'b', 'c']


def test_filter_param_names__remove_technical_columns():
    result = filter_param_names(['a', 'stepsize__', 'c'])

    assert result == ['a', 'c']
