from tarpan.shared.compare import model_weights


def test_weights():
    weights = model_weights(deviances=[[1, 2, 3], [2, 3, 4], [7, 8, 9]])

    actual_weights = [
        round(weight, 5)
        for weight in weights
    ]

    assert actual_weights == [0.81749, 0.18241, 0.0001]
