import numpy as np
from allen_brain_ml.features import firing_rate


def test_firing_rate():
    assert firing_rate(20, 2.0) == 10.0

def test_firing_rate_array():
    spike_counts = np.array([20, 15, 30])
    durations = np.array([2.0, 1.5, 3.0])

    expected = np.array([10.0, 10.0, 10.0])

    np.testing.assert_array_equal(
        firing_rate(spike_counts, durations),
        expected,
    )