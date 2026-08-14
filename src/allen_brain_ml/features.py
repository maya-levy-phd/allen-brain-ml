import numpy as np


def firing_rate(
    spike_count: int | np.ndarray,
    duration_seconds: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate firing rate in spikes per second."""
    if np.any(duration_seconds <= 0):
        raise ValueError("Duration must be greater than zero.")
    return spike_count / duration_seconds


