import numpy as np
from allen_brain_ml.data import validate_duration


def firing_rate(
    spike_count: int | np.ndarray,
    duration_seconds: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate firing rate in spikes per second."""
    validate_duration(duration_seconds)
    return spike_count / duration_seconds


