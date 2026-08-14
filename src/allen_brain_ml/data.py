import numpy as np

def clean_session_name(name: str) -> str:
    """Return a session name with surrounding whitespace removed."""
    return name.strip()


def normalize_session_names(names: list[str]) -> list[str]:
    """Clean whitespace from a collection of session names."""
    return [name.strip() for name in names]

def validate_duration(
    duration_seconds: float | np.ndarray,
) -> None:
    """Validate that all durations are strictly positive."""
    if np.any(duration_seconds <= 0):
        raise ValueError("Duration must be greater than zero.")
