import pytest
import numpy as np
from allen_brain_ml.data import clean_session_name, normalize_session_names, validate_duration


def test_clean_session_name():
    assert clean_session_name('   session_001   ') == 'session_001'


def test_normalize_session_names():
    names = ["  session_001 ", "session_002  ", "  session_003"]
    assert normalize_session_names(names) == [
        "session_001",
        "session_002",
        "session_003",
    ]

def test_validate_duration():
    validate_duration(2.0)

def test_validate_duration_zero():
    with pytest.raises(ValueError):
        validate_duration(0.0)

def test_validate_duration_negative():
    with pytest.raises(ValueError):
        validate_duration(-1.0)

def test_validate_duration_array():
    durations = np.array([1.0, 2.5, 10.0])
    validate_duration(durations)

def test_validate_duration_array_with_invalid_value():
    durations = np.array([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        validate_duration(durations)

