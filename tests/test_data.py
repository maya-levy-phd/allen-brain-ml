from allen_brain_ml.data import clean_session_name


def test_clean_session_name():
    assert clean_session_name('   session_001   ') == 'session_001'

