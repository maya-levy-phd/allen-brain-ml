def clean_session_name(name: str) -> str:
    """Return a session name with surrounding whitespace removed."""
    return name.strip()


def normalize_session_names(names: list[str]) -> list[str]:
    """Clean whitespace from a collection of session names."""
    return [name.strip() for name in names]

