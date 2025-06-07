import os
from pathlib import Path

def is_valid_file(file_path: str, allowed_extensions=None) -> bool:
    """
    Check if the file exists and has an allowed extension.
    """
    if not os.path.isfile(file_path):
        return False
    if allowed_extensions:
        ext = Path(file_path).suffix.lower()
        return ext in [e.lower() for e in allowed_extensions]
    return True

def is_non_empty_string(value: str) -> bool:
    """
    Check if the value is a non-empty string.
    """
    return isinstance(value, str) and bool(value.strip())

def is_valid_directory(directory: str) -> bool:
    """
    Check if the directory exists.
    """
    return os.path.isdir(directory)

def validate_file_type(file_path: str, allowed_extensions=None) -> bool:
    """
    Validate that the file exists and has an allowed extension.
    """
    return is_valid_file(file_path, allowed_extensions)