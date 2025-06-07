import os
from pathlib import Path
from typing import List
from fastapi import UploadFile

def list_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    List all files in a directory, optionally filtering by file extensions.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if extensions:
                if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    files.append(os.path.join(root, filename))
            else:
                files.append(os.path.join(root, filename))
    return files

def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read the contents of a file.
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()

def write_file(file_path: str, content: str, encoding: str = "utf-8"):
    """
    Write content to a file.
    """
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)

def ensure_dir(directory: str):
    """
    Ensure that a directory exists.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """
    Save an uploaded file to the specified destination.
    Returns the saved file path.
    """
    ensure_dir(os.path.dirname(destination))
    with open(destination, "wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)
    return destination

def cleanup_temp_file(file_path: str):
    """
    Remove a temporary file if it exists.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass