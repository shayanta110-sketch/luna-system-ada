"""File tools module for writing content to files with directory creation and security validation."""

import os
import re
from pathlib import Path
from typing import Optional, Union, List


def write_file(
    file_path: Union[str, Path],
    content: str,
    mode: str = "w",
    create_dirs: bool = True,
    validate_path: bool = True,
    allowed_root: Optional[Union[str, Path]] = None,
) -> int:
    """
    Write content to a file with optional directory creation and security validation.

    Args:
        file_path: Path to the target file.
        content: String content to write.
        mode: File opening mode ('w' for write, 'a' for append, etc.). Default 'w'.
        create_dirs: If True, create missing parent directories. Default True.
        validate_path: If True, perform basic security validation. Default True.
        allowed_root: If provided, restrict file writes to within this directory.

    Returns:
        Number of bytes written.

    Raises:
        ValueError: If path validation fails.
        OSError: If unable to write due to permissions or other OS errors.
    """
    path = Path(file_path).expanduser().resolve()

    # Security validation
    if validate_path:
        _validate_path_security(path, allowed_root)

    # Create parent directories if needed
    if create_dirs:
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    # Write content to file
    encoding = "utf-8"
    with open(path, mode, encoding=encoding) as f:
        bytes_written = f.write(content)

    return bytes_written


def _validate_path_security(path: Path, allowed_root: Optional[Union[str, Path]] = None) -> None:
    """
    Perform security validation on a file path.

    Args:
        path: Path to validate.
        allowed_root: Optional root directory to restrict paths to.

    Raises:
        ValueError: If path contains suspicious patterns or is outside allowed root.
    """
    # Check for path traversal attempts
    path_str = str(path)
    suspicious_patterns = [r"\.\./", r"\.\.\\", r"~/", r"~\\"]
    for pattern in suspicious_patterns:
        if re.search(pattern, path_str):
            raise ValueError(
                f"Security validation failed: Path contains suspicious pattern '{pattern}': {path_str}"
            )

    # Ensure path is absolute
    if not path.is_absolute():
        raise ValueError(f"Security validation failed: Path must be absolute: {path_str}")

    # Check against allowed root if specified
    if allowed_root is not None:
        root_path = Path(allowed_root).expanduser().resolve()
        if not root_path.is_absolute():
            raise ValueError(f"Allowed root must be absolute: {root_path}")
        try:
            path.relative_to(root_path)
        except ValueError:
            raise ValueError(
                f"Security validation failed: Path {path} is not within allowed root {root_path}"
            )

    # Basic sanity checks (no null bytes, no excessive length)
    if "\x00" in path_str:
        raise ValueError("Security validation failed: Path contains null byte")
    if len(path_str) > 4096:
        raise ValueError("Security validation failed: Path exceeds maximum length")


def read_file(file_path: Union[str, Path], validate_path: bool = True) -> str:
    """
    Read content from a file.

    Args:
        file_path: Path to the target file.
        validate_path: If True, perform basic security validation.

    Returns:
        String content of the file.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If path validation fails.
    """
    path = Path(file_path).expanduser().resolve()

    if validate_path:
        _validate_path_security(path)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def file_exists(file_path: Union[str, Path], validate_path: bool = True) -> bool:
    """
    Check if a file exists.

    Args:
        file_path: Path to the target file.
        validate_path: If True, perform basic security validation.

    Returns:
        True if the file exists and is a regular file, False otherwise.
    """
    path = Path(file_path).expanduser().resolve()

    if validate_path:
        _validate_path_security(path)

    return path.is_file()


def list_directory(
    dir_path: Union[str, Path],
    validate_path: bool = True,
    allowed_root: Optional[Union[str, Path]] = None,
    pattern: Optional[str] = None,
) -> List[Path]:
    """
    List contents of a directory with security validation.

    Args:
        dir_path: Path to the target directory.
        validate_path: If True, perform basic security validation. Default True.
        allowed_root: If provided, restrict directory access to within this root.
        pattern: Optional glob pattern to filter results (e.g., "*.txt").

    Returns:
        List of Path objects for files and directories within the target directory.

    Raises:
        ValueError: If path validation fails or directory doesn't exist.
        NotADirectoryError: If path exists but is not a directory.
    """
    path = Path(dir_path).expanduser().resolve()

    if validate_path:
        _validate_path_security(path, allowed_root)

    if not path.exists():
        raise ValueError(f"Directory does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    if pattern:
        return list(path.glob(pattern))
    else:
        return list(path.iterdir())


def delete_file(
    file_path: Union[str, Path],
    validate_path: bool = True,
    allowed_root: Optional[Union[str, Path]] = None,
    missing_ok: bool = False,
) -> None:
    """
    Delete a file with security validation.

    Args:
        file_path: Path to the target file to delete.
        validate_path: If True, perform basic security validation. Default True.
        allowed_root: If provided, restrict file deletion to within this root.
        missing_ok: If True, suppress FileNotFoundError when file doesn't exist.

    Raises:
        ValueError: If path validation fails.
        FileNotFoundError: If file doesn't exist and missing_ok is False.
        IsADirectoryError: If path points to a directory (use rmdir for directories).
        PermissionError: If lacking permissions to delete the file.
    """
    path = Path(file_path).expanduser().resolve()

    if validate_path:
        _validate_path_security(path, allowed_root)

    if not path.exists():
        if missing_ok:
            return
        else:
            raise FileNotFoundError(f"File does not exist: {path}")

    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")

    path.unlink()
