"""Discovery of indexable files inside a source directory.

The rules implemented here are the single source of truth for what gets
indexed: hidden folders, well known build/cache folders, secrets and
environment files, symlinked folders and oversized files are all skipped.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

#: Extensions that can be parsed, mapped to a friendly file type name.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".html": "html",
    ".htm": "html",
    ".docx": "word",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "text",
    ".csv": "csv",
    ".py": "code",
    ".js": "code",
    ".jsx": "code",
    ".ts": "code",
    ".tsx": "code",
    ".java": "code",
    ".c": "code",
    ".h": "code",
    ".cpp": "code",
    ".hpp": "code",
    ".cs": "code",
    ".go": "code",
    ".rb": "code",
    ".rs": "code",
    ".php": "code",
    ".swift": "code",
    ".kt": "code",
    ".scala": "code",
    ".sh": "code",
    ".bash": "code",
    ".sql": "code",
    ".json": "code",
    ".yaml": "code",
    ".yml": "code",
    ".toml": "code",
    ".ini": "code",
    ".cfg": "code",
    ".xml": "code",
    ".tex": "text",
    ".log": "text",
    ".ipynb": "code",
    ".jsonl": "code",
}

#: Files that are always treated as secrets.
SENSITIVE_FILE_NAMES = {
    name.lower() for name in getattr(settings, "SENSITIVE_FILE_NAMES", [])
}
SENSITIVE_FILE_EXTENSIONS = {
    ext.lower() for ext in getattr(settings, "SENSITIVE_FILE_EXTENSIONS", [])
}


def ignored_directories() -> set[str]:
    """Return the configured directory names that must never be traversed."""
    return {name.lower() for name in getattr(settings, "IGNORED_DIRECTORIES", [])}


def is_hidden(name: str) -> bool:
    """Return True for dot-files and dot-directories."""
    return name.startswith(".") and name not in {".", ".."}


def is_sensitive_file(name: str) -> bool:
    """Return True when a file name looks like a secret or credential store."""
    lowered = name.lower()
    if lowered in SENSITIVE_FILE_NAMES:
        return True
    if lowered.startswith(".env"):
        return True
    if any(lowered.endswith(ext) for ext in SENSITIVE_FILE_EXTENSIONS):
        return True
    return False


def is_ignored(path: Path) -> bool:
    """Central exclusion rule for a single path.

    Returns True when the path must not be indexed.
    """
    name = path.name
    if is_hidden(name):
        return True
    if is_sensitive_file(name):
        return True
    if name.lower() in ignored_directories():
        return True
    if path.is_file() and path.suffix.lower() in SENSITIVE_FILE_EXTENSIONS:
        return True
    return False


def is_supported_file(path: Path) -> bool:
    """Return True when the file extension is one the parsers understand."""
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return True
    # A few formats are recognised by a compound extension.
    return ".".join(part.lower() for part in path.name.split(".")[-2:]) in SUPPORTED_EXTENSIONS


def file_type_for(path: Path) -> str:
    """Return the friendly file type used in metadata and the UI."""
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[suffix]
    compound = ".".join(part.lower() for part in path.name.split(".")[-2:])
    return SUPPORTED_EXTENSIONS.get(compound, "text")


def discover_supported_files(root: Path) -> list[str]:
    """Recursively find indexable files under ``root``.

    Returns a sorted list of POSIX style relative paths.  Symlinked
    directories are not followed, which prevents escaping the source root
    and avoids duplicate work.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    max_files = int(getattr(settings, "MAX_SOURCE_FILES", 5000))
    max_size = int(getattr(settings, "MAX_FILE_SIZE_BYTES", 25 * 1024 * 1024))
    ignored = ignored_directories()

    discovered: list[str] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)

        # Prune directories in place so os.walk does not descend into them.
        kept_dirs = []
        for dirname in dirnames:
            if is_hidden(dirname) or dirname.lower() in ignored:
                continue
            candidate = current / dirname
            if candidate.is_symlink():
                # Never traverse symlinked directories.
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            if len(discovered) >= max_files:
                truncated = True
                break
            if is_hidden(filename) or is_sensitive_file(filename):
                continue
            candidate = current / filename
            if candidate.is_symlink():
                continue
            if not candidate.is_file():
                continue
            if not is_supported_file(candidate):
                continue
            try:
                size = candidate.stat().st_size
            except OSError:
                continue
            if size > max_size:
                logger.info("Skipping %s: larger than %s bytes", candidate, max_size)
                continue
            try:
                relative = candidate.relative_to(root)
            except ValueError:  # pragma: no cover - defensive
                continue
            discovered.append(relative.as_posix())

        if truncated:
            break

    if truncated:
        logger.warning(
            "Stopped discovery at %s files (MAX_SOURCE_FILES).", max_files
        )
    return sorted(discovered)
