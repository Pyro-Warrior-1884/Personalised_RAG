"""Validation and preparation of knowledge base sources.

Two source types are supported:

* ``local_directory`` - a folder on the machine that runs Django.  The path
  must live inside one of the approved roots configured with
  ``ALLOWED_SOURCE_ROOTS``.
* ``github`` - a public GitHub repository that is cloned into a dedicated,
  application managed workspace.  Repository code is never executed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from django.conf import settings

logger = logging.getLogger(__name__)

GIT_TIMEOUT_SECONDS = 300
GITHUB_ALLOWED_HOSTS = {"github.com", "www.github.com"}
GITHUB_REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class SourceValidationError(Exception):
    """Raised when a source cannot be used as a knowledge base."""


@dataclass
class PreparedSource:
    """The outcome of :func:`prepare_source`."""

    root: Path
    source_type: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Local directories
# ---------------------------------------------------------------------------
def allowed_roots() -> list[Path]:
    """Return the list of approved local roots."""
    roots: list[Path] = []
    for root in getattr(settings, "ALLOWED_SOURCE_ROOTS", []):
        try:
            resolved = Path(root).expanduser().resolve()
        except (OSError, RuntimeError):  # pragma: no cover - defensive
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _is_within(candidate: Path, root: Path) -> bool:
    """Return True when ``candidate`` is ``root`` or lives inside ``root``."""
    if candidate == root:
        return True
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def validate_local_directory(path: str | os.PathLike[str]) -> Path:
    """Resolve and validate a local directory path.

    The path is expanded, resolved (following symlinks) and then checked
    against the configured approved roots.  Anything that escapes those roots
    - including via ``..`` or a symlink - is rejected.
    """
    if path is None:
        raise SourceValidationError("Please enter a directory path.")

    raw = str(path).strip()
    if not raw:
        raise SourceValidationError("Please enter a directory path.")

    if "\x00" in raw:
        raise SourceValidationError("The path contains invalid characters.")

    try:
        candidate = Path(raw).expanduser()
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise SourceValidationError(f"The path could not be resolved: {exc}") from exc

    if not resolved.exists():
        raise SourceValidationError(f"The path does not exist: {resolved}")
    if not resolved.is_dir():
        raise SourceValidationError(f"The path is not a directory: {resolved}")

    roots = allowed_roots()
    if not roots:
        raise SourceValidationError(
            "No approved source roots are configured. Set ALLOWED_SOURCE_ROOTS in your "
            ".env.local file to a directory that may be indexed."
        )
    if not any(_is_within(resolved, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise SourceValidationError(
            f"That directory is outside the approved roots. Allowed roots: {allowed}"
        )
    return resolved


# ---------------------------------------------------------------------------
# GitHub repositories
# ---------------------------------------------------------------------------
def validate_github_url(url: str) -> str:
    """Validate a GitHub repository URL and return its normalised form."""
    if url is None:
        raise SourceValidationError("Please enter a repository URL.")

    raw = str(url).strip()
    if not raw:
        raise SourceValidationError("Please enter a repository URL.")
    if "\x00" in raw:
        raise SourceValidationError("The URL contains invalid characters.")

    parsed = urlparse(raw)
    if parsed.scheme.lower() != "https":
        raise SourceValidationError("Only https:// repository URLs are allowed.")
    if not parsed.hostname or parsed.hostname.lower() not in GITHUB_ALLOWED_HOSTS:
        raise SourceValidationError("Only public github.com repository URLs are allowed.")
    if parsed.username or parsed.password:
        raise SourceValidationError("Credentials in repository URLs are not allowed.")
    if parsed.port not in (None, 443):
        raise SourceValidationError("Only the standard https port is allowed.")

    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    if not GITHUB_REPO_PATTERN.match(path):
        raise SourceValidationError(
            "The URL must look like https://github.com/owner/repository"
        )
    if path.count("/") != 1:
        raise SourceValidationError(
            "Only repository roots are supported (no subdirectories, branches or file paths)."
        )

    owner, repo = path.split("/", 1)
    return f"https://github.com/{owner}/{repo}.git"


def repository_workspace_path(normalised_url: str) -> Path:
    """Return the managed checkout directory for a repository URL."""
    parsed = urlparse(normalised_url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    owner, repo = path.split("/", 1)
    safe_owner = re.sub(r"[^A-Za-z0-9._-]", "_", owner)
    safe_repo = re.sub(r"[^A-Za-z0-9._-]", "_", repo)
    workspace = Path(settings.REPOSITORY_WORKSPACE)
    return workspace / f"{safe_owner}__{safe_repo}"


def _run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a git command with a hardened environment and a timeout."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
        "GIT_CONFIG_NOSYSTEM": "1",
        "LC_ALL": "C",
    }
    if cwd is not None:
        env["GIT_DIR"] = str(cwd / ".git")
        env["GIT_WORK_TREE"] = str(cwd)
    try:
        return subprocess.run(  # noqa: S603 - argument array, no shell
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SourceValidationError(
            "Git is not installed or not on PATH, so repositories cannot be cloned."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceValidationError(
            "The git command timed out while preparing the repository."
        ) from exc


def _git_head_commit(repo_path: Path) -> str:
    result = _run_git(["rev-parse", "HEAD"], cwd=repo_path)
    if result.returncode == 0:
        return result.stdout.strip()
    return ""


def clone_or_update_repository(url: str) -> tuple[Path, str]:
    """Clone or update a repository inside the managed workspace.

    Returns the checkout path and the current commit hash.  No repository
    code is executed: only git plumbing commands are used.
    """
    normalised = validate_github_url(url)
    target = repository_workspace_path(normalised)
    target.parent.mkdir(parents=True, exist_ok=True)

    if (target / ".git").exists():
        logger.info("Updating existing checkout at %s", target)
        fetch = _run_git(["fetch", "--depth", "1", "origin"], cwd=target)
        if fetch.returncode != 0:
            logger.warning("git fetch failed: %s", (fetch.stderr or "").strip()[:500])
        else:
            reset = _run_git(["reset", "--hard", "FETCH_HEAD"], cwd=target)
            if reset.returncode != 0:
                logger.warning("git reset failed: %s", (reset.stderr or "").strip()[:500])
    else:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        result = _run_git(
            ["clone", "--depth", "1", "--single-branch", "--no-tags", normalised, str(target)]
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[:300]
            raise SourceValidationError(f"The repository could not be cloned. {detail}")

    return target, _git_head_commit(target)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def prepare_source(source_type: str, source_location: str) -> PreparedSource:
    """Validate a source and return the local directory that should be indexed."""
    if source_type == "local_directory":
        resolved = validate_local_directory(source_location)
        return PreparedSource(
            root=resolved,
            source_type=source_type,
            metadata={
                "resolved_path": str(resolved),
                "requested_path": str(source_location),
            },
        )

    if source_type == "github":
        target, commit = clone_or_update_repository(source_location)
        return PreparedSource(
            root=target,
            source_type=source_type,
            metadata={
                "resolved_path": str(target),
                "repository_url": validate_github_url(source_location),
                "commit_hash": commit,
            },
        )

    raise SourceValidationError(f"Unsupported source type: {source_type}")
