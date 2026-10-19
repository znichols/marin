# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Workspace bundle creation for job submission."""

import logging
import os
import re
import subprocess
import tempfile
import tomllib
import zipfile
from collections.abc import Sequence
from pathlib import Path

from iris.cluster.bundle import MAX_BUNDLE_SIZE_BYTES

logger = logging.getLogger(__name__)

# Default exclude pattern applied to all bundles.  Matches against the
# *relative* path (forward-slash separated, no leading slash).
DEFAULT_EXCLUDE = re.compile(
    r"""
      \.mov$                        # video files
    | \.pyc$                        # bytecode
    | \.egg-info(/|$)               # egg metadata
    | (^|/)__pycache__(/|$)         # pycache at any depth
    | (^|/)\.git(/|$)               # .git at any depth
    | (^|/)gha-creds-[^/]+\.json$   # GitHub Actions credentials
    | (^|/)\.mypy_cache(/|$)
    | (^|/)\.pytest_cache(/|$)
    | (^|/)\.ruff_cache(/|$)
    | (^|/)\.venv(/|$)
    | (^|/)node_modules(/|$)
    | (^|/)venv(/|$)
    | (^|/)docs/figures(/|$)
    | (^|/)docs/images(/|$)
    | (^|/)docs/reports(/|$)
    | (^|/)docs/static(/|$)
    | (^|/)tests/snapshots(/|$)
    """,
    re.VERBOSE,
)

# Glob patterns for generated files that are gitignored but required at runtime.
# These are produced by build hooks (e.g. hatch_build.py protobuf generation)
# and must be included in task bundles so that `uv sync` inside containers can
# skip regeneration.
GENERATED_ARTIFACT_GLOBS = [
    "src/iris/rpc/*_pb2.py",
    "src/iris/rpc/*_pb2.pyi",
    "src/iris/rpc/*_connect.py",
    "lib/iris/src/iris/rpc/*_pb2.py",
    "lib/iris/src/iris/rpc/*_pb2.pyi",
    "lib/iris/src/iris/rpc/*_connect.py",
]


def _should_exclude(relative: str, exclude: re.Pattern[str]) -> bool:
    return bool(exclude.search(relative))


def _merge_exclude(extra: re.Pattern[str] | None) -> re.Pattern[str]:
    if extra is None:
        return DEFAULT_EXCLUDE
    # DEFAULT_EXCLUDE is authored in verbose form; scope re.VERBOSE to it with an
    # inline (?x:...) group so the caller's pattern keeps its literal whitespace
    # and '#' instead of having them silently treated as formatting.
    return re.compile(f"(?x:{DEFAULT_EXCLUDE.pattern})|(?:{extra.pattern})")


def _workspace_exclude(workspace: Path) -> re.Pattern[str] | None:
    project_path = workspace / "pyproject.toml"
    if not project_path.is_file():
        return None
    with project_path.open("rb") as project_file:
        project = tomllib.load(project_file)
    patterns = project.get("tool", {}).get("iris", {}).get("bundle", {}).get("exclude", [])
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns)) if patterns else None


def collect_workspace_files(
    workspace: str | Path,
    *,
    exclude: re.Pattern[str] | None = None,
    extra_includes: Sequence[str] = (),
) -> list[Path]:
    """Collect the list of files to include in a workspace bundle.

    Uses git ls-files when available (respecting .gitignore), then adds back gitignored
    files needed at runtime: Iris's generated protobuf artifacts plus any caller-supplied
    ``extra_includes`` globs (relative to ``workspace``, e.g. a package's built frontend
    ``dist``). ``[tool.iris.bundle].exclude`` in the workspace's ``pyproject.toml``
    excludes matching relative paths; explicit ``extra_includes`` override those
    repository exclusions. Falls back to pattern-based exclusion when git is unavailable.

    Args:
        workspace: Root directory to bundle.
        exclude: Extra regex to exclude (merged with DEFAULT_EXCLUDE).
        extra_includes: Glob patterns for gitignored or repository-excluded files the
            caller needs bundled. Default and caller ``exclude`` patterns still apply.

    Returns:
        Sorted list of absolute paths.
    """
    workspace = Path(workspace)
    merged = _merge_exclude(exclude)
    files = _get_git_non_ignored_files(workspace, merged)
    if files is None:
        files = {
            f for f in workspace.rglob("*") if f.is_file() and not _should_exclude(str(f.relative_to(workspace)), merged)
        }
    _include_bundle_files(workspace, files, merged, GENERATED_ARTIFACT_GLOBS)
    repository_exclude = _workspace_exclude(workspace)
    if repository_exclude is not None:
        files = {f for f in files if not _should_exclude(f.relative_to(workspace).as_posix(), repository_exclude)}
    _include_bundle_files(workspace, files, merged, extra_includes)
    return sorted(f for f in files if f.is_file())


def create_workspace_zip(
    workspace: str | Path,
    *,
    exclude: re.Pattern[str] | None = None,
    extra_includes: Sequence[str] = (),
    max_size_bytes: int | None = MAX_BUNDLE_SIZE_BYTES,
) -> bytes:
    """Create a zip of the workspace and return the raw bytes.

    Suitable for Iris bundle uploads where the caller sends bytes directly.
    ``extra_includes`` re-includes caller-specified gitignored or repository-excluded files (see
    :func:`collect_workspace_files`).
    """
    workspace = Path(workspace)
    files = collect_workspace_files(workspace, exclude=exclude, extra_includes=extra_includes)

    fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="workspace_")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in files:
                zf.write(file, file.relative_to(workspace))
        buf = Path(tmp_path).read_bytes()
    finally:
        os.unlink(tmp_path)

    if max_size_bytes is not None and len(buf) > max_size_bytes:
        size_mb = len(buf) / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        raise ValueError(
            f"Bundle size {size_mb:.1f}MB exceeds maximum {max_mb:.0f}MB. "
            "Consider excluding large files or using .gitignore."
        )

    return buf


def _get_git_non_ignored_files(
    workspace: Path,
    exclude: re.Pattern[str],
) -> set[Path] | None:
    """Get files that are not ignored by git.

    Returns None if git is not available or this isn't a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f for f in result.stdout.splitlines() if f and not _should_exclude(f, exclude)]
        return {workspace / f for f in files}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug("Git not available, using pattern-based exclusion: %s", e)
        return None


def _include_bundle_files(
    workspace: Path,
    files: set[Path],
    exclude: re.Pattern[str],
    patterns: Sequence[str],
) -> None:
    """Add files matching runtime artifact or explicit caller globs."""
    added = 0
    for pattern in patterns:
        for path in workspace.glob(pattern):
            if path.is_file() and path not in files and not _should_exclude(str(path.relative_to(workspace)), exclude):
                files.add(path)
                added += 1
    if added:
        logger.debug("Included %d generated build artifact(s) in bundle", added)
