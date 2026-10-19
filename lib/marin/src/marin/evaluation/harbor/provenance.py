# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity for Marin's vendored Harbor runtime."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from marin.evaluation.harbor.driver_protocol import FULL_GIT_COMMIT_LENGTH

_RUNTIME_SOURCE_DIRECTORIES = (
    "vendor/harbor/src",
    "lib/rigging/src",
    "lib/finestore/src",
)
_RUNTIME_MANIFESTS = (
    "vendor/harbor/UPSTREAM.toml",
    "vendor/harbor/pyproject.toml",
    "lib/rigging/pyproject.toml",
    "lib/finestore/pyproject.toml",
)


@dataclass(frozen=True)
class HarborSource:
    repository: str
    commit: str
    version: str


def vendored_harbor_source(project: Path) -> HarborSource:
    """Read the upstream identity preserved alongside the vendored source."""
    with (project / "UPSTREAM.toml").open("rb") as metadata_file:
        metadata = tomllib.load(metadata_file)
    source = HarborSource(metadata["repository"], metadata["commit"], metadata["version"])
    if len(source.commit) != FULL_GIT_COMMIT_LENGTH or any(
        character not in "0123456789abcdef" for character in source.commit
    ):
        raise ValueError(f"Harbor source metadata in {project} does not identify an upstream commit")
    return source


def vendored_runtime_digest(workspace_root: Path) -> str:
    """Hash packaged Harbor assets and the local Marin dependencies it imports."""
    paths = [workspace_root / relative for relative in _RUNTIME_MANIFESTS]
    for relative in _RUNTIME_SOURCE_DIRECTORIES:
        directory = workspace_root / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(workspace_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return f"sha256:{digest.hexdigest()}"
