# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import posixpath
import time

from fsspec.callbacks import TqdmCallback
from rigging.filesystem import StoragePath, url_to_fs

logger = logging.getLogger(__name__)


def download_from_storage(source_path: str, destination_path: str) -> None:
    """Download a remote directory unless ``destination_path`` already exists."""
    if os.path.exists(destination_path):
        print(f"Skipping download: {destination_path} already exists.")
        return

    print(f"Downloading {source_path} to {destination_path}.")
    start_time: float = time.time()
    fs, source_key = url_to_fs(source_path)

    if not fs.exists(source_key):
        raise FileNotFoundError(f"{source_path} does not exist.")

    # The slash is needed to download the contents of the folder to `destination_path`
    os.makedirs(destination_path, exist_ok=True)
    fs.get(posixpath.join(source_key, ""), destination_path, recursive=True, callback=TqdmCallback())

    elapsed_time_seconds: float = time.time() - start_time
    print(f"Downloaded {source_path} to {destination_path} ({elapsed_time_seconds:.2f}s).")


def upload_to_storage(local_path: str, destination_path: str) -> None:
    """Upload a local directory to an object-store destination."""
    print(f"Uploading {local_path}.")
    fs, destination_key = url_to_fs(destination_path)
    # The slash uploads the directory's contents rather than nesting its basename.
    fs.put(os.path.join(local_path, ""), destination_key, recursive=True)
    logger.info("Uploaded %s to %s.", local_path, destination_path)


def discover_hf_checkpoints(base_path: str) -> list[str]:
    """Discover Hugging Face checkpoints under ``base_path``, sorted by mtime ascending (most recent last).

    A directory counts as a checkpoint when it contains both ``config.json``
    (matched by the glob) and ``tokenizer_config.json``.
    """
    config_paths = [str(m) for m in StoragePath(os.path.join(base_path, "**/config.json")).glob()]
    config_paths.sort(key=lambda path: StoragePath(path).mtime())
    return [
        os.path.dirname(path)
        for path in config_paths
        if StoragePath(os.path.join(os.path.dirname(path), "tokenizer_config.json")).exists()
    ]
