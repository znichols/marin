# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate MMAU tasks from a checksum-pinned, cached transcription dataset."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from marin.evaluation.harbor.driver_config import harbor_runtime_command
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.pipeline import UrlSource, download_step

MMAU_SOURCE = UrlSource(
    url=(
        "https://raw.githubusercontent.com/marin-community/harbor/"
        "3010980a375c6487d39a6b4f921ac61709e5e003/"
        "adapters/mmau/mmau-test-mini-transcribe-qwen.json"
    ),
    sha256="2058e6ca2ebad0f669f5478bfca02b2476b17786fb6ca1906e4c56599ffec425",
    filename="mmau-test-mini-transcribe-qwen.json",
    format=SourceFormat.JSON,
)
_ADAPTER_SCRIPT = Path(__file__).resolve().parents[2] / "vendor/harbor/adapters/mmau/src/mmau/main.py"


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--artifact-root", required=True, help="Local or region-local object-store prefix for cached input.")
@click.argument("generator_args", nargs=-1, type=click.UNPROCESSED)
def main(artifact_root: str, generator_args: tuple[str, ...]) -> None:
    """Cache the pinned input, then forward arguments after -- to the MMAU generator."""
    with artifact_prefix(artifact_root):
        artifact = run(download_step(MMAU_SOURCE, CampaignRuntime()), max_concurrent=1)[0]
    with TemporaryDirectory(prefix="marin-mmau-") as directory:
        dataset_path = Path(directory) / MMAU_SOURCE.filename
        copy(str(StoragePath(artifact.path) / MMAU_SOURCE.filename), str(dataset_path))
        subprocess.run(
            harbor_runtime_command(
                _ADAPTER_SCRIPT,
                *generator_args,
                "--dataset-path",
                str(dataset_path),
            ),
            check=True,
        )


if __name__ == "__main__":
    main()
