# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture Snowball parity reports and promote reviewed native reports."""

from pathlib import Path

import click
from rigging.filesystem import StoragePath

from tests.cluster.vllm.backend_parity import ObservationReport
from tests.cluster.vllm.snowball import (
    SNOWBALL_EXPORTED_GPU,
    SNOWBALL_EXPORTED_TPU,
    SNOWBALL_NATIVE_GPU,
    SNOWBALL_NATIVE_TPU,
    SNOWBALL_VLLM_GPU,
    SNOWBALL_VLLM_TPU,
    golden_digest_for_platform,
    read_representative_goldens,
)
from tests.cluster.vllm.snowball_exported_levanter import capture_exported_levanter
from tests.cluster.vllm.snowball_levanter import capture_native_levanter
from tests.cluster.vllm.snowball_promote_golden import GoldenPromotionProvenance, promote_native_report
from tests.cluster.vllm.snowball_vllm import capture_vllm


def _write_report(report: ObservationReport, output: str) -> None:
    StoragePath(output).write_bytes((report.model_dump_json() + "\n").encode())


@click.group()
def cli() -> None:
    """Create or promote Snowball next-token parity records."""


@cli.command("native")
@click.option("--platform", type=click.Choice(("gpu", "tpu")), required=True)
@click.option("--output", required=True)
@click.option("--same-process-repeats", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--batch-index", type=int, multiple=True)
@click.option("--include-gpu-reference", is_flag=True)
@click.option("--candidate-revision", default="development", show_default=True)
def capture_native(
    platform: str,
    output: str,
    same_process_repeats: int,
    batch_index: tuple[int, ...],
    include_gpu_reference: bool,
    candidate_revision: str,
) -> None:
    """Capture the regional native Levanter checkpoint."""
    if include_gpu_reference and platform != "tpu":
        raise click.UsageError("--include-gpu-reference is only meaningful for TPU captures")
    cell = SNOWBALL_NATIVE_GPU if platform == "gpu" else SNOWBALL_NATIVE_TPU
    report = capture_native_levanter(
        cell,
        same_process_repeats=same_process_repeats,
        batch_indices=batch_index or None,
        additional_goldens=(read_representative_goldens(),) if include_gpu_reference else (),
        golden_digest=golden_digest_for_platform(platform),
        candidate_revision=candidate_revision,
    )
    _write_report(report, output)


@cli.command("exported")
@click.option("--platform", type=click.Choice(("gpu", "tpu")), required=True)
@click.option("--output", required=True)
@click.option("--same-process-repeats", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--batch-index", type=int, multiple=True)
@click.option("--candidate-revision", default="development", show_default=True)
def capture_exported(
    platform: str,
    output: str,
    same_process_repeats: int,
    batch_index: tuple[int, ...],
    candidate_revision: str,
) -> None:
    """Capture Levanter after loading the regional HF export."""
    cell = SNOWBALL_EXPORTED_GPU if platform == "gpu" else SNOWBALL_EXPORTED_TPU
    report = capture_exported_levanter(
        cell,
        same_process_repeats=same_process_repeats,
        batch_indices=batch_index or None,
        golden_digest=golden_digest_for_platform(platform),
        candidate_revision=candidate_revision,
    )
    _write_report(report, output)


@cli.command("vllm")
@click.option("--platform", type=click.Choice(("gpu", "tpu")), required=True)
@click.option("--output", required=True)
@click.option("--same-process-repeats", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--batch-index", type=int, multiple=True)
@click.option("--case-id", multiple=True)
@click.option("--candidate-revision", default="development", show_default=True)
def capture_served_vllm(
    platform: str,
    output: str,
    same_process_repeats: int,
    batch_index: tuple[int, ...],
    case_id: tuple[str, ...],
    candidate_revision: str,
) -> None:
    """Capture the standard vLLM OpenAI endpoint."""
    cell = SNOWBALL_VLLM_GPU if platform == "gpu" else SNOWBALL_VLLM_TPU
    report = capture_vllm(
        cell,
        same_process_repeats=same_process_repeats,
        batch_indices=batch_index or None,
        case_ids=case_id or None,
        golden_digest=golden_digest_for_platform(platform),
        candidate_revision=candidate_revision,
    )
    _write_report(report, output)


@cli.command("promote")
@click.option("--report-uri", required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--manifest", type=click.Path(path_type=Path), required=True)
@click.option("--platform", type=click.Choice(("gpu", "tpu")), required=True)
@click.option(
    "--expected-provenance",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
def promote(
    report_uri: str,
    output: Path,
    manifest: Path,
    platform: str,
    expected_provenance: Path,
) -> None:
    """Promote a complete native report after validating reviewed provenance."""
    promote_native_report(
        report_uri=report_uri,
        output_path=output,
        manifest_path=manifest,
        platform=platform,
        expected_provenance=GoldenPromotionProvenance.model_validate_json(expected_provenance.read_bytes()),
    )


if __name__ == "__main__":
    cli()
