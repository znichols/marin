# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One Snowball golden-comparison matrix across inference methods and platforms.

PYTEST_DONT_REWRITE: serialized remote functions must not depend on pytest.
"""

import uuid

import pytest
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from rigging.filesystem import StoragePath

from tests.cluster.vllm.backend_parity import persist_report_before_assert
from tests.cluster.vllm.snowball import (
    PARITY_CELLS,
    SNOWBALL_GPU,
    SNOWBALL_TPU,
    ParityCell,
    golden_digest_for_platform,
    read_goldens_for_platform,
    read_prompt_fixture,
    report_expectation,
)
from tests.cluster.vllm.snowball_contracts import NUMERICAL_SOURCE_REVISION

PENDING_TIMEOUT = 30 * 60.0
RUNTIME_TIMEOUT = 60 * 60.0
pytestmark = [pytest.mark.cluster, pytest.mark.slow, pytest.mark.timeout(PENDING_TIMEOUT + RUNTIME_TIMEOUT + 300)]


def run_parity_cell(cell, goldens, additional_goldens, candidate_revision, report_uri) -> None:
    """Capture once, persist once, then run every independent cell check."""
    report = cell.capture(
        goldens=goldens,
        additional_goldens=additional_goldens,
        golden_digest=golden_digest_for_platform(cell.platform),
        candidate_revision=candidate_revision,
    )
    fixture = read_prompt_fixture(goldens, fixture_uri=cell.location.prompt_fixture_uri)
    expectation = report_expectation(fixture, data_parallel_size=cell.data_parallel_size)
    persist_report_before_assert(
        report,
        report_uri,
        cell.comparison_checks(report, goldens, expectation),
    )


def _job(cell: ParityCell, run_id: str) -> JobRequest:
    goldens = read_goldens_for_platform(cell.platform)
    additional_goldens = tuple(read_goldens_for_platform(platform) for platform in cell.additional_reference_platforms)
    report_uri = str(StoragePath(cell.artifact_root) / run_id / f"{cell.name}.json")
    entrypoint = Entrypoint.from_callable(
        run_parity_cell,
        args=[cell, goldens, additional_goldens, NUMERICAL_SOURCE_REVISION, report_uri],
    )
    if cell.platform == "gpu":
        if cell.backend == "vllm":
            environment = create_environment(
                setup_scripts=[default_setup_script(packages=["marin-core"])],
                env_vars={
                    "VLLM_BATCH_INVARIANT": "1",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                },
            )
            ram = "512g"
        else:
            environment = create_environment(
                extras=["gpu"],
                sync_packages=["marin-levanter", "marin-core"],
                env_vars={
                    "JAX_COMPILATION_CACHE_DIR": SNOWBALL_GPU.compilation_cache_dir,
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
                },
            )
            ram = "256g"
        resources = ResourceConfig.with_gpu("H100", count=8, cpu=64, ram=ram, disk="128g")
    else:
        environment = create_environment(
            extras=["tpu"],
            sync_packages=["marin-core"] if cell.backend == "vllm" else ["marin-levanter", "marin-core"],
            env_vars=(
                {}
                if cell.backend == "vllm"
                else {
                    "JAX_COMPILATION_CACHE_DIR": SNOWBALL_TPU.compilation_cache_dir,
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
                }
            ),
        )
        resources = ResourceConfig.with_tpu(
            "v6e-8",
            cpu=160 if cell.backend == "vllm" else 64,
            ram="640g" if cell.backend == "vllm" else "256g",
            disk="100g",
            regions=("us-east5",),
        )
    return JobRequest(
        name=f"snowball-{cell.name}-{run_id[:8]}",
        entrypoint=entrypoint,
        resources=resources,
        environment=environment,
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
    )


@pytest.mark.parametrize("cell", PARITY_CELLS, ids=lambda cell: cell.name)
def test_snowball_parity_matrix(
    cell: ParityCell,
    request: pytest.FixtureRequest,
    run_test_job,
) -> None:
    client_fixture = "marin_gpu_client" if cell.platform == "gpu" else "iris_client"
    client: IrisClient = request.getfixturevalue(client_fixture)
    run_id = uuid.uuid4().hex
    run_test_job(
        client,
        _job(cell, run_id),
        pending_timeout=PENDING_TIMEOUT,
        runtime_timeout=RUNTIME_TIMEOUT,
    )
