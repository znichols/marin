# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The eval group and its later serving child must share the requested Iris policy."""

from pathlib import Path

from iris.cluster.config import load_config

from experiments.evaluation.hardware import Platform
from experiments.evaluation.launch import LaunchSpec, launch_group, plan_runs


def _spec(**overrides) -> LaunchSpec:
    values = dict(
        model="qwen3-8b",
        evals=("grug-opencode-id",),
        platform=Platform.GPU,
        accelerator="H100x8",
        limit=2,
        records_prefix="s3://marin-us-east-02a/test/evals",
        cluster="marin",
        priority=2,  # PRIORITY_BAND_INTERACTIVE
        max_retries_failure=6,
    )
    values.update(overrides)
    return LaunchSpec(**values)


def test_plan_carries_explicit_priority_and_retries_to_the_serving_child():
    serve = plan_runs(_spec())[0].serve

    assert serve.priority == 2
    assert serve.max_retries_failure == 6


def test_group_submission_carries_the_same_priority_and_retries(monkeypatch):
    captured = {}

    class _Job:
        def __str__(self) -> str:
            return "/test/eval-group"

    class _Client:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return _Job()

    # Harbor runs require a Daytona control-plane key.  Keep this unit test
    # hermetic: production resolves the approved GSM secret only when callers
    # have not supplied the launch-environment value.
    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "test-daytona-key")
    monkeypatch.setattr("experiments.evaluation.launch._git_sha", lambda: "test")
    launch_group(_spec(), _Client())

    assert captured["max_retries_failure"] == 6
    assert captured["priority_band"] == 2


def test_direct_target_cluster_launch_does_not_create_a_self_federation_constraint(monkeypatch):
    """A direct RNO controller cannot schedule to itself as a federation peer."""
    captured = {}

    class _Job:
        def __str__(self) -> str:
            return "/test/eval-group"

    class _Client:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return _Job()

    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "test-daytona-key")
    monkeypatch.setattr("experiments.evaluation.launch._git_sha", lambda: "test")
    launch_group(_spec(model="grug-agentic-s3-step1903", accelerator=None, cluster="cw-rno2a"), _Client())

    assert captured["constraints"] is None


def test_rno2a_declares_the_public_origin_needed_by_daytona_harbor_agents():
    """A Harbor agent outside CoreWeave must receive a minted, public endpoint URL."""
    config_path = Path(__file__).parents[2] / "lib/iris/config/cw-rno2a.yaml"

    assert load_config(config_path).dashboard_url == "https://iris-cw-rno2a.oa.dev"
