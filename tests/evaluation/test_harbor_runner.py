# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import Mock

from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.runner import HarborExecutor, HarborRunConfig, HarborTrial, _write_samples, run_harbor
from marin.inference.types import OpenAIEndpoint, RunningModel

from experiments.evaluation.evals import EVALS
from experiments.evaluation.models import models


def test_materialize_harbor_dataset_downloads_hf_revision_as_local_tasks(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / ".gitattributes").write_text("*.gz filter=lfs")
    (snapshot / "task-one").mkdir()
    calls: list[dict] = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("marin.evaluation.harbor.dataset.snapshot_download", download)

    path = materialize_harbor_dataset(
        "hf://DCAgent2/terminal_bench_2",
        "main",
        tmp_path / "workdir",
        hf_token=None,
    )

    assert path == Path(snapshot)
    assert calls == [
        {
            "repo_id": "DCAgent2/terminal_bench_2",
            "repo_type": "dataset",
            "revision": "main",
            "local_dir": str(tmp_path / "workdir" / "hf_dataset"),
            "cache_dir": str(tmp_path / "workdir" / "hf_cache"),
            "token": False,
        }
    ]
    assert not (snapshot / ".gitattributes").exists()


def test_write_samples_uses_a_path_safe_name_for_hf_dataset(tmp_path):
    trial = HarborTrial(task_id="task-one", reward=0.0, status="completed", trajectory=None, error=None)

    path = _write_samples([trial], "hf://DCAgent2/terminal_bench_2", str(tmp_path))

    assert path == str(tmp_path / "samples_harbor.parquet")
    assert Path(path).exists()


def test_run_harbor_derives_url_and_served_name_from_running_model(tmp_path, monkeypatch):
    captured: dict = {}

    def run_driver(config_file: Path) -> None:
        captured.update(json.loads(config_file.read_text())["job_config"])
        trial_dir = Path(captured["jobs_dir"]) / captured["job_name"] / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner._run_driver", run_driver)
    monkeypatch.setattr("marin.evaluation.harbor.runner.materialize_harbor_dataset", Mock(return_value=None))
    model = RunningModel(
        endpoint=OpenAIEndpoint(
            base_url="https://iris.example/proxy/t/token/serve.model/v1",
            model="qwen3-0.6b",
        )
    )

    result = run_harbor(
        model,
        HarborRunConfig(dataset=f"toy-{tmp_path.name}", version="1.0", agent="terminus-2"),
        str(tmp_path),
        hf_token=None,
    )

    assert captured["agents"][0]["model_name"] == "hosted_vllm/qwen3-0.6b"
    assert captured["agents"][0]["kwargs"]["api_base"] == model.endpoint.base_url
    assert captured["datasets"] == [{"name": f"toy-{tmp_path.name}", "version": "1.0", "n_tasks": None}]
    assert result.total_trials == 1
    assert result.accuracy == 1.0


def test_grug_opencode_policy_binds_the_capped_dataset_and_endpoint(tmp_path, monkeypatch):
    captured: dict = {}

    def run_driver(config_file: Path) -> None:
        captured.update(json.loads(config_file.read_text())["job_config"])
        job_dir = Path(captured["jobs_dir"]) / captured["job_name"]
        job_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("marin.evaluation.harbor.runner._run_driver", run_driver)
    monkeypatch.setattr("marin.evaluation.harbor.runner.materialize_harbor_dataset", Mock(return_value=None))
    definition = EVALS["grug-opencode-id"]
    executor = definition.executor_for(models()["grug-agentic-s3-step1903"], limit=1)
    assert isinstance(executor, HarborExecutor)
    model = RunningModel(
        endpoint=OpenAIEndpoint(
            base_url="https://iris.example/proxy/t/token/serve.grug/v1",
            model="grug-agentic-s3-step1903",
        )
    )

    run_harbor(model, executor.config, str(tmp_path), hf_token=None)

    assert captured["n_attempts"] == 3
    assert captured["n_concurrent_trials"] == 256
    assert captured["timeout_multiplier"] == 2.0
    assert captured["retry"]["max_retries"] == 6
    assert "AgentTimeoutError" in captured["retry"]["exclude_exceptions"]
    assert captured["environment"] == {
        "type": "daytona",
        "force_build": True,
        "delete": True,
        "override_cpus": 2,
        "override_memory_mb": 8192,
        "override_storage_mb": 8192,
        "kwargs": {"auto_snapshot": True},
    }
    assert captured["verifier"]["max_timeout_sec"] == 14400
    assert captured["datasets"] == [
        {
            "name": "DCAgent/dev_set_v2",
            "ref": "1.0",
            "n_tasks": 1,
        }
    ]
    agent = captured["agents"][0]
    assert agent["name"] == "opencode"
    assert agent["model_name"] == "hosted_vllm/grug-agentic-s3-step1903"
    assert agent["max_timeout_sec"] == 7200
    assert agent["override_setup_timeout_sec"] == 600
    assert agent["kwargs"]["api_base"] == model.endpoint.base_url
    assert agent["kwargs"]["model_info"]["max_input_tokens"] == 64512
    assert agent["kwargs"]["trajectory_config"] == {"raw_content": False, "linear_history": True}
