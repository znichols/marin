# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import (
    HarborAgentConfig,
    HarborDriverConfig,
    HarborEnvironmentConfig,
    HarborRetryConfig,
    HarborRunConfig,
    HarborVerifierConfig,
    native_job_config,
)
from marin.evaluation.harbor.runner import (
    HarborTrial,
    _restore_completed_trials,
    _upload_trials,
    _write_samples,
    run_harbor,
)
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem import StoragePath


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


def test_harbor_trial_tree_round_trips_object_storage(tmp_path):
    completed_trial = tmp_path / "completed" / "trial-one"
    (completed_trial / "agent").mkdir(parents=True)
    (completed_trial / "result.json").write_text('{"task_name": "trial-one"}')
    (completed_trial / "agent" / "trajectory.json").write_text('{"steps": []}')
    storage_root = f"memory://{tmp_path.name}/eval"

    _upload_trials(completed_trial.parent, storage_root)

    stored_result = StoragePath(storage_root) / "harbor_trials" / "trial-one" / "result.json"
    assert stored_result.read_text() == '{"task_name": "trial-one"}'
    restored = tmp_path / "restored"
    assert _restore_completed_trials(storage_root, restored) == 1
    assert (restored / "trial-one" / "agent" / "trajectory.json").read_text() == '{"steps": []}'


def test_native_job_config_translates_the_external_harbor_contract():
    config = HarborDriverConfig(
        job_name="job",
        jobs_dir="/tmp/jobs",
        dataset_path=None,
        endpoint_url="https://iris.example/v1",
        served_model="served-model",
        run=HarborRunConfig(
            dataset="aime",
            revision="1.0",
            agent=HarborAgentConfig(
                name="opencode",
                max_output_tokens=16384,
                max_timeout=7200,
                setup_timeout=600,
                kwargs={"model_info": {"max_input_tokens": 64512}},
            ),
            environment=HarborEnvironmentConfig(
                environment_type="daytona",
                cpus=2,
                memory_mb=8192,
                storage_mb=8192,
            ),
            attempts=3,
            retry=HarborRetryConfig(max_retries=6, min_wait=2.0, max_wait=90.0),
            verifier=HarborVerifierConfig(max_timeout=14400),
        ),
    )

    native = native_job_config(config)

    assert native["datasets"] == [{"name": "aime", "version": "1.0", "n_tasks": None}]
    assert native["n_attempts"] == 3
    assert native["retry"]["min_wait_sec"] == 2.0
    assert native["retry"]["max_wait_sec"] == 90.0
    assert native["environment"]["override_cpus"] == 2
    assert native["environment"]["override_memory_mb"] == 8192
    assert native["environment"]["override_storage_mb"] == 8192
    assert native["verifier"]["max_timeout_sec"] == 14400
    agent = native["agents"][0]
    assert agent["model_name"] == "hosted_vllm/served-model"
    assert agent["override_setup_timeout_sec"] == 600
    assert agent["kwargs"]["api_base"] == "https://iris.example/v1"
    assert agent["kwargs"]["model_info"]["max_input_tokens"] == 64512
    assert agent["kwargs"]["model_info"]["max_output_tokens"] == 16384


def test_run_harbor_normalizes_a_completed_external_trial(tmp_path, monkeypatch):
    captured: dict = {}

    def run_driver(command, *, check, env) -> None:
        assert check
        driver_config = HarborDriverConfig.from_dict(json.loads(Path(command[-1]).read_text()))
        captured["driver_config"] = driver_config
        captured["env"] = env
        trial_dir = Path(driver_config.jobs_dir) / driver_config.job_name / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner.subprocess.run", run_driver)
    monkeypatch.setenv("DAYTONA_API_KEY", "daytona-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-harbor")
    model = RunningModel(
        endpoint=OpenAIEndpoint(
            base_url="https://iris.example/proxy/t/token/serve.model/v1",
            model="qwen3-0.6b",
        )
    )

    result = run_harbor(
        model,
        HarborRunConfig(
            dataset=f"toy-{tmp_path.name}",
            revision="1.0",
            agent=HarborAgentConfig(name="terminus-2"),
            environment=HarborEnvironmentConfig(environment_type="daytona"),
        ),
        str(tmp_path),
        hf_token=None,
    )

    assert captured["driver_config"].endpoint_url == model.endpoint.base_url
    assert captured["driver_config"].served_model == "qwen3-0.6b"
    assert captured["env"]["DAYTONA_API_KEY"] == "daytona-key"
    assert "OPENAI_API_KEY" not in captured["env"]
    assert result.total_trials == 1
    assert result.accuracy == 1.0
