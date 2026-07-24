# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from marin.evaluation import harbor_runner


def test_harbor_driver_executes_by_run_path_without_shadowing_harbor_config(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(harbor_runner.subprocess, "run", lambda cmd, check: captured.update(cmd=cmd, check=check))

    harbor_runner._run_driver(Path(tmp_path / "driver_config.json"))

    command = captured["cmd"]
    python_index = command.index("python")
    assert command[python_index + 1 : python_index + 3] == ["-c", harbor_runner._DRIVER_RUNNER]
    assert command[python_index + 3] == harbor_runner._DRIVER
    assert "runpy.run_path" in harbor_runner._DRIVER_RUNNER
    assert captured["check"] is True


def test_materialize_hf_dataset_uses_the_explicit_revision(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "dataset")

    monkeypatch.setattr(harbor_runner, "snapshot_download", fake_snapshot_download)

    result = harbor_runner._materialize_hf_dataset("hf://org/harbor-tasks", "deadbeef", tmp_path)

    assert result == tmp_path / "dataset"
    assert captured["repo_id"] == "org/harbor-tasks"
    assert captured["revision"] == "deadbeef"
    assert captured["repo_type"] == "dataset"


def test_non_hf_dataset_does_not_materialize(tmp_path):
    assert harbor_runner._materialize_hf_dataset("aime", "1.0", tmp_path) is None
