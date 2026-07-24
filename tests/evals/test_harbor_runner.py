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
