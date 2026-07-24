# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import marin.inference.vllm_backend as vllm_backend


class _ObjectStore:
    def __init__(self) -> None:
        self.files = {
            "bucket/checkpoint/config.json": b'{"architectures": ["Test"]}',
            "bucket/checkpoint/model.safetensors": b"weights",
        }

    def find(self, root: str) -> list[str]:
        assert root == "bucket/checkpoint"
        return list(self.files)

    def info(self, remote_file: str) -> dict[str, int | str]:
        return {"type": "file", "size": len(self.files[remote_file])}

    def get_file(self, remote_file: str, local_file: str) -> None:
        Path(local_file).write_bytes(self.files[remote_file])


def test_object_store_models_are_staged_to_a_deterministic_local_directory(monkeypatch, tmp_path) -> None:
    store = _ObjectStore()
    monkeypatch.setattr(vllm_backend, "url_to_fs", lambda _path: (store, "bucket/checkpoint"))
    monkeypatch.setenv(vllm_backend._LOCAL_MODEL_STAGING_ROOT_ENV, str(tmp_path))

    staged = Path(vllm_backend.stage_object_store_model_locally("s3://bucket/checkpoint"))

    assert staged.parent == tmp_path / "marin-models"
    assert (staged / "config.json").read_bytes() == store.files["bucket/checkpoint/config.json"]
    assert (staged / "model.safetensors").read_bytes() == store.files["bucket/checkpoint/model.safetensors"]


def test_local_models_do_not_need_object_store_staging() -> None:
    assert vllm_backend.stage_object_store_model_locally("/models/checkpoint") == "/models/checkpoint"
