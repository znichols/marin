# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cached MMAU inputs reach the independent generator without refetching."""

import hashlib
import json
import os
import threading
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from click.testing import CliRunner
from fray.current_client import set_current_client
from fray.local_backend import LocalClient

from experiments.evaluation import generate_mmau


@pytest.fixture
def source_server(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    payload = [
        {
            "id": "fixture",
            "question": "Which sound?",
            "transcript": "A bell.",
            "choices": ["Bell", "Dog"],
            "answer": "Bell",
        }
    ]
    source_file = source_dir / generate_mmau.MMAU_SOURCE.filename
    source_file.write_text(json.dumps(payload))
    handler = partial(SimpleHTTPRequestHandler, directory=str(source_dir))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield replace(
                generate_mmau.MMAU_SOURCE,
                url=f"http://127.0.0.1:{server.server_port}/{source_file.name}",
                sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
            ), payload
        finally:
            server.shutdown()
            thread.join()


def test_generator_uses_verified_cached_input(tmp_path, monkeypatch, source_server):
    source, payload = source_server
    monkeypatch.setattr(generate_mmau, "MMAU_SOURCE", source)
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    uv = binary_dir / "uv"
    uv.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

args = sys.argv[1:]
dataset = Path(args[args.index('--dataset-path') + 1])
output = Path(args[args.index('--output-dir') + 1])
output.mkdir(parents=True)
(output / 'generated.json').write_text(json.dumps({
    'records': json.loads(dataset.read_text()),
    'source': dataset.name,
    'limit': args[args.index('--limit') + 1],
}))
"""
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary_dir}{os.pathsep}{os.environ['PATH']}")
    cache = tmp_path / "artifacts"
    output = tmp_path / "tasks"
    args = ["--artifact-root", str(cache), "--", "--output-dir", str(output), "--limit", "1"]
    client = LocalClient()
    try:
        with set_current_client(client):
            result = CliRunner().invoke(generate_mmau.main, args)
            assert result.exit_code == 0, result.exception
            expected = {"records": payload, "source": source.filename, "limit": "1"}
            assert json.loads((output / "generated.json").read_text()) == expected

            def unexpected_download(*_args, **_kwargs):
                raise AssertionError("A completed input artifact must be reusable without HTTP")

            monkeypatch.setattr(requests, "get", unexpected_download)
            second_output = tmp_path / "second-tasks"
            args[args.index(str(output))] = str(second_output)
            repeated = CliRunner().invoke(generate_mmau.main, args)
            assert repeated.exit_code == 0, repeated.exception
            assert json.loads((second_output / "generated.json").read_text()) == expected
    finally:
        client.shutdown()
