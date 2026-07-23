# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for native vLLM server log routing.

``_LogPump`` forwards the subprocess's stdout/stderr to the parent's fds and to the on-disk logs,
routing by severity and flushing/draining on teardown; these exercise that.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time

import pytest
from marin.inference.vllm_server import (
    _NATIVE_LOG_TAIL_LINES,
    TransientStartupError,
    VllmServerHandle,
    _engine_kwargs_to_cli_args,
    _log_file_contains,
    _LogPump,
    _native_logs_tail,
    _start_vllm_native_server,
)
from rigging.timing import ExponentialBackoff


def test_engine_kwargs_forward_dtype_to_vllm_command() -> None:
    assert _engine_kwargs_to_cli_args({"dtype": "float16"}) == ["--dtype", "float16"]


def _spawn(script: str, *, start_new_session: bool = False) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=start_new_session,
    )


def test_log_pump_forwards_to_parent_fds_and_persists(tmp_path, capsys):
    # The child writes an INFO line on each of its streams plus an ERROR line. Severity, not the
    # source stream, picks the parent fd: INFO -> parent stdout, ERROR -> parent stderr.
    script = (
        "import sys\n"
        "print('INFO worker throughput: 42 tokens/s')\n"
        "sys.stdout.flush()\n"
        "print('INFO 07-17 gen throughput: 100.0 tokens/s', file=sys.stderr)\n"
        "print('ERROR 07-17 EngineCore boom', file=sys.stderr)\n"
        "sys.stderr.flush()\n"
    )
    proc = _spawn(script)
    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"
    pump = _LogPump(proc, str(stdout_log), str(stderr_log))
    pump.start()
    assert proc.wait(timeout=10) == 0
    pump.join(timeout=5)
    pump.close()

    # On-disk logs keep stdout/stderr provenance (they back diagnostics() and the failure tail).
    assert "worker throughput: 42 tokens/s" in stdout_log.read_text()
    stderr_text = stderr_log.read_text()
    assert "gen throughput: 100.0 tokens/s" in stderr_text
    assert "EngineCore boom" in stderr_text

    # Both INFO lines (including the one the child wrote to its stderr) go to the parent's stdout;
    # only the ERROR line goes to stderr.
    captured = capsys.readouterr()
    assert "worker throughput: 42 tokens/s" in captured.out
    assert "gen throughput: 100.0 tokens/s" in captured.out
    assert "EngineCore boom" not in captured.out
    assert "EngineCore boom" in captured.err


def test_native_logs_tail_sees_final_lines_after_join(tmp_path):
    # The startup-failure path joins the pump before building its diagnostic, so the tail must
    # include the child's final lines once join() returns.
    script = "import sys; print('LAST_STDOUT_LINE'); print('LAST_STDERR_LINE', file=sys.stderr)"
    proc = _spawn(script)
    pump = _LogPump(proc, str(tmp_path / "stdout.log"), str(tmp_path / "stderr.log"))
    pump.start()
    proc.wait(timeout=10)
    pump.join(timeout=5)

    tail = _native_logs_tail(str(tmp_path))
    assert "LAST_STDOUT_LINE" in tail
    assert "LAST_STDERR_LINE" in tail
    pump.close()


def test_native_logs_tail_includes_unterminated_final_fragment(tmp_path):
    # A child that crashes mid-line leaves a final fragment with no trailing newline. The pump
    # flushes on EOF so the startup-failure tail — read right after join(), before close() — sees
    # it; without that flush the line-buffered file would hold the newline-less fragment.
    proc = _spawn("import sys; sys.stderr.write('FATAL partial line no newline'); sys.stderr.flush()")
    pump = _LogPump(proc, str(tmp_path / "stdout.log"), str(tmp_path / "stderr.log"))
    pump.start()
    proc.wait(timeout=10)
    pump.join(timeout=5)

    assert "FATAL partial line no newline" in _native_logs_tail(str(tmp_path))
    pump.close()


def test_handle_stop_terminates_drains_and_is_idempotent(tmp_path):
    # The child logs a line, then blocks; stop() must terminate it, drain that line to the
    # on-disk log, and be safe to call again.
    proc = _spawn("import sys, time; print('SERVE_READY'); sys.stdout.flush(); time.sleep(30)", start_new_session=True)
    pump = _LogPump(proc, str(tmp_path / "stdout.log"), str(tmp_path / "stderr.log"))
    pump.start()
    try:
        process_group_id = os.getpgid(proc.pid)
    except ProcessLookupError:
        process_group_id = None
    handle = VllmServerHandle(
        server_url="http://127.0.0.1:0/v1",
        port=0,
        process=proc,
        process_group_id=process_group_id,
        log_dir=str(tmp_path),
        log_pump=pump,
    )

    # Wait until the child has started Python and its line is pumped to disk, so teardown below
    # is deterministic rather than racing the child's startup.
    deadline = time.monotonic() + 10
    while "SERVE_READY" not in _native_logs_tail(str(tmp_path)):
        if time.monotonic() > deadline:
            raise AssertionError("child never logged SERVE_READY")
        time.sleep(0.05)

    handle.stop(timeout_seconds=5)
    assert proc.poll() is not None  # terminated
    # Teardown flushed and closed the on-disk logs, so the tail still reads the child's output.
    assert "SERVE_READY" in _native_logs_tail(str(tmp_path))

    handle.stop(timeout_seconds=5)  # second call must not raise


# --- issue #7525: bounded retry around a transient Run:ai streamer read fault during startup ---

# Prints the libstreamer raise-site text the classifier keys on, then exits like the real crash.
_STREAMER_FAIL_LINE = (
    "print(\"ValueError: Could not receive runai_response from libstreamer due to: b'File access error'\","
    " file=sys.stderr)"
)
_FAST_BACKOFF = ExponentialBackoff(initial=0.001, maximum=0.01, factor=2.0, jitter=0.0)


def _serve_forever_script() -> str:
    # Binds the --port that _start_vllm_native_server appends to the launcher command and answers
    # /v1/models with 200 so readiness polling succeeds.
    return (
        "import sys, http.server\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers()\n"
        '        self.wfile.write(b\'{"data": [{"id": "fake-model"}]}\')\n'
        "    def log_message(self, *a): pass\n"
        "http.server.HTTPServer(('127.0.0.1', port), H).serve_forever()\n"
    )


def _fail_then_serve_script(counter_path: str, fail_until: int) -> str:
    # Fails with the streamer signature for the first `fail_until` starts, counting via a file that
    # persists across attempts, then serves.
    return (
        "import sys, os\n"
        f"counter = r'{counter_path}'\n"
        "n = int(open(counter).read()) + 1 if os.path.exists(counter) else 1\n"
        "open(counter, 'w').write(str(n))\n"
        f"if n <= {fail_until}:\n"
        f"    {_STREAMER_FAIL_LINE}\n"
        "    sys.exit(1)\n"
    ) + _serve_forever_script()


class _FakeLauncher:
    def __init__(self, script: str) -> None:
        self._script = script

    def command(self) -> list[str]:
        return [sys.executable, "-c", self._script]

    def env(self) -> dict[str, str]:
        return {}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _record_mkdtemp(monkeypatch) -> list[str]:
    """Record every log dir _start_vllm_native_server creates, so tests can count spawns and assert
    that failed attempts are cleaned up."""
    created: list[str] = []
    real = tempfile.mkdtemp

    def recording(*args, **kwargs):
        directory = real(*args, **kwargs)
        created.append(directory)
        return directory

    monkeypatch.setattr(tempfile, "mkdtemp", recording)
    return created


def test_log_file_contains_matches_beyond_the_capped_tail(tmp_path):
    # The marker sits near the top of a log far longer than the tail cap. Classification reads the
    # whole file, so it is found even though a tail-only view (used for the human-readable message)
    # would have scrolled past it — the reason classification does not reuse the tail.
    stderr = tmp_path / "stderr.log"
    lines = ["ValueError: Could not receive runai_response from libstreamer\n"]
    lines += [f"teardown noise line {i}\n" for i in range(_NATIVE_LOG_TAIL_LINES * 2)]
    stderr.write_text("".join(lines))
    (tmp_path / "stdout.log").write_text("nothing relevant\n")

    assert _log_file_contains(str(tmp_path), "could not receive runai_response")
    assert "runai_response" not in _native_logs_tail(str(tmp_path))
    assert not _log_file_contains(str(tmp_path), "cuda out of memory")


def test_startup_succeeds_on_first_attempt_without_retry(monkeypatch):
    created = _record_mkdtemp(monkeypatch)
    handle = _start_vllm_native_server(
        model_name_or_path="fake-model",
        port=_free_port(),
        timeout_seconds=30,
        launcher=_FakeLauncher(_serve_forever_script()),
        poll_interval_seconds=0.05,
        backoff=_FAST_BACKOFF,
    )
    try:
        assert len(created) == 1  # the common path spawns exactly once
    finally:
        handle.stop()


def test_startup_retries_transient_failure_then_succeeds(monkeypatch, tmp_path):
    created = _record_mkdtemp(monkeypatch)
    counter = tmp_path / "starts"
    handle = _start_vllm_native_server(
        model_name_or_path="fake-model",
        port=_free_port(),
        timeout_seconds=30,
        launcher=_FakeLauncher(_fail_then_serve_script(str(counter), fail_until=2)),
        max_attempts=3,
        poll_interval_seconds=0.05,
        backoff=_FAST_BACKOFF,
    )
    try:
        assert counter.read_text() == "3"  # failed twice, served on the third start
        assert len(created) == 3
        # Only the successful attempt's log dir survives; the two failed ones were cleaned up.
        assert sum(os.path.isdir(directory) for directory in created) == 1
    finally:
        handle.stop()


def test_startup_does_not_retry_permanent_failure(monkeypatch):
    created = _record_mkdtemp(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        _start_vllm_native_server(
            model_name_or_path="fake-model",
            port=_free_port(),
            timeout_seconds=30,
            launcher=_FakeLauncher(
                "import sys; print('RuntimeError: CUDA out of memory', file=sys.stderr); sys.exit(1)"
            ),
            max_attempts=3,
            poll_interval_seconds=0.05,
            backoff=_FAST_BACKOFF,
        )
    assert not isinstance(excinfo.value, TransientStartupError)
    assert "CUDA out of memory" in str(excinfo.value)
    assert len(created) == 1  # a non-streamer failure fails fast


def test_startup_exhausts_attempts_and_preserves_diagnostics(monkeypatch, tmp_path):
    created = _record_mkdtemp(monkeypatch)
    counter = tmp_path / "starts"
    with pytest.raises(TransientStartupError) as excinfo:
        _start_vllm_native_server(
            model_name_or_path="fake-model",
            port=_free_port(),
            timeout_seconds=30,
            launcher=_FakeLauncher(_fail_then_serve_script(str(counter), fail_until=99)),
            max_attempts=3,
            poll_interval_seconds=0.05,
            backoff=_FAST_BACKOFF,
        )
    assert counter.read_text() == "3"
    assert len(created) == 3
    err = excinfo.value
    # The streamer error and the attempted command survive in the final diagnostic, and the log dir
    # the message points at still exists on disk.
    assert "Could not receive runai_response from libstreamer" in str(err)
    assert "fake-model" in str(err)
    assert err.__notes__  # enriched with a summary note before re-raising
    assert sum(os.path.isdir(directory) for directory in created) == 1


def test_startup_hang_times_out_without_retry(monkeypatch):
    # A server that stays alive but never becomes ready is a TimeoutError, not a streamer fault, so it
    # must not be retried into 3x the wait.
    created = _record_mkdtemp(monkeypatch)
    with pytest.raises(TimeoutError):
        _start_vllm_native_server(
            model_name_or_path="fake-model",
            port=_free_port(),
            timeout_seconds=3,
            launcher=_FakeLauncher("import time; time.sleep(30)"),
            max_attempts=3,
            poll_interval_seconds=0.05,
            backoff=_FAST_BACKOFF,
        )
    assert len(created) == 1
