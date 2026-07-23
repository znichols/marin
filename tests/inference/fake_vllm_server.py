# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stand-in for the real ``vllm serve`` child, driven by test_vllm_server.py to exercise
``_start_vllm_native_server``'s startup-retry path without a GPU or a real vLLM.

``_FakeLauncher`` passes one mode-argument group first; ``_start_vllm_native_server`` then appends the
real ``serve <model> ... --port <p>`` arguments, so ``--port`` is read from argv and the rest ignored.
Each mode records its start in the counter file, letting a test assert how many attempts actually ran.

Modes:
  serve                          Answer /v1/models with 200 (a ready server).
  hang <counter>                 Record the start, then sleep without ever becoming ready.
  fail <counter> <n> [message]   Record the start; for the first <n> starts print <message> (default:
                                 the libstreamer read fault) to stderr and exit 1, then serve.
"""

import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STREAMER_FAULT = "ValueError: Could not receive runai_response from libstreamer due to: b'File access error'"


def _record_start(counter: str) -> int:
    path = Path(counter)
    n = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(n))
    return n


def _serve() -> None:
    port = int(sys.argv[sys.argv.index("--port") + 1])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "fake-model"}]}')

        def log_message(self, *args) -> None:
            pass

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    mode = sys.argv[1]
    if mode == "serve":
        _serve()
    elif mode == "hang":
        _record_start(sys.argv[2])
        time.sleep(30)
    elif mode == "fail":
        started = _record_start(sys.argv[2])
        fail_until = int(sys.argv[3])
        # A custom message is present only when arg 4 is not the appended ``serve`` subcommand.
        message = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "serve" else STREAMER_FAULT
        if started <= fail_until:
            print(message, file=sys.stderr)
            sys.exit(1)
        _serve()
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
