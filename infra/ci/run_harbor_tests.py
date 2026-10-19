# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check Harbor's standalone lock and run its offline runtime tests."""

import subprocess
import sys
from pathlib import Path

HARBOR_TEST_PATTERNS = (
    "vendor/harbor/tests/unit/models",
    "vendor/harbor/tests/unit/test_job_*.py",
    "vendor/harbor/tests/unit/test_trial_*.py",
    "vendor/harbor/tests/unit/test_verifier*.py",
    "vendor/harbor/tests/unit/test_task_client.py",
    "vendor/harbor/tests/unit/test_archive.py",
    "vendor/harbor/tests/unit/test_benchmark_metadata.py",
    "vendor/harbor/tests/unit/test_environment_preflight.py",
    "vendor/harbor/tests/unit/test_runtime.py",
    "vendor/harbor/tests/unit/test_task_packager.py",
    "vendor/harbor/tests/unit/test_task_packager_multi_step.py",
    "vendor/harbor/tests/unit/test_telemetry_observer.py",
    "vendor/harbor/tests/unit/environments/test_daytona*.py",
)


def run_harbor_tests(repo_root: Path, extra_args: tuple[str, ...] = ()) -> int:
    """Run the same isolated Harbor checks locally and in CI."""
    lock = subprocess.run(["uv", "lock", "--project", "vendor/harbor", "--check"], cwd=repo_root, check=False)
    if lock.returncode:
        return lock.returncode
    test_paths = sorted({str(path) for pattern in HARBOR_TEST_PATTERNS for path in repo_root.glob(pattern)})
    result = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--project",
            "vendor/harbor",
            "--frozen",
            "--extra",
            "runtime",
            "--group",
            "test",
            "pytest",
            "-q",
            "-c",
            "vendor/harbor/pyproject.toml",
            *test_paths,
            *extra_args,
        ],
        cwd=repo_root,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(run_harbor_tests(Path(__file__).resolve().parents[2], tuple(sys.argv[1:])))
