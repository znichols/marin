# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Credentials and launch-environment variables supplied to evaluation jobs.

An eval job (the group orchestrator, an evalchemy client, a Harbor runner) runs on a cluster worker
with none of the launcher's ambient credentials. Most values come from a fixed environment allowlist;
mechanism-specific helpers may also resolve approved secrets before submission.
"""

import os

from rigging.secrets import resolve_secret_spec

# Credentials and W&B metadata propagated from an eval orchestrator into its serve and mechanism children.
EVAL_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "HF_TOKEN",
)

# Forwarded verbatim from the launch environment into an eval job (present ones only).
EVAL_ENV_KEYS: tuple[str, ...] = (
    *EVAL_RUNTIME_ENV_KEYS,
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "E2B_API_KEY",
    "MODAL_API_KEY",
    "TPU_CI",
    "MARIN_PREFIX",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION",
    "VLLM_TPU_SKIP_PRECOMPILE",
)

DAYTONA_SDK_API_KEY_ENV = "DAYTONA_API_KEY"
DAYTONA_EVAL_API_KEY_ENV = "DAYTONA_EVAL_API_KEY"
DAYTONA_ENVIRONMENT_TYPE = "daytona"
_DAYTONA_GCP_SECRET = "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest"


def env_vars_from_keys(keys: tuple[str, ...]) -> dict[str, str]:
    """The subset of ``keys`` present in the launch environment, as an env-var mapping."""
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def daytona_sdk_env() -> dict[str, str]:
    """Resolve the approved eval credential under the name read by the Daytona SDK."""
    secret = resolve_secret_spec(
        (
            f"env:{DAYTONA_SDK_API_KEY_ENV}",
            f"env:{DAYTONA_EVAL_API_KEY_ENV}",
            _DAYTONA_GCP_SECRET,
        )
    )
    return {DAYTONA_SDK_API_KEY_ENV: secret.value}
