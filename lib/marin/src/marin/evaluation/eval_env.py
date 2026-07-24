# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Which launch-environment variables an eval job needs, and how they reach it.

An eval job (the group orchestrator, an evalchemy client, a Harbor runner) runs on a cluster worker
with none of the launcher's ambient credentials. The launcher forwards a fixed allowlist of variables
from its own environment into the job's :class:`EnvironmentSpec`; this module is the single definition
of that allowlist plus the one credential that needs renaming on the way in.

The Daytona SDK reads ``DAYTONA_API_KEY``, but the eval key is stored in Google Secret Manager (and so
in the launch environment) under ``DAYTONA_EVAL_API_KEY`` to keep it distinct from the data/RL keys.
:func:`daytona_sdk_env` bridges the two names so the control-plane credential is injected under the
name the SDK expects, rather than shipping the raw GSM name into every sandbox.
"""

import os

# Forwarded verbatim from the launch environment into an eval job (present ones only).
EVAL_ENV_KEYS: tuple[str, ...] = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "HF_TOKEN",
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

# The Google Secret Manager / launch-env name for the Daytona eval key.
DAYTONA_EVAL_API_KEY_ENV = "DAYTONA_EVAL_API_KEY"
# The name the Daytona SDK itself reads.
DAYTONA_SDK_API_KEY_ENV = "DAYTONA_API_KEY"


def env_vars_from_keys(keys: tuple[str, ...]) -> dict[str, str]:
    """The subset of ``keys`` present in the launch environment, as an env-var mapping."""
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def _read_gsm_secret(secret_id: str) -> str:
    """Read one approved launch secret from the active Google project.

    This remains a launch-host operation: the value is passed only through the
    orchestrator's environment under the Daytona SDK name and is never written
    into a config, record, or log.
    """
    import google.auth  # noqa: PLC0415 - only Harbor/Daytona launches need this optional dependency.
    from google.cloud import secretmanager  # noqa: PLC0415

    _, project_id = google.auth.default()
    if not project_id:
        raise RuntimeError("Google application-default credentials did not identify a project")
    response = secretmanager.SecretManagerServiceClient().access_secret_version(
        request={"name": f"projects/{project_id}/secrets/{secret_id}/versions/latest"}
    )
    return response.payload.data.decode("utf-8").strip()


def daytona_eval_api_key() -> str:
    """Resolve the Daytona eval key from the environment or its approved GSM secret."""
    key = os.environ.get(DAYTONA_EVAL_API_KEY_ENV)
    if key:
        return key
    try:
        key = _read_gsm_secret(DAYTONA_EVAL_API_KEY_ENV)
    except Exception as exc:
        raise RuntimeError(
            f"could not load {DAYTONA_EVAL_API_KEY_ENV} from Google Secret Manager; "
            "set the environment variable or authenticate the launch host for the approved secret"
        ) from exc
    if not key:
        raise RuntimeError(f"Google Secret Manager returned an empty {DAYTONA_EVAL_API_KEY_ENV} secret")
    return key


def daytona_sdk_env(*, required: bool = False) -> dict[str, str]:
    """Map the resolved eval credential to the ``DAYTONA_API_KEY`` the SDK reads.

    Evalchemy launches do not require Daytona and retain their no-secret path.
    A Harbor-on-Daytona launch passes ``required=True`` so it resolves the approved
    GSM secret automatically when the local environment has not already supplied it.
    """
    if required:
        return {DAYTONA_SDK_API_KEY_ENV: daytona_eval_api_key()}
    key = os.environ.get(DAYTONA_EVAL_API_KEY_ENV)
    return {DAYTONA_SDK_API_KEY_ENV: key} if key else {}
