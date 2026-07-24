# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from rigging.secrets import ResolvedSecret

from marin.evaluation import eval_env


def test_daytona_sdk_env_prefers_sdk_then_eval_env_before_secret_manager(monkeypatch):
    observed: list[tuple[str, ...]] = []

    def resolve(refs: tuple[str, ...]) -> ResolvedSecret:
        observed.append(refs)
        return ResolvedSecret(value="daytona-key", source=refs[1])

    monkeypatch.setattr(eval_env, "resolve_secret_spec", resolve)

    assert eval_env.daytona_sdk_env() == {"DAYTONA_API_KEY": "daytona-key"}
    assert observed == [
        (
            "env:DAYTONA_API_KEY",
            "env:DAYTONA_EVAL_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest",
        )
    ]
