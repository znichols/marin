# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors

from marin.evaluation import eval_env


def test_daytona_sdk_env_prefers_an_explicit_launch_environment(monkeypatch):
    monkeypatch.setenv(eval_env.DAYTONA_EVAL_API_KEY_ENV, "env-key")
    monkeypatch.setattr(eval_env, "_read_gsm_secret", lambda _: (_ for _ in ()).throw(AssertionError("unexpected")))

    assert eval_env.daytona_sdk_env(required=True) == {eval_env.DAYTONA_SDK_API_KEY_ENV: "env-key"}


def test_daytona_sdk_env_resolves_the_approved_gsm_secret_for_harbor(monkeypatch):
    monkeypatch.delenv(eval_env.DAYTONA_EVAL_API_KEY_ENV, raising=False)
    monkeypatch.setattr(eval_env, "_read_gsm_secret", lambda secret_id: "gsm-key" if secret_id else "")

    assert eval_env.daytona_sdk_env(required=True) == {eval_env.DAYTONA_SDK_API_KEY_ENV: "gsm-key"}


def test_evalchemy_path_never_loads_a_daytona_secret(monkeypatch):
    monkeypatch.delenv(eval_env.DAYTONA_EVAL_API_KEY_ENV, raising=False)
    monkeypatch.setattr(eval_env, "_read_gsm_secret", lambda _: (_ for _ in ()).throw(AssertionError("unexpected")))

    assert eval_env.daytona_sdk_env() == {}
