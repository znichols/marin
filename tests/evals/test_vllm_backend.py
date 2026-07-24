# Copyright The Marin Authors

from marin.inference.vllm_backend import _served_model_name_args


def test_served_model_name_uses_the_model_path_by_default():
    assert _served_model_name_args("s3://bucket/model", ()) == ("--served-model-name", "s3://bucket/model")


def test_served_model_name_does_not_duplicate_an_explicit_canonical_name():
    assert _served_model_name_args("s3://bucket/model", ("--served-model-name", "canonical-model")) == ()
    assert _served_model_name_args("s3://bucket/model", ("--served-model-name=canonical-model",)) == ()
