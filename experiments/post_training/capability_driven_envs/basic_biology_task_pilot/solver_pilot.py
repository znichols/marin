# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded OpenRouter solver-trial pilot for three private, provisional TaskSpecs.

Uses TaskForge's Stage 50 lowering and run_trials through the existing BioTasks
calibrator. These solver trials are diagnostic: controls, adversarial validation,
Stage 60 review, expert reference validation, and rights clearance are absent.
"""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import fsspec
from taskcompendium.models import TaskSpec
from taskforge.content_hash import digest

from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.finalize import read_json, write_new
from experiments.post_training.capability_driven_envs.biotasks_library.calibrate_glm_v1 import (
    CalibrationConfig,
    _calibrate,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import _key_budget

TASK_SNAPSHOT_SHA256 = "e6183697a80f32faade375d3a5b8d3896f161ec06b19470c8e3f10b9329dd960"
PILOT_IDS_SHA256 = "1976035f4576d9f91c4a54c3c39d94765eae78b2e0c6311b023a9c6645e97449"
REVISION = "gov-bio-stage50-openrouter-solver-pilot-v1"
MAX_PILOT_SPEND_USD = Decimal("2")
MIN_REMAINING_USD = Decimal("5")
MODELS = (
    ("openai/gpt-4.1-mini", "openai", "OpenAI", 1047576),
    ("google/gemini-2.5-flash", "google-ai-studio", "Google AI Studio", 1048576),
)


def load_pilot(tasks_uri: str, ids_uri: str) -> list[TaskSpec]:
    """Ensure both model routes see exactly the curated private snapshot tasks."""
    with fsspec.open(tasks_uri, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != TASK_SNAPSHOT_SHA256:
        raise ValueError("Wrong provisional TaskSpec snapshot")
    with fsspec.open(ids_uri, "rb") as stream:
        ids_raw = stream.read()
    if hashlib.sha256(ids_raw).hexdigest() != PILOT_IDS_SHA256:
        raise ValueError("Pilot identifier file changed")
    selection = json.loads(ids_raw)
    ids = selection["task_ids"]
    if len(ids) != 3 or len(set(ids)) != 3 or selection["status"] != "pilot_only_not_scientifically_validated":
        raise ValueError("Expected exactly three unique private pilot tasks")
    selected: dict[str, TaskSpec] = {}
    for line in raw.splitlines():
        task = TaskSpec.model_validate_json(line)
        if task.id in ids:
            if "not-training-ready" not in task.tags or "rights-unreviewed" not in task.tags:
                raise ValueError("Pilot task missing release blockers")
            selected[task.id] = task
    if set(selected) != set(ids):
        raise ValueError("Missing a selected frozen TaskSpec")
    return [selected[item_id] for item_id in ids]


async def run(tasks_uri: str, ids_uri: str, output_root: str) -> dict[str, Any]:
    tasks = load_pilot(tasks_uri, ids_uri)
    start_usage, remaining = _key_budget()
    if remaining < MIN_REMAINING_USD:
        raise ValueError("OpenRouter key has insufficient balance for the pilot")
    results = []
    for model, tag, provider, context_window in MODELS:
        config = CalibrationConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            model=model,
            provider_tag=tag,
            provider_name=provider,
            context_window=context_window,
            max_retries=0,
            campaign_id=REVISION,
        )
        for task in tasks:
            path = output_root.rstrip("/") + "/" + model.replace("/", "--") + "/" + task.id + ".json"
            fs, key = fsspec.core.url_to_fs(path)
            if fs.exists(key):
                saved, _ = read_json(path)
                if saved["frozen_task_digest"] != digest(task) or saved["model"] != model:
                    raise ValueError("Existing trial is not for the pinned task and model")
                results.append(saved)
                continue
            usage, remaining = _key_budget()
            if usage < start_usage or usage - start_usage >= MAX_PILOT_SPEND_USD or remaining < MIN_REMAINING_USD:
                raise ValueError("OpenRouter pilot budget gate reached")
            with tempfile.TemporaryDirectory(prefix="basic-biology-stage50-") as temporary:
                outcome = await _calibrate(task, config, Path(temporary))
            result = {
                **outcome,
                "campaign_format": "basic-biology-stage50-solver-pilot-v1",
                "task_family": "basic-biology-mcq",
            }
            if result["frozen_task_digest"] != digest(task) or result["model"] != model:
                raise ValueError("Solver result changed task or model identity")
            write_new(path, (json.dumps(result, sort_keys=True) + "\n").encode())
            print(
                json.dumps(
                    {
                        "task_id": task.id,
                        "model": model,
                        "trial": result["trial"],
                        "route_verified": result["route_verified"],
                    }
                ),
                flush=True,
            )
            results.append(result)
    statuses = [row["trial"]["status"] for row in results]
    return {
        "revision": REVISION,
        "task_snapshot_sha256": TASK_SNAPSHOT_SHA256,
        "pilot_selection_sha256": PILOT_IDS_SHA256,
        "expected_pairs": len(tasks) * len(MODELS),
        "recorded_pairs": len(results),
        "graded": statuses.count("graded"),
        "ungraded": statuses.count("ungraded"),
        "route_verified_pairs": sum(row["route_verified"] for row in results),
        "scientific_validation": False,
        "rights_cleared": False,
        "controls_or_adversaries_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY must be supplied through the job environment")
    print(json.dumps(asyncio.run(run(args.tasks, args.ids, args.output)), indent=2))


if __name__ == "__main__":
    main()
