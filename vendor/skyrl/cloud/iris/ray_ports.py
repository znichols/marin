# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray endpoints allocated by Iris for each task on a shared host."""

import hashlib
import os
import socket
from enum import StrEnum

class RayPortAllocation(StrEnum):
    IRIS = "iris"
    RANK_PARTITIONED_SMOKE = "rank_partitioned_smoke"


class RayPort(StrEnum):
    HEAD = "ray_head"
    DASHBOARD = "ray_dashboard"
    CLIENT = "ray_client"
    METRICS = "ray_metrics"
    RUNTIME_ENV_AGENT = "ray_runtime_env_agent"
    DASHBOARD_AGENT_GRPC = "ray_dashboard_agent_grpc"
    DASHBOARD_AGENT_LISTEN = "ray_dashboard_agent_listen"
    NODE_MANAGER = "ray_node_manager"
    OBJECT_MANAGER = "ray_object_manager"


def allocated_ray_port(port: RayPort) -> int:
    """Return the task's required scheduler allocation for a Ray endpoint."""
    return int(os.environ[f"IRIS_PORT_{port.value.upper()}"])


def _smoke_rank_port_range() -> range:
    """Disjoint per-rank slice for a bounded smoke, without controller changes."""
    task_id = os.environ["IRIS_TASK_ID"]
    job_id, rank_text = task_id.rsplit("/", 1)
    rank = int(rank_text.split(":", 1)[0])
    num_tasks = int(os.environ["IRIS_NUM_TASKS"])
    block_size = 32
    slots = 12_000 // (block_size * num_tasks)
    if not 1 < num_tasks <= 16 or not 0 <= rank < num_tasks or slots < 1:
        raise ValueError("rank-partitioned Ray smoke ports require 2–16 task ranks")
    slot = int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:8], "big") % slots
    start = 20_000 + (slot * num_tasks + rank) * block_size
    return range(start, start + block_size)


def _free_port_in_block(candidates: range, used: set[int]) -> int:
    for candidate in candidates:
        if candidate in used:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("", candidate))
            except OSError:
                continue
        return candidate
    raise RuntimeError("No free Ray ports remain in this smoke rank's block")


def resolve_ray_ports(allocation: RayPortAllocation = RayPortAllocation.IRIS) -> None:
    """Resolve Iris's task-allocated port placeholders before Ray bootstrap."""
    ports = {port: allocated_ray_port(port) for port in RayPort}
    if allocation is RayPortAllocation.RANK_PARTITIONED_SMOKE:
        if all(ports.values()):
            return  # A deployed Iris controller already reserved these ports.
        if any(ports.values()):
            raise ValueError("Smoke port allocation cannot mix Iris ports with zero placeholders")
        candidates = _smoke_rank_port_range()
        used: set[int] = set()
        for port in RayPort:
            value = _free_port_in_block(candidates, used)
            used.add(value)
            os.environ[f"IRIS_PORT_{port.value.upper()}"] = str(value)
        return
    if allocation is not RayPortAllocation.IRIS:
        raise ValueError(f"Unsupported Ray port allocation: {allocation}")
    # Import only when needed: Iris platform types pull in torch, while the
    # launch CLI must be able to load configuration without a training stack.
    from iris.cluster.platforms.types import find_free_port

    reserved = set(ports.values()) - {0}
    for port, value in ports.items():
        if value:
            continue
        # Kubernetes supplies zero placeholders; worker-backed tasks supply
        # concrete allocations. Use the same kernel selection as Iris's JAX launcher.
        value = find_free_port()
        while value in reserved:
            value = find_free_port()
        reserved.add(value)
        os.environ[f"IRIS_PORT_{port.value.upper()}"] = str(value)
