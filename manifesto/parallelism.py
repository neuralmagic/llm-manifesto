"""Derive and validate the local/global TP/PP/DP layout of a role.

Impossible layouts are hard errors: the renderer never rounds a requested
parallel size to something that happens to fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import RoleSpec


@dataclass(frozen=True)
class ParallelLayout:
    tp_world_size: int
    pp_world_size: int
    tp_local_size: int
    model_parallel_local_size: int
    dp_local_size: int
    dp_world_size: int

    @property
    def model_parallel_world_size(self) -> int:
        return self.tp_world_size * self.pp_world_size

    @property
    def cross_node_tp(self) -> bool:
        return self.tp_world_size > self.tp_local_size

    @property
    def cross_node_model_parallel(self) -> bool:
        return self.model_parallel_world_size > self.model_parallel_local_size

    @property
    def distributed_dp(self) -> bool:
        """The DP world contains model-parallel groups spanning multiple nodes."""
        return self.cross_node_model_parallel and self.dp_world_size > 1

    @property
    def tp_node_count(self) -> int:
        return self.tp_world_size // self.tp_local_size

    @property
    def model_parallel_node_count(self) -> int:
        return self.model_parallel_world_size // self.model_parallel_local_size

    @property
    def serving_worker_indices(self) -> tuple[int, ...]:
        """LWS workers that host an API server for model-parallel groups."""
        return tuple(
            range(
                0,
                self.model_parallel_node_count * self.dp_world_size,
                self.model_parallel_node_count,
            )
        )


def parallel_layout(role: "RoleSpec") -> ParallelLayout:
    parallelism = role.parallelism
    gpus = role.gpus_per_pod
    nodes = role.lws.size
    model_parallel_size = parallelism.tp * parallelism.pp

    if model_parallel_size <= gpus:
        model_parallel_local = model_parallel_size
    else:
        if model_parallel_size % gpus:
            if parallelism.pp == 1:
                raise ValueError(
                    f"{role.name}: tp={parallelism.tp} is not divisible by "
                    f"{gpus} GPUs per pod"
                )
            raise ValueError(
                f"{role.name}: model parallel size tp={parallelism.tp} x "
                f"pp={parallelism.pp} ({model_parallel_size}) is not divisible by "
                f"{gpus} GPUs per pod"
            )
        model_parallel_local = gpus
        model_parallel_nodes = model_parallel_size // model_parallel_local
        required_nodes = model_parallel_nodes * parallelism.dp_size
        if nodes != required_nodes:
            group_description = (
                f"{model_parallel_nodes} nodes per TP group"
                if parallelism.pp == 1
                else f"{model_parallel_nodes} nodes per model-parallel group"
            )
            shape = (
                f"tp={parallelism.tp}"
                if parallelism.pp == 1
                else f"tp={parallelism.tp} with pp={parallelism.pp}"
            )
            raise ValueError(
                f"{role.name}: {shape} with dp={parallelism.dp_size} needs "
                f"lws.size={required_nodes} ({group_description}), got {nodes}"
            )
    if gpus % model_parallel_local:
        if parallelism.pp == 1:
            raise ValueError(
                f"{role.name}: {gpus} GPUs per pod is not divisible by local TP "
                f"{model_parallel_local}"
            )
        raise ValueError(
            f"{role.name}: {gpus} GPUs per pod is not divisible by local "
            f"model parallel size {model_parallel_local}"
        )

    # This remains useful to existing equations. A node can contain partial TP
    # groups when PP is also enabled, so model_parallel_local_size is the
    # authoritative local GPU count for launching and allocation.
    tp_local = min(parallelism.tp, model_parallel_local)

    if parallelism.dp_enabled:
        if model_parallel_size > model_parallel_local:
            dp_local = 1
        elif parallelism.dp_size % nodes:
            raise ValueError(
                f"{role.name}: dp={parallelism.dp_size} does not divide evenly across {nodes} LWS nodes"
            )
        else:
            dp_local = parallelism.dp_size // nodes
        if model_parallel_local * dp_local != gpus:
            local_description = (
                f"local TP {model_parallel_local}"
                if parallelism.pp == 1
                else f"local model parallel size {model_parallel_local}"
            )
            raise ValueError(
                f"{role.name}: {dp_local} local DP ranks x {local_description} "
                f"needs {dp_local * model_parallel_local} GPUs per pod, got {gpus}"
            )
        dp_world = parallelism.dp_size
    else:
        dp_local = 1
        dp_world = 1
        if gpus != model_parallel_local:
            local_description = (
                f"local TP {model_parallel_local}"
                if parallelism.pp == 1
                else f"local model parallel size {model_parallel_local}"
            )
            raise ValueError(
                f"{role.name}: DP is disabled but {local_description} leaves "
                f"{gpus - model_parallel_local} of {gpus} GPUs idle"
            )

    return ParallelLayout(
        tp_world_size=parallelism.tp,
        pp_world_size=parallelism.pp,
        tp_local_size=tp_local,
        model_parallel_local_size=model_parallel_local,
        dp_local_size=dp_local,
        dp_world_size=dp_world,
    )
