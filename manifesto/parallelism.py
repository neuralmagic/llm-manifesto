"""Derive and validate the local/global TP/DP layout of a role.

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
    tp_local_size: int
    dp_local_size: int
    dp_world_size: int

    @property
    def cross_node_tp(self) -> bool:
        return self.tp_world_size > self.tp_local_size

    @property
    def distributed_dp(self) -> bool:
        """The DP world contains TP groups that each span multiple nodes."""
        return self.cross_node_tp and self.dp_world_size > 1

    @property
    def tp_node_count(self) -> int:
        return self.tp_world_size // self.tp_local_size

    @property
    def serving_worker_indices(self) -> tuple[int, ...]:
        """LWS workers that host an API server for cross-node TP groups."""
        return tuple(
            range(0, self.tp_node_count * self.dp_world_size, self.tp_node_count)
        )


def parallel_layout(role: "RoleSpec") -> ParallelLayout:
    parallelism = role.parallelism
    gpus = role.gpus_per_pod
    nodes = role.lws.size

    if parallelism.tp <= gpus:
        tp_local = parallelism.tp
    else:
        if parallelism.tp % gpus:
            raise ValueError(
                f"{role.name}: tp={parallelism.tp} is not divisible by {gpus} GPUs per pod"
            )
        tp_local = gpus
        tp_nodes = parallelism.tp // tp_local
        required_nodes = tp_nodes * parallelism.dp_size
        if nodes != required_nodes:
            raise ValueError(
                f"{role.name}: tp={parallelism.tp} with dp={parallelism.dp_size} needs "
                f"lws.size={required_nodes} ({tp_nodes} nodes per TP group), got {nodes}"
            )
    if gpus % tp_local:
        raise ValueError(f"{role.name}: {gpus} GPUs per pod is not divisible by local TP {tp_local}")

    if parallelism.dp_enabled:
        if parallelism.tp > tp_local:
            dp_local = 1
        elif parallelism.dp_size % nodes:
            raise ValueError(
                f"{role.name}: dp={parallelism.dp_size} does not divide evenly across {nodes} LWS nodes"
            )
        else:
            dp_local = parallelism.dp_size // nodes
        if tp_local * dp_local != gpus:
            raise ValueError(
                f"{role.name}: {dp_local} local DP ranks x local TP {tp_local} "
                f"needs {dp_local * tp_local} GPUs per pod, got {gpus}"
            )
        dp_world = parallelism.dp_size
    else:
        dp_local = 1
        dp_world = 1
        if gpus != tp_local:
            raise ValueError(
                f"{role.name}: DP is disabled but local TP {tp_local} leaves "
                f"{gpus - tp_local} of {gpus} GPUs idle"
            )

    return ParallelLayout(
        tp_world_size=parallelism.tp,
        tp_local_size=tp_local,
        dp_local_size=dp_local,
        dp_world_size=dp_world,
    )
