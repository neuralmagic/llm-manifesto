"""Place TP/PP workers and DP ranks evenly across the nodes of a role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import RoleSpec


@dataclass(frozen=True)
class ParallelLayout:
    tp_world_size: int
    pp_world_size: int
    dp_world_size: int
    node_count: int
    gpus_per_node: int

    @property
    def gpus_per_dp_rank(self) -> int:
        return self.tp_world_size * self.pp_world_size

    @property
    def nodes_per_dp_rank(self) -> int:
        return max(1, self.gpus_per_dp_rank // self.gpus_per_node)

    @property
    def dp_local_size(self) -> int:
        """DP ranks with workers on each node, including ranks spanning nodes."""
        return max(1, self.dp_world_size // self.node_count)

    @property
    def tp_local_size(self) -> int:
        return min(self.tp_world_size, self.gpus_per_node)


def parallel_layout(role: "RoleSpec") -> ParallelLayout:
    parallelism = role.parallelism
    # gpus_per_pod validates that TP x PP x DP divides evenly across the nodes.
    layout = ParallelLayout(
        tp_world_size=parallelism.tp,
        pp_world_size=parallelism.pp,
        dp_world_size=parallelism.dp_size,
        node_count=role.lws.size,
        gpus_per_node=role.gpus_per_pod,
    )
    # Each node holds whole DP ranks, or each DP rank spans whole nodes.
    # Equal GPU totals alone do not guarantee this placement is possible.
    gpus = layout.gpus_per_node
    rank_gpus = layout.gpus_per_dp_rank
    shape = (
        f"tp={parallelism.tp}"
        if parallelism.pp == 1
        else f"tp={parallelism.tp} x pp={parallelism.pp} ({rank_gpus})"
    )
    if rank_gpus > gpus and rank_gpus % gpus:
        raise ValueError(
            f"{role.name}: {shape} is not divisible by {gpus} GPUs per pod"
        )
    if gpus > rank_gpus and gpus % rank_gpus:
        raise ValueError(
            f"{role.name}: {gpus} GPUs per pod is not divisible by "
            f"{rank_gpus} GPUs per DP rank ({shape})"
        )
    return layout
