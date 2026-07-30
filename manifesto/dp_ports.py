"""Derive public/backend serving ports from the local data-parallel layout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolePorts:
    public: list[int]
    backend: list[int]

    @property
    def rank_count(self) -> int:
        return len(self.public)


def derive_ports(*, rank_count: int, public_base: int = 8000, backend_base: int | None = None) -> RolePorts:
    if rank_count < 1:
        raise ValueError("rank_count must be >= 1")
    backend_start = backend_base if backend_base is not None else public_base
    return RolePorts(
        public=[public_base + offset for offset in range(rank_count)],
        backend=[backend_start + offset for offset in range(rank_count)],
    )
