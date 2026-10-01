"""Cluster-owned Slurm allocation and container settings."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SlurmAllocationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # The count is derived from the model's parallel layout.
    gres: str = "gpu"

    @field_validator("gres")
    @classmethod
    def validate_gres(cls, value: str) -> str:
        if not re.fullmatch(r"gpu(?::[A-Za-z][A-Za-z0-9_.-]*)?", value):
            raise ValueError("Slurm gres must be 'gpu' or 'gpu:TYPE', without a count")
        return value


class SlurmBind(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    read_only: bool = False

    @field_validator("source", "target")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not value.startswith("/") or any(c in value for c in ":,\n\r\0"):
            raise ValueError("bind paths must be absolute and contain no colon, comma or control characters")
        return value


class SlurmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    time: str = "01:00:00"
    exclusive: bool = True
    runtime: Literal["apptainer", "singularity", "pyxis", "native"] = "apptainer"
    binds: list[SlurmBind] = Field(default_factory=list)
    ssh_host: str | None = None
    setup: list[str] = Field(default_factory=list)

    @field_validator("ssh_host")
    @classmethod
    def ssh_destination(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", value):
            raise ValueError("ssh_host must be a hostname, SSH alias, or user@host")
        return value

    @field_validator("partition", "account", "qos", "constraint", "time")
    @classmethod
    def directive_value(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_.,:&|*+\[\]/=-]+", value):
            raise ValueError("Slurm directive values must be nonempty single tokens")
        return value

    @field_validator("time")
    @classmethod
    def time_limit(cls, value: str) -> str:
        if not re.fullmatch(r"(?:[0-9]+-)?[0-9]+(?::[0-9]{1,2}){0,2}", value):
            raise ValueError("Slurm time must use minutes, HH:MM:SS or D-HH:MM:SS")
        return value
