"""Cluster profile schema and concrete cluster facts used while rendering pods.

The pydantic models mirror the sections of a cluster YAML profile. Unknown
keys are rejected so a typo'd profile fails at load instead of silently
falling back to defaults.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .equations import render_mapping
from .images import DEFAULT_IMAGES


class PersistentVolumeClaimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage_class_name: str | None = None
    access_modes: list[str]
    size: str


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shared_volume: dict[str, Any] | None = None
    shared_claim: PersistentVolumeClaimConfig | None = None
    shared_mount_path: str = "/mnt/shared"
    local_nvme_path: str | None = None

    @model_validator(mode="after")
    def validate_shared_claim_volume(self) -> "StorageConfig":
        if self.shared_claim is None:
            return self
        volume = self.shared_volume or {}
        claim_name = volume.get("persistentVolumeClaim", {}).get("claimName")
        if not claim_name:
            raise ValueError(
                "storage.shared_claim requires "
                "storage.shared_volume.persistentVolumeClaim.claimName"
            )
        return self

    @property
    def shared_claim_name(self) -> str | None:
        volume = self.shared_volume or {}
        return volume.get("persistentVolumeClaim", {}).get("claimName")


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_root: str | None = None
    log_root: str | None = None
    cache_root: str | None = None


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_host_path: str | None = None
    jit_host_path: str | None = None
    hf_home: str | None = None

    @model_validator(mode="after")
    def host_cache_paths_must_be_paired(self) -> "CacheConfig":
        if bool(self.hf_host_path) != bool(self.jit_host_path):
            raise ValueError("cache.hf_host_path and cache.jit_host_path must be configured together")
        return self


class RdmaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_name: str | None = None
    value: str = "1"

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, value: Any) -> str:
        return str(value)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pvc: str | None = None
    mount_path: str = "/mnt/logs"
    root: str | None = None


class PodDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shm_size: str = "2Gi"
    dns_policy: Literal["ClusterFirst", "Default", "ClusterFirstWithHostNet", "None"] | None = None
    dns_config: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    affinity: dict[str, Any] = Field(default_factory=dict)
    tolerations: list[dict[str, Any]] = Field(default_factory=list)
    extra_volumes: list[dict[str, Any]] = Field(default_factory=list)
    extra_volume_mounts: list[dict[str, Any]] = Field(default_factory=list)
    container_security_context: dict[str, Any] | None = None
    image_pull_secrets: list[str] = Field(default_factory=list)
    image_pull_policy: Literal["Always", "IfNotPresent", "Never"] | None = None
    termination_grace_period_seconds: int | None = Field(None, ge=0)
    working_dir: str | None = None

    @field_validator("image_pull_secrets")
    @classmethod
    def require_secret_names(cls, value: list[str]) -> list[str]:
        for name in value:
            if len(name) > 253 or not re.fullmatch(
                r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
                r"(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*",
                name,
            ) or any(len(label) > 63 for label in name.split(".")):
                raise ValueError(
                    "image_pull_secrets entries must be Kubernetes Secret names"
                )
        return value

    def image_pull_secret_refs(self) -> list[dict[str, str]]:
        return [{"name": name} for name in self.image_pull_secrets]


class AcceleratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_name: str
    presence_label: str
    node_selector: dict[str, str] = Field(default_factory=dict)
    gpu_arch: str
    torch_cuda_arch_list: str

    @field_validator("resource_name")
    @classmethod
    def require_extended_resource_name(cls, value: str) -> str:
        if not re.fullmatch(
            r"[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?/[A-Za-z0-9]"
            r"(?:[-A-Za-z0-9_.]*[A-Za-z0-9])?",
            value,
        ):
            raise ValueError(
                "accelerator resource_name must be a Kubernetes extended "
                f"resource such as 'nvidia.com/gpu', got {value!r}"
            )
        return value


class AcceleratorsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default: str
    profiles: dict[str, AcceleratorConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def default_must_exist(self) -> "AcceleratorsConfig":
        if self.default not in self.profiles:
            raise ValueError(f"default accelerator is not defined: {self.default}")
        return self

    def get(self, name: str | None = None) -> AcceleratorConfig:
        selected = name or self.default
        try:
            return self.profiles[selected]
        except KeyError as exc:
            choices = ", ".join(sorted(self.profiles))
            raise ValueError(
                f"unknown accelerator {selected!r} for this cluster; choose one of: {choices}"
            ) from exc


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: str = "istio"
    service_type: str = "ClusterIP"


class NamingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_prefix: bool = False


class FabricProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: dict[str, Any] = Field(default_factory=dict)
    computed_env: dict[str, Any] = Field(default_factory=dict)


class FabricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ucx_net_devices: str
    default_profile: str = "standard"
    expert_parallel_profiles: dict[str, str] = Field(default_factory=dict)
    default_env: dict[str, Any] = Field(default_factory=dict)
    profiles: dict[str, FabricProfileConfig] = Field(default_factory=dict)
    imex_resource_claim_template: str | None = None

    @model_validator(mode="after")
    def referenced_profiles_must_exist(self) -> "FabricConfig":
        referenced = {self.default_profile, *self.expert_parallel_profiles.values()}
        missing = referenced - self.profiles.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"undefined fabric profiles: {names}")
        return self

    def profile(self, name: str) -> FabricProfileConfig:
        try:
            return self.profiles[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self.profiles))
            raise ValueError(
                f"unknown fabric profile {name!r}; choose one of: {choices}"
            ) from exc


class LlmdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release: str | None = None
    images: dict[str, str] = Field(default_factory=dict)

    @property
    def resolved_release(self) -> str:
        return self.release or DEFAULT_IMAGES.get("llm_d.release")

    @property
    def epp(self) -> str:
        return self._image("epp")

    @property
    def envoy(self) -> str:
        return self._image("envoy")

    @property
    def routing_sidecar(self) -> str:
        return self._image("routing_sidecar")

    def _image(self, name: str) -> str:
        release = self.resolved_release
        template = self.images.get(name, DEFAULT_IMAGES.get(f"llm_d.{name}", release=release))
        return template.format(release=release)


class OpenShiftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scc: str | None = None


class KueueConfig(BaseModel):
    """Optional LocalQueue used to admit GPU serving workloads."""

    model_config = ConfigDict(extra="forbid")

    local_queue: str | None = None

    @field_validator("local_queue")
    @classmethod
    def require_label_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) > 63 or not re.fullmatch(
            r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?",
            value,
        ):
            raise ValueError(
                "kueue.local_queue must be a lowercase DNS-1123 label "
                "with at most 63 characters"
            )
        return value


class Cluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    gpus_per_node: int = Field(ge=1)
    accelerators: AcceleratorsConfig
    platform: Literal["kubernetes", "openshift"] = "kubernetes"
    naming: NamingConfig = Field(default_factory=NamingConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rdma: RdmaConfig = Field(default_factory=RdmaConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pod_defaults: PodDefaults = Field(default_factory=PodDefaults)
    fabric: FabricConfig
    llm_d: LlmdConfig = Field(default_factory=LlmdConfig)
    openshift: OpenShiftConfig = Field(default_factory=OpenShiftConfig)
    kueue: KueueConfig = Field(default_factory=KueueConfig)

    @model_validator(mode="after")
    def default_hf_home(self) -> "Cluster":
        if self.cache.hf_home is not None:
            return self
        if self.cache.hf_host_path and self.cache.jit_host_path:
            self.cache.hf_home = "/var/cache/huggingface"
        elif self.storage.local_nvme_path:
            self.cache.hf_home = "/mnt/local/hf_cache"
        elif self.storage.shared_volume:
            self.cache.hf_home = f"{self.storage.shared_mount_path}/hf_cache"
        return self

    # Path templates. Explicit profile values win; defaults derive from the
    # shared mount so they are declared exactly once.

    @property
    def user_root_template(self) -> str:
        return self.paths.user_root or f"{self.storage.shared_mount_path}/{{user}}"

    @property
    def log_root_template(self) -> str | None:
        explicit = self.logging.root or self.paths.log_root
        if explicit:
            return explicit
        if self.logging.pvc:
            return f"{self.logging.mount_path}/{{user}}/{{release}}"
        if self.storage.shared_volume:
            return f"{self.user_root_template}/logs"
        return None

    @property
    def cache_root_template(self) -> str | None:
        if self.paths.cache_root:
            return self.paths.cache_root
        if self.cache.jit_host_path:
            return "/var/cache/vllm/{user}/{gpu_arch}/{cuda}/{cache_key}/{release}"
        if self.storage.shared_volume:
            return f"{self.storage.shared_mount_path}/{{user}}/jit-cache/{{gpu_arch}}/{{cuda}}/{{cache_key}}/{{release}}"
        if self.storage.local_nvme_path:
            return f"/mnt/local/jit-cache/{{gpu_arch}}/{{cuda}}/{{cache_key}}/{{release}}"
        return None

    def user_root(self, *, user: str, release: str) -> str:
        return self.user_root_template.format(user=user, release=release)

    def log_root(self, *, user: str, release: str) -> str:
        if self.log_root_template is None:
            raise ValueError("persistent logging requires a configured filesystem")
        return self.log_root_template.format(user=user, release=release)

    def cache_root(self, *, user: str, release: str, gpu_arch: str, cuda: str, cache_key: str) -> str:
        if self.cache_root_template is None:
            raise ValueError("persistent caches require a configured filesystem")
        return self.cache_root_template.format(
            user=user, release=release, gpu_arch=gpu_arch, cuda=cuda, cache_key=cache_key
        )

    @property
    def has_cache_filesystem(self) -> bool:
        return self.cache_root_template is not None

    @property
    def has_log_filesystem(self) -> bool:
        return self.log_root_template is not None

    def with_path_overrides(
        self,
        *,
        user_root: str | None = None,
        log_root: str | None = None,
        cache_root: str | None = None,
    ) -> "Cluster":
        cluster = self.model_copy(deep=True)
        if user_root:
            cluster.paths.user_root = user_root
        if log_root:
            cluster.logging.root = log_root
        if cache_root:
            cluster.paths.cache_root = cache_root
        return cluster

    def base_volumes(self) -> list[dict]:
        volumes = [
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": self.pod_defaults.shm_size}},
        ]
        if self.cache.hf_host_path and self.cache.jit_host_path:
            volumes.extend(
                [
                    {
                        "name": "hf-cache",
                        "hostPath": {"path": self.cache.hf_host_path, "type": "DirectoryOrCreate"},
                    },
                    {
                        "name": "jit-cache",
                        "hostPath": {"path": self.cache.jit_host_path, "type": "DirectoryOrCreate"},
                    },
                ]
            )
        else:
            if self.storage.shared_volume:
                volumes.append({"name": "shared-storage", **self.storage.shared_volume})
            if self.storage.local_nvme_path:
                volumes.append(
                    {
                        "name": "local-nvme",
                        "hostPath": {"path": self.storage.local_nvme_path, "type": "Directory"},
                    }
                )
        if self.logging.pvc and self.logging.mount_path not in self._base_mount_paths():
            volumes.append({"name": "logs", "persistentVolumeClaim": {"claimName": self.logging.pvc}})
        volumes.extend(self.pod_defaults.extra_volumes)
        return volumes

    def volume_mounts(self) -> list[dict]:
        mounts = self._base_volume_mounts()
        if self.logging.pvc and self.logging.mount_path not in self._base_mount_paths():
            mounts.append({"name": "logs", "mountPath": self.logging.mount_path})
        mounts.extend(self.pod_defaults.extra_volume_mounts)
        return mounts

    def _base_mount_paths(self) -> set[str]:
        return {mount["mountPath"] for mount in self._base_volume_mounts()}

    def _base_volume_mounts(self) -> list[dict]:
        if self.cache.hf_host_path and self.cache.jit_host_path:
            return [
                {"name": "dshm", "mountPath": "/dev/shm"},
                {"name": "hf-cache", "mountPath": "/var/cache/huggingface"},
                {"name": "jit-cache", "mountPath": "/var/cache/vllm"},
            ]
        mounts = [{"name": "dshm", "mountPath": "/dev/shm"}]
        if self.storage.shared_volume:
            mounts.append({"name": "shared-storage", "mountPath": self.storage.shared_mount_path})
        if self.storage.local_nvme_path:
            mounts.append({"name": "local-nvme", "mountPath": "/mnt/local"})
        return mounts

    def fabric_profile_for(self, *, topology: str, role_name: str, expert_parallel: bool) -> str:
        if not expert_parallel:
            return self.fabric.default_profile
        return self.fabric.expert_parallel_profiles.get(role_name, self.fabric.default_profile)

    def fabric_env(self, profile: str, context: dict | None = None) -> dict[str, str]:
        format_context = {"ucx_net_devices": self.fabric.ucx_net_devices}
        env = {key: str(value).format(**format_context) for key, value in self.fabric.default_env.items()}
        profile_config = self.fabric.profile(profile)
        env |= {key: str(value) for key, value in profile_config.env.items()}
        if profile_config.computed_env:
            env |= {
                key: str(value)
                for key, value in render_mapping(profile_config.computed_env, context or {}).items()
            }
        return env

def load_cluster(path: str | Path) -> Cluster:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Cluster.model_validate(data)
