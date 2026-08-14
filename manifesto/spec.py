"""Pydantic models and loader for user-authored deployment YAML specs."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .cluster import AcceleratorConfig, Cluster
from .images import apply_image_refs
from .overrides import load_spec_data
from .parallelism import parallel_layout


class TopologyKind(StrEnum):
    AGGREGATED = "aggregated"
    PD = "pd"


class RoutingKind(StrEnum):
    LOAD_AWARE = "load_aware"
    PD = "pd"
    DISABLED = "disabled"


class RoutingFrontend(StrEnum):
    STANDALONE = "standalone"
    GATEWAY = "gateway"


class DpLoadBalancing(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"


# GPU count assumed when a spec is loaded without a cluster profile to infer from.
DEFAULT_GPUS_PER_POD = 8
DEFAULT_CPU_BASE = 6
DEFAULT_CPU_PER_GPU = 2
DEFAULT_MEMORY_PER_GPU_GI = 64
DEFAULT_MINIMUM_MEMORY_GI = 128
_CONFIG_MAP_KEY_RE = re.compile(r"[A-Za-z0-9._-]{1,253}")


class LwsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: int = Field(1, ge=1)
    replicas: int = Field(1, ge=1)
    same_topology_key: str | None = None


class ProfilingSpec(BaseModel):
    """Role-level profiler settings mapped onto vLLM's --profiler-config.

    Field names mirror vLLM's ProfilerConfig so the mapping stays obvious.
    The trace directory is supplied by the renderer unless overridden here,
    which keeps user- and cluster-specific paths out of model specs.
    """

    model_config = ConfigDict(extra="forbid")

    profiler: Literal["torch", "cuda"] | None = None
    trace_dir: str | None = None
    ignore_frontend: bool | None = None
    torch_profiler_with_stack: bool | None = None
    torch_profiler_record_shapes: bool | None = None
    torch_profiler_with_memory: bool | None = None
    torch_profiler_with_flops: bool | None = None
    capture_torch_profiler: bool | None = None
    detailed_trace_annotation: bool | None = None
    delay_iterations: int | None = Field(None, ge=0)
    max_iterations: int | None = Field(None, ge=0)
    warmup_iterations: int | None = Field(None, ge=0)
    active_iterations: int | None = Field(None, ge=1)
    wait_iterations: int | None = Field(None, ge=0)

    @property
    def enabled(self) -> bool:
        return self.profiler is not None

    def profiler_config(self, trace_dir: str | None) -> dict[str, Any]:
        """Build the --profiler-config payload, omitting unset fields."""
        if not self.enabled:
            return {}
        config: dict[str, Any] = {"profiler": self.profiler}
        if self.profiler == "torch":
            resolved_dir = self.trace_dir or trace_dir
            if not resolved_dir:
                raise ValueError(
                    "torch profiling requires a log filesystem or an explicit "
                    "profiling.trace_dir"
                )
            config["torch_profiler_dir"] = resolved_dir
        elif self.trace_dir:
            raise ValueError("profiling.trace_dir only applies to the torch profiler")
        for name in (
            "ignore_frontend",
            "torch_profiler_with_stack",
            "torch_profiler_record_shapes",
            "torch_profiler_with_memory",
            "torch_profiler_with_flops",
            "capture_torch_profiler",
            "detailed_trace_annotation",
            "delay_iterations",
            "max_iterations",
            "warmup_iterations",
            "active_iterations",
            "wait_iterations",
        ):
            value = getattr(self, name)
            if value is not None:
                config[name] = value
        return config


class ParallelismSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tp: int = Field(1, ge=1)
    dp: int | bool | None = None
    ep: bool = False
    gpus: int | None = Field(None, ge=1, validation_alias=AliasChoices("gpus", "gpus_per_node"))

    @field_validator("dp")
    @classmethod
    def validate_dp(cls, value: int | bool | None) -> int | bool | None:
        if value is True:
            raise ValueError("dp: true is ambiguous; use a global size or false")
        if isinstance(value, int) and not isinstance(value, bool) and value < 1:
            raise ValueError("dp must be >= 1")
        return value

    @property
    def dp_size(self) -> int:
        """Requested global data-parallel size; 1 means data parallelism is off."""
        if self.dp is None or self.dp is False:
            return 1
        return self.dp

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1


class ResourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # These fallbacks match the built-in formulas for DEFAULT_GPUS_PER_POD.
    # Rendering recalculates omitted fields for the resolved per-pod GPU count.
    cpu: str = "22"
    memory: str = "512Gi"
    gpus: int = Field(DEFAULT_GPUS_PER_POD, ge=0)
    ephemeral_storage: str | None = None

    @field_validator("cpu", "memory", mode="before")
    @classmethod
    def coerce_quantities(cls, value: Any) -> str:
        return value if value is None else str(value)


class RoleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    workload_name: str | None = None
    lws: LwsSpec = Field(default_factory=LwsSpec)
    parallelism: ParallelismSpec = Field(default_factory=ParallelismSpec)
    profiling: ProfilingSpec = Field(default_factory=ProfilingSpec)
    serving_port_base: int = 8000
    backend_port_base: int | None = None
    routing_proxy: bool = False
    dp_load_balancing: DpLoadBalancing = DpLoadBalancing.INTERNAL
    kv_transfer_config: dict[str, Any] | None = None
    vllm_args: dict[str, Any] = Field(
        default_factory=dict, validation_alias=AliasChoices("vllm", "vllm_args")
    )
    vllm_raw_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    fabric_profile: str | None = None
    pre_launch: list[str] = Field(default_factory=list)
    vars: dict[str, Any] = Field(default_factory=dict)
    computed: dict[str, dict[str, Any]] = Field(default_factory=dict)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    shm_size: str | None = None

    @property
    def gpus_per_pod(self) -> int:
        if self.parallelism.gpus is not None:
            return self.parallelism.gpus
        return DEFAULT_GPUS_PER_POD

    @model_validator(mode="after")
    def default_backend_port(self) -> "RoleSpec":
        if self.routing_proxy and self.backend_port_base is None:
            self.backend_port_base = 8200
        return self


class ModelSpec(BaseModel):
    id: str
    label: str | None = None
    image: str
    served_name: str | None = None
    hf_home: str | None = None

    @property
    def label_value(self) -> str:
        return self.label or self.id.rsplit("/", 1)[-1]


class IdleShutdownSpec(BaseModel):
    """Scale an idle Manifesto instance to zero after a bounded timeout."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    timeout_minutes: int = Field(45, ge=1)


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vllm_env: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    pre_launch: list[str] = Field(default_factory=list)
    sidecars: list[str] = Field(default_factory=lambda: ["dcgm-exporter", "node-exporter"])
    idle_shutdown: IdleShutdownSpec = Field(default_factory=IdleShutdownSpec)


class EppSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str | None = None
    replicas: int = Field(1, ge=1)
    plugins_config_file: str = "plugins.yaml"
    plugin_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("plugins_config_file")
    @classmethod
    def require_config_map_key(cls, value: str) -> str:
        if not _CONFIG_MAP_KEY_RE.fullmatch(value):
            raise ValueError("plugins_config_file must be a ConfigMap key, not a path")
        return value

    @field_validator("plugin_configs")
    @classmethod
    def require_valid_config_map_keys(
        cls, value: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        invalid = [name for name in value if not _CONFIG_MAP_KEY_RE.fullmatch(name)]
        if invalid:
            raise ValueError(f"plugin_configs contains invalid ConfigMap key: {invalid[0]!r}")
        return value

    @model_validator(mode="after")
    def require_selected_config(self) -> "EppSpec":
        if self.plugins_config_file != "plugins.yaml" and not self.plugin_configs:
            raise ValueError("a custom plugins_config_file requires plugin_configs")
        if self.plugin_configs and self.plugins_config_file not in self.plugin_configs:
            raise ValueError(
                f"plugins_config_file {self.plugins_config_file!r} is not present in plugin_configs"
            )
        return self


class RoutingSpec(BaseModel):
    kind: RoutingKind | None = None
    frontend: RoutingFrontend = RoutingFrontend.STANDALONE
    epp_image: str | None = None
    plugin_config: dict[str, Any] | None = None
    replicas: int = Field(1, ge=1)
    target_role: str | None = None
    epp: EppSpec | None = None

    @model_validator(mode="after")
    def reject_mixed_epp_configuration(self) -> "RoutingSpec":
        if self.epp is None:
            return self
        legacy_fields = {"epp_image", "plugin_config", "replicas"} & self.model_fields_set
        if legacy_fields:
            fields = ", ".join(sorted(legacy_fields))
            raise ValueError(f"routing.epp cannot be combined with legacy routing fields: {fields}")
        return self


class CacheSpec(BaseModel):
    cuda: str = "cu13"
    key: str | None = None
    cleanup_on_crash: bool = True


class DeploymentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release: str
    namespace: str = "default"
    accelerator: str | None = None
    topology: TopologyKind
    model: ModelSpec
    roles: list[RoleSpec]
    routing: RoutingSpec = Field(default_factory=RoutingSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    cache: CacheSpec = Field(default_factory=CacheSpec)
    vars: dict[str, Any] = Field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        if self.cache.key:
            return _safe_cache_key(self.cache.key)
        image = self.model.image
        if "@" in image:
            identity = image.rsplit("@", 1)[1]
        else:
            image_name = image.rsplit("/", 1)[-1]
            identity = image_name.rsplit(":", 1)[1] if ":" in image_name else "latest"
        return _safe_cache_key(identity)

    def accelerator_config(self, cluster: Cluster) -> AcceleratorConfig:
        return cluster.accelerators.get(self.accelerator)

    @field_validator("roles")
    @classmethod
    def require_unique_roles(cls, roles: list[RoleSpec]) -> list[RoleSpec]:
        names = [role.name for role in roles]
        if len(names) != len(set(names)):
            raise ValueError("role names must be unique")
        return roles

    @model_validator(mode="after")
    def apply_topology_defaults(self) -> "DeploymentSpec":
        if self.routing.kind is None:
            self.routing.kind = RoutingKind.PD if self.topology == TopologyKind.PD else RoutingKind.LOAD_AWARE
        if self.routing.target_role is None:
            self.routing.target_role = "decode"
        if self.topology == TopologyKind.PD:
            if self.routing.kind != RoutingKind.PD:
                raise ValueError("pd topology requires routing.kind: pd")
            decode = self.role("decode")
            decode.routing_proxy = True
            decode.serving_port_base = 8000
            decode.backend_port_base = 8200
        llm_d_enabled = self.routing.kind != RoutingKind.DISABLED
        for role in self.roles:
            resolved_dp_mode = (
                DpLoadBalancing.EXTERNAL
                if role.parallelism.dp_enabled and llm_d_enabled
                else DpLoadBalancing.INTERNAL
            )
            if (
                "dp_load_balancing" in role.model_fields_set
                and role.dp_load_balancing != resolved_dp_mode
            ):
                raise ValueError(
                    f"{role.name}: dp_load_balancing is derived as {resolved_dp_mode}; "
                    "use routing.kind: disabled to select direct vLLM"
                )
            role.dp_load_balancing = resolved_dp_mode
            if role.routing_proxy and not llm_d_enabled:
                raise ValueError(f"{role.name}: routing_proxy requires llm-d routing")
            if role.dp_load_balancing == DpLoadBalancing.EXTERNAL and _api_server_count(role.vllm_args) > 1:
                raise ValueError(
                    f"{role.name}: api_server_count > 1 is incompatible with external DP load balancing"
                )
        return self

    def role(self, name: str) -> RoleSpec:
        for role in self.roles:
            if role.name == name:
                return role
        raise KeyError(f"unknown role: {name}")

    def apply_cluster_defaults(self, cluster: Cluster) -> None:
        cluster.accelerators.get(self.accelerator)
        if self.model.hf_home is None:
            self.model.hf_home = cluster.cache.hf_home
        for role in self.roles:
            if role.parallelism.gpus is None:
                role.parallelism.gpus = _infer_gpus_per_pod(role, cluster.gpus_per_node)
            if "gpus" not in role.resources.model_fields_set:
                role.resources.gpus = role.gpus_per_pod
            if "cpu" not in role.resources.model_fields_set:
                role.resources.cpu = str(
                    DEFAULT_CPU_BASE + DEFAULT_CPU_PER_GPU * role.gpus_per_pod
                )
            if "memory" not in role.resources.model_fields_set:
                memory_gi = max(
                    DEFAULT_MEMORY_PER_GPU_GI * role.gpus_per_pod,
                    DEFAULT_MINIMUM_MEMORY_GI,
                )
                role.resources.memory = f"{memory_gi}Gi"
            parallel_layout(role)


def _api_server_count(vllm_args: dict[str, Any]) -> int:
    value = vllm_args.get("api_server_count", vllm_args.get("api-server-count", 1))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _infer_gpus_per_pod(role: RoleSpec, _cluster_gpus_per_node: int) -> int:
    parallelism = role.parallelism
    ranks = parallelism.tp * parallelism.dp_size
    if ranks % role.lws.size:
        raise ValueError(
            f"{role.name}: tp={parallelism.tp} x dp={parallelism.dp_size} "
            f"does not divide evenly across lws.size={role.lws.size}"
        )
    return ranks // role.lws.size


def _safe_cache_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "latest"


def load_spec(
    path: str | Path,
    cluster: Cluster | None = None,
    *,
    accelerator: str | None = None,
) -> DeploymentSpec:
    data = load_spec_data(path)
    data = apply_image_refs(data)
    spec = DeploymentSpec.model_validate(data)
    if accelerator is not None:
        spec.accelerator = accelerator
    if cluster is not None:
        spec.apply_cluster_defaults(cluster)
    return spec
