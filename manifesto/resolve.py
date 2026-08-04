"""Resolve a spec role into concrete ports, paths, env vars, and vLLM arguments."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .cluster import Cluster
from .equations import render_mapping
from .features import FeatureContext, FeaturePlan, resolve_features
from .instance import Instance
from .dp_ports import RolePorts, derive_ports
from .parallelism import ParallelLayout, parallel_layout
from .spec import DeploymentSpec, RoleSpec, RoutingKind, TopologyKind


DEFAULT_VLLM_ARGS: dict[str, Any] = {
    "disable_access_log_for_endpoints": "/health,/v1/models,/metrics",
}


@dataclass(frozen=True)
class ResolvedRole:
    ports: RolePorts
    log_dir: str | None
    trace_dir: str | None
    vllm_env: str | None
    persistent_cache: bool
    fabric_profile: str
    env: dict[str, str]
    env_provenance: dict[str, str]
    vllm_args: dict[str, Any]
    features: FeaturePlan
    vllm_raw_args: list[str]
    resource_claims: list[dict[str, str]]


def resolve_role(spec: DeploymentSpec, instance: Instance, cluster: Cluster, role: RoleSpec) -> ResolvedRole:
    layout = parallel_layout(role)
    ports = derive_ports(
        rank_count=layout.dp_local_size,
        public_base=role.serving_port_base,
        backend_base=role.backend_port_base,
    )
    context = _variable_context(spec, role, layout)
    computed_env = render_mapping(role.computed.get("env", {}), context)
    context |= computed_env
    computed_vllm_args = render_mapping(role.computed.get("vllm", {}), context)
    vllm_args = DEFAULT_VLLM_ARGS | role.vllm_args | computed_vllm_args

    log_dir = (
        f"{cluster.log_root(user=instance.user_slug, release=instance.release_slug)}/{role.name}"
        if cluster.has_log_filesystem
        else None
    )
    trace_dir = None
    if role.profiling.enabled:
        if "profiler_config" in vllm_args:
            raise ValueError(
                f"{role.name}: set profiling: instead of a profiler_config vLLM arg"
            )
        default_trace_dir = f"{log_dir}/traces" if log_dir else None
        profiler_config = role.profiling.profiler_config(default_trace_dir)
        trace_dir = profiler_config.get("torch_profiler_dir")
        vllm_args = vllm_args | {"profiler_config": profiler_config}

    fabric_profile = role.fabric_profile or cluster.fabric_profile_for(
        topology=spec.topology.value,
        role_name=role.name,
        expert_parallel=role.parallelism.ep,
    )

    cache_prefix = None
    if cluster.has_cache_filesystem:
        cache_prefix = cluster.cache_root(
            user=instance.user_slug,
            release=instance.release_slug,
            gpu_arch=spec.accelerator_config(cluster).gpu_arch,
            cuda=spec.cache.cuda,
            cache_key=spec.cache_key,
        )
    vllm_env = spec.runtime.vllm_env
    if vllm_env is not None:
        _validate_vllm_env_path(vllm_env, cluster)
    env, env_provenance = _resolve_env(
        (
            (
                "manifesto",
                _base_env(
                    spec,
                    cache_prefix,
                    vllm_env=vllm_env,
                    platform=cluster.platform,
                ),
            ),
            (f"fabric:{fabric_profile}", cluster.fabric_env(fabric_profile, context)),
            ("runtime", spec.runtime.env),
            ("role", role.env),
            ("computed role", {key: str(value) for key, value in computed_env.items()}),
        )
    )
    features = resolve_features(
        FeatureContext(
            role_name=role.name,
            dp_enabled=role.parallelism.dp_enabled,
            expert_parallel=role.parallelism.ep,
            prefill_decode=spec.topology == TopologyKind.PD,
            llm_d_enabled=spec.routing.kind != RoutingKind.DISABLED,
            routing_proxy=role.routing_proxy,
            multi_node=role.lws.size > 1,
            kv_transfer_config=role.kv_transfer_config,
            all2all_backend=_optional_string(vllm_args.get("all2all_backend")),
            imex_resource_claim_template=cluster.fabric.imex_resource_claim_template,
            api_server_count=_api_server_count(vllm_args),
            explicit_env=frozenset(env),
        )
    )

    return ResolvedRole(
        ports=ports,
        log_dir=log_dir,
        trace_dir=trace_dir,
        vllm_env=vllm_env,
        persistent_cache=cache_prefix is not None,
        fabric_profile=fabric_profile,
        env=env,
        env_provenance=env_provenance,
        vllm_args=vllm_args,
        features=features,
        vllm_raw_args=[*role.vllm_raw_args],
        resource_claims=[
            {
                "name": claim.name,
                "resourceClaimTemplateName": claim.template,
            }
            for claim in features.resource_claims
        ],
    )


def _variable_context(spec: DeploymentSpec, role: RoleSpec, layout: ParallelLayout) -> dict[str, Any]:
    return {
        **spec.vars,
        **role.vars,
        "gpus_per_pod": role.gpus_per_pod,
        "tp": layout.tp_world_size,
        "tp_world_size": layout.tp_world_size,
        "tp_local_size": layout.tp_local_size,
        "dp_enabled": role.parallelism.dp_enabled,
        "dp_local_size": layout.dp_local_size,
        "dp_world_size": layout.dp_world_size,
        "lws_size": role.lws.size,
        "lws_replicas": role.lws.replicas,
    }


def _base_env(
    spec: DeploymentSpec,
    cache_prefix: str | None,
    *,
    vllm_env: str | None,
    platform: str,
) -> dict[str, str]:
    env = {
        "VLLM_NO_USAGE_STATS": "1",
        "TQDM_DISABLE": "1",
    }
    if spec.model.hf_home:
        env["HF_HOME"] = spec.model.hf_home
    if cache_prefix:
        env |= {
            "HOME": f"{cache_prefix}/home",
            "XDG_CACHE_HOME": f"{cache_prefix}/xdg",
            "VLLM_CACHE_ROOT": f"{cache_prefix}/vllm",
            "FLASHINFER_CACHE_DIR": f"{cache_prefix}/flashinfer",
            "FLASHINFER_WORKSPACE_BASE": f"{cache_prefix}/flashinfer-workspace",
            "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": f"{cache_prefix}/fa-cute-dsl",
            "TRITON_CACHE_DIR": f"{cache_prefix}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{cache_prefix}/torchinductor",
            "TILELANG_CACHE_DIR": f"{cache_prefix}/tilelang",
        }
    if vllm_env:
        env["MANIFESTO_VLLM_ENV"] = vllm_env
    if platform == "openshift":
        # OpenShift commonly assigns an arbitrary UID absent from /etc/passwd.
        # Python getpass (used by torch during import) honors USER first.
        env["USER"] = "vllm"
    return env


def _validate_vllm_env_path(vllm_env: str, cluster: Cluster) -> None:
    path = PurePosixPath(posixpath.normpath(vllm_env))
    if not path.is_absolute():
        raise ValueError("runtime.vllm_env must be an absolute path")
    mount_paths = [PurePosixPath(mount["mountPath"]) for mount in cluster.volume_mounts()]
    if not any(path == mount or path.is_relative_to(mount) for mount in mount_paths):
        rendered = ", ".join(str(mount) for mount in mount_paths)
        raise ValueError(
            f"runtime.vllm_env {vllm_env!r} is not covered by a model pod volume mount "
            f"({rendered})"
        )


def _resolve_env(
    layers: tuple[tuple[str, dict[str, Any]], ...],
) -> tuple[dict[str, str], dict[str, str]]:
    """Merge documented precedence layers while retaining useful provenance."""

    env: dict[str, str] = {}
    provenance: dict[str, str] = {}
    for origin, layer in layers:
        for key, value in layer.items():
            env[key] = str(value)
            provenance[key] = origin
    return env, provenance


def _api_server_count(vllm_args: dict[str, Any]) -> int:
    value = vllm_args.get("api_server_count", vllm_args.get("api-server-count", 1))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None
