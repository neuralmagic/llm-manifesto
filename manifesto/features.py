"""Resolve serving features and backend-specific runtime contributions.

Features describe user-visible serving behavior. Connectors, collective
implementations, workload kind, and platform resources are derived details and
remain visible in the resolved plan without being mislabeled as features.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable


class Feature(StrEnum):
    DATA_PARALLEL = "data-parallel"
    EXPERT_PARALLEL = "expert-parallel"
    PREFILL_DECODE = "prefill-decode"
    LLM_D = "llm-d"


class WorkloadKind(StrEnum):
    DEPLOYMENT = "Deployment"
    LEADER_WORKER_SET = "LeaderWorkerSet"


@dataclass(frozen=True)
class FeatureContract:
    """Dependencies and hard requirements for one serving feature."""

    implies: frozenset[Feature] = frozenset()
    requires: frozenset[Feature] = frozenset()


CONTRACTS: dict[Feature, FeatureContract] = {
    Feature.DATA_PARALLEL: FeatureContract(),
    Feature.EXPERT_PARALLEL: FeatureContract(),
    Feature.PREFILL_DECODE: FeatureContract(implies=frozenset({Feature.LLM_D})),
    Feature.LLM_D: FeatureContract(),
}


@dataclass(frozen=True)
class FeatureContext:
    role_name: str
    dp_enabled: bool
    expert_parallel: bool
    prefill_decode: bool
    llm_d_enabled: bool
    routing_proxy: bool
    multi_node: bool
    kv_transfer_config: dict[str, Any] | None
    all2all_backend: str | None
    imex_resource_claim_template: str | None
    api_server_count: int
    explicit_env: frozenset[str]


@dataclass(frozen=True)
class FieldRefEnv:
    name: str
    field_path: str
    source: str


@dataclass(frozen=True)
class ResourceClaim:
    name: str
    template: str
    source: str


class _Contributions:
    """Collect derived output with deterministic collision detection."""

    def __init__(self) -> None:
        self.field_ref_env: dict[str, FieldRefEnv] = {}
        self.resource_claims: dict[str, ResourceClaim] = {}

    def add_field_ref_env(self, source: str, name: str, field_path: str) -> None:
        contribution = FieldRefEnv(name=name, field_path=field_path, source=source)
        self._add_unique(self.field_ref_env, name, contribution)

    def add_resource_claim(self, source: str, name: str, template: str) -> None:
        contribution = ResourceClaim(name=name, template=template, source=source)
        self._add_unique(self.resource_claims, name, contribution)

    @staticmethod
    def _add_unique(target: dict[str, Any], name: str, contribution: Any) -> None:
        previous = target.get(name)
        if previous is not None and previous != contribution:
            raise ValueError(
                f"runtime contribution conflict for {name}: "
                f"{previous.source} and {contribution.source}"
            )
        target[name] = contribution


@dataclass(frozen=True)
class FeaturePlan:
    enabled: frozenset[Feature]
    backends: frozenset[str]
    workload_kind: WorkloadKind
    external_dp: bool
    routing_proxy: bool
    field_ref_env: tuple[FieldRefEnv, ...]
    resource_claims: tuple[ResourceClaim, ...]

    def has(self, feature: Feature) -> bool:
        return feature in self.enabled


Detector = Callable[[FeatureContext], bool]


DETECTORS: dict[Feature, Detector] = {
    Feature.DATA_PARALLEL: lambda ctx: ctx.dp_enabled,
    Feature.EXPERT_PARALLEL: lambda ctx: ctx.expert_parallel,
    Feature.PREFILL_DECODE: lambda ctx: ctx.prefill_decode,
    Feature.LLM_D: lambda ctx: ctx.llm_d_enabled,
}


def resolve_features(context: FeatureContext) -> FeaturePlan:
    """Resolve serving features, then derive deployment and backend details."""

    validate_contracts(CONTRACTS)
    enabled = {feature for feature, detect in DETECTORS.items() if detect(context)}
    changed = True
    while changed:
        changed = False
        for feature in tuple(enabled):
            additions = CONTRACTS[feature].implies - enabled
            if additions:
                enabled.update(additions)
                changed = True

    for feature in sorted(enabled, key=str):
        missing = CONTRACTS[feature].requires - enabled
        if missing:
            names = ", ".join(sorted(str(item) for item in missing))
            raise ValueError(f"{context.role_name}: feature {feature} requires {names}")

    external_dp = {
        Feature.DATA_PARALLEL,
        Feature.LLM_D,
    }.issubset(enabled)
    if external_dp and context.api_server_count > 1:
        raise ValueError(
            f"{context.role_name}: llm-d external DP is incompatible with "
            "api_server_count > 1"
        )

    connectors = connector_backends(context.kv_transfer_config)
    backends = {f"connector:{connector}" for connector in connectors}
    if context.all2all_backend:
        backends.add(f"all2all:{context.all2all_backend}")

    contributions = _Contributions()
    for connector in sorted(connectors):
        contributor = CONNECTOR_CONTRIBUTORS.get(connector.casefold())
        if contributor:
            contributor(context, connector, contributions)
    if context.imex_resource_claim_template:
        contributions.add_resource_claim(
            "platform:imex",
            "compute-domain-channel",
            context.imex_resource_claim_template,
        )

    return FeaturePlan(
        enabled=frozenset(enabled),
        backends=frozenset(backends),
        workload_kind=(
            WorkloadKind.LEADER_WORKER_SET
            if context.multi_node
            else WorkloadKind.DEPLOYMENT
        ),
        external_dp=external_dp,
        routing_proxy=context.routing_proxy,
        field_ref_env=tuple(contributions.field_ref_env.values()),
        resource_claims=tuple(contributions.resource_claims.values()),
    )


ConnectorContributor = Callable[[FeatureContext, str, _Contributions], None]


def _contribute_nixl(
    context: FeatureContext,
    connector: str,
    contributions: _Contributions,
) -> None:
    if "VLLM_NIXL_SIDE_CHANNEL_HOST" not in context.explicit_env:
        contributions.add_field_ref_env(
            f"backend:{connector}",
            "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "status.podIP",
        )


CONNECTOR_CONTRIBUTORS: dict[str, ConnectorContributor] = {
    "nixlconnector": _contribute_nixl,
}


def validate_contracts(contracts: dict[Feature, FeatureContract]) -> None:
    """Fail early when the static feature graph is incomplete or cyclic."""

    missing = set(Feature) - contracts.keys()
    referenced = {
        dependency
        for contract in contracts.values()
        for dependency in contract.implies | contract.requires
    }
    missing.update(referenced - contracts.keys())
    if missing:
        names = ", ".join(sorted(str(item) for item in missing))
        raise ValueError(f"feature contracts missing definitions for: {names}")

    visiting: list[Feature] = []
    visited: set[Feature] = set()

    def visit(feature: Feature) -> None:
        if feature in visiting:
            cycle = visiting[visiting.index(feature) :] + [feature]
            raise ValueError(
                "feature implication cycle: " + " -> ".join(str(item) for item in cycle)
            )
        if feature in visited:
            return
        visiting.append(feature)
        for dependency in contracts[feature].implies:
            visit(dependency)
        visiting.pop()
        visited.add(feature)

    for feature in contracts:
        visit(feature)


def connector_backends(value: Any) -> frozenset[str]:
    """Collect every connector named by a nested KV-transfer configuration."""

    connectors: set[str] = set()
    if isinstance(value, dict):
        connector = value.get("kv_connector")
        if isinstance(connector, str):
            connectors.add(connector)
        for item in value.values():
            connectors.update(connector_backends(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            connectors.update(connector_backends(item))
    return frozenset(connectors)
