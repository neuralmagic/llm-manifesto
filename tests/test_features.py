"""Feature and backend resolution tests independent of Kubernetes rendering."""

import pytest
from pydantic import ValidationError

from manifesto.cluster import FabricConfig
from manifesto.features import (
    CONTRACTS,
    Feature,
    FeatureContext,
    FeatureContract,
    WorkloadKind,
    resolve_features,
    validate_contracts,
)


def _context(**overrides) -> FeatureContext:
    values = {
        "role_name": "decode",
        "dp_enabled": False,
        "expert_parallel": False,
        "prefill_decode": False,
        "llm_d_enabled": False,
        "routing_proxy": False,
        "multi_node": False,
        "workload_kind": None,
        "kv_transfer_config": None,
        "all2all_backend": None,
        "imex_resource_claim_template": None,
        "api_server_count": 1,
        "explicit_env": frozenset(),
    }
    values.update(overrides)
    return FeatureContext(**values)


def test_prefill_decode_transitively_enables_llm_d():
    plan = resolve_features(_context(prefill_decode=True, routing_proxy=True))

    assert plan.has(Feature.PREFILL_DECODE)
    assert plan.has(Feature.LLM_D)
    assert plan.routing_proxy is True


def test_external_dp_is_derived_from_data_parallel_plus_llm_d():
    assert resolve_features(_context(dp_enabled=True)).external_dp is False
    assert resolve_features(_context(llm_d_enabled=True)).external_dp is False
    assert (
        resolve_features(_context(dp_enabled=True, llm_d_enabled=True)).external_dp
        is True
    )


def test_direct_vllm_keeps_data_parallel_internal():
    plan = resolve_features(_context(dp_enabled=True, llm_d_enabled=False))

    assert plan.has(Feature.DATA_PARALLEL)
    assert not plan.has(Feature.LLM_D)
    assert plan.external_dp is False


def test_nixl_backend_contributes_only_its_required_field_env():
    plan = resolve_features(
        _context(kv_transfer_config={"kv_connector": "NixlConnector"})
    )

    assert plan.backends == frozenset({"connector:NixlConnector"})
    contribution = plan.field_ref_env[0]
    assert (contribution.name, contribution.field_path, contribution.source) == (
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "status.podIP",
        "backend:NixlConnector",
    )


def test_other_connector_backends_are_preserved_without_nixl_env():
    plan = resolve_features(
        _context(kv_transfer_config={"kv_connector": "LMCacheConnectorV1"})
    )

    assert plan.backends == frozenset({"connector:LMCacheConnectorV1"})
    assert plan.field_ref_env == ()


def test_explicit_nixl_host_suppresses_backend_default():
    plan = resolve_features(
        _context(
            kv_transfer_config={"kv_connector": "NixlConnector"},
            explicit_env=frozenset({"VLLM_NIXL_SIDE_CHANNEL_HOST"}),
        )
    )

    assert plan.field_ref_env == ()


def test_multi_node_shape_selects_leader_worker_set_without_becoming_feature():
    single = resolve_features(_context())
    multi = resolve_features(_context(multi_node=True))

    assert single.workload_kind == WorkloadKind.DEPLOYMENT
    assert multi.workload_kind == WorkloadKind.LEADER_WORKER_SET
    assert multi.enabled == single.enabled


def test_explicit_workload_kind_overrides_single_node_default():
    plan = resolve_features(
        _context(workload_kind=WorkloadKind.LEADER_WORKER_SET)
    )

    assert plan.workload_kind == WorkloadKind.LEADER_WORKER_SET


def test_multi_node_role_rejects_explicit_deployment():
    with pytest.raises(ValueError, match="multi-node roles require a LeaderWorkerSet"):
        resolve_features(
            _context(
                multi_node=True,
                workload_kind=WorkloadKind.DEPLOYMENT,
            )
        )


def test_deepep_is_reported_as_all2all_backend_not_feature():
    plan = resolve_features(
        _context(expert_parallel=True, all2all_backend="deepep_v2")
    )

    assert plan.has(Feature.EXPERT_PARALLEL)
    assert "all2all:deepep_v2" in plan.backends
    assert all("deepep" not in str(feature) for feature in plan.enabled)


def test_imex_claim_is_derived_from_platform_capability():
    plan = resolve_features(
        _context(imex_resource_claim_template="compute-domain-template")
    )

    claim = plan.resource_claims[0]
    assert (claim.name, claim.template, claim.source) == (
        "compute-domain-channel",
        "compute-domain-template",
        "platform:imex",
    )


def test_feature_registry_rejects_implication_cycles():
    contracts = dict(CONTRACTS)
    contracts[Feature.LLM_D] = FeatureContract(
        implies=frozenset({Feature.PREFILL_DECODE})
    )

    with pytest.raises(ValueError, match="feature implication cycle"):
        validate_contracts(contracts)


def test_fabric_config_rejects_undefined_profile_references():
    with pytest.raises(ValidationError, match="undefined fabric profiles: missing"):
        FabricConfig.model_validate(
            {
                "ucx_net_devices": "",
                "default_profile": "standard",
                "expert_parallel_profiles": {"decode": "missing"},
                "profiles": {"standard": {}},
            }
        )
