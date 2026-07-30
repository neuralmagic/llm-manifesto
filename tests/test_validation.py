"""Tests for hard validation of impossible or contradictory role configurations."""

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from manifesto.cluster import CacheConfig, load_cluster
from manifesto.overrides import load_routing_profile
from manifesto.parallelism import parallel_layout
from manifesto.spec import DeploymentSpec, DpLoadBalancing, EppSpec

ROOT = Path(__file__).resolve().parents[1]


def _spec_with_role(role: dict) -> dict:
    return {
        "release": "bad",
        "topology": "aggregated",
        "model": {"id": "model", "image": "image"},
        "routing": {"kind": "disabled"},
        "roles": [role],
    }


def _role(role: dict) -> DeploymentSpec:
    return DeploymentSpec.model_validate(_spec_with_role(role)).role(role["name"])


def test_uneven_global_dp_split_is_an_error():
    role = _role(
        {
            "name": "decode",
            "lws": {"size": 4},
            "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 10, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="dp=10 does not divide evenly across 4 LWS nodes"):
        parallel_layout(role)


def test_uneven_global_tp_split_is_an_error():
    role = _role(
        {
            "name": "prefill",
            "lws": {"size": 3},
            "parallelism": {"gpus_per_node": 4, "tp": 10, "dp": False, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="tp=10 is not divisible by 4 GPUs per pod"):
        parallel_layout(role)


def test_cross_node_tp_with_dp_uses_one_global_lws_node_rank_space():
    role = _role(
        {
            "name": "decode",
            "lws": {"size": 16},
            "parallelism": {"gpus_per_node": 4, "tp": 16, "dp": 4, "ep": True},
        }
    )

    layout = parallel_layout(role)
    assert layout.tp_local_size == 4
    assert layout.dp_local_size == 1
    assert layout.distributed_dp is True


def test_cross_node_tp_with_dp_requires_all_tp_groups_in_one_lws_group():
    role = _role(
        {
            "name": "decode",
            "lws": {"size": 2, "replicas": 1},
            "parallelism": {"gpus_per_node": 4, "tp": 8, "dp": 2, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="needs lws.size=4"):
        parallel_layout(role)


def test_idle_gpus_without_dp_is_an_error():
    role = _role(
        {
            "name": "prefill",
            "lws": {"size": 1},
            "parallelism": {"gpus_per_node": 4, "tp": 2, "dp": False, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="DP is disabled but local TP 2 leaves 2 of 4 GPUs idle"):
        parallel_layout(role)


def test_dp_tp_gpu_partition_mismatch_is_an_error():
    role = _role(
        {
            "name": "decode",
            "lws": {"size": 4},
            "parallelism": {"gpus_per_node": 4, "tp": 2, "dp": 4, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="needs 2 GPUs per pod, got 4"):
        parallel_layout(role)


def test_gpus_not_divisible_by_local_tp_is_an_error():
    role = _role(
        {
            "name": "decode",
            "lws": {"size": 1},
            "parallelism": {"gpus_per_node": 4, "tp": 3, "dp": False, "ep": True},
        }
    )

    with pytest.raises(ValueError, match="4 GPUs per pod is not divisible by local TP 3"):
        parallel_layout(role)


def test_dp_true_is_rejected():
    with pytest.raises(ValidationError, match="dp: true is ambiguous"):
        DeploymentSpec.model_validate(
            _spec_with_role({"name": "decode", "parallelism": {"tp": 1, "dp": True}})
        )


def test_epp_selected_plugin_config_must_exist():
    with pytest.raises(ValidationError, match="is not present in plugin_configs"):
        EppSpec(
            plugins_config_file="kv-aware.yaml",
            plugin_configs={"default.yaml": {"kind": "EndpointPickerConfig"}},
        )


def test_epp_plugin_config_file_must_be_a_config_map_key():
    with pytest.raises(ValidationError, match="must be a ConfigMap key"):
        EppSpec(plugins_config_file="profiles/kv-aware.yaml")


def test_epp_custom_selected_file_requires_plugin_configs():
    with pytest.raises(ValidationError, match="requires plugin_configs"):
        EppSpec(plugins_config_file="kv-aware.yaml")


def test_routing_profile_must_be_an_endpoint_picker_config(tmp_path):
    profile = tmp_path / "invalid-routing.yaml"
    profile.write_text("kind: ConfigMap\n")
    with pytest.raises(ValueError, match="kind: EndpointPickerConfig"):
        load_routing_profile(profile)


def test_host_cache_paths_must_be_configured_as_a_pair():
    with pytest.raises(ValidationError, match="must be configured together"):
        CacheConfig.model_validate({"hf_host_path": "/cache/hf"})


def test_routing_proxy_sets_default_port_bases():
    data = _spec_with_role(
        {
            "name": "decode",
            "lws": {"size": 4},
            "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
            "routing_proxy": True,
        }
    )
    data["routing"] = {"kind": "load_aware"}
    spec = DeploymentSpec.model_validate(data)

    role = spec.role("decode")
    assert role.routing_proxy is True
    assert role.serving_port_base == 8000
    assert role.backend_port_base == 8200


def test_routing_proxy_requires_llm_d_routing():
    with pytest.raises(ValidationError, match="routing_proxy requires llm-d routing"):
        DeploymentSpec.model_validate(
            _spec_with_role(
                {
                    "name": "decode",
                    "lws": {"size": 4},
                    "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
                    "routing_proxy": True,
                }
            )
        )


def test_llm_d_with_data_parallelism_infers_external_dp():
    data = _spec_with_role(
        {
            "name": "decode",
            "lws": {"size": 4},
            "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
        }
    )
    data["routing"] = {"kind": "load_aware"}

    spec = DeploymentSpec.model_validate(data)

    assert spec.role("decode").dp_load_balancing == DpLoadBalancing.EXTERNAL


def test_direct_vllm_with_data_parallelism_uses_internal_dp():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "decode",
                "lws": {"size": 4},
                "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
            }
        )
    )

    assert spec.role("decode").dp_load_balancing == DpLoadBalancing.INTERNAL


def test_explicit_dp_mode_must_match_derived_serving_mode():
    data = _spec_with_role(
        {
            "name": "decode",
            "lws": {"size": 4},
            "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
            "dp_load_balancing": "internal",
        }
    )
    data["routing"] = {"kind": "load_aware"}

    with pytest.raises(ValidationError, match="dp_load_balancing is derived as external"):
        DeploymentSpec.model_validate(data)


def test_external_dp_with_multiple_api_servers_is_an_error():
    data = _spec_with_role(
        {
            "name": "decode",
            "lws": {"size": 4},
            "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
            "vllm": {"api_server_count": 4},
        }
    )
    data["routing"] = {"kind": "load_aware"}

    with pytest.raises(ValidationError, match="api_server_count > 1 is incompatible"):
        DeploymentSpec.model_validate(data)


def test_pd_topology_sets_decode_proxy_without_role_flag():
    spec = DeploymentSpec.model_validate(
        {
            "release": "pd",
            "topology": "pd",
            "model": {"id": "model", "image": "image"},
            "roles": [
                {
                    "name": "decode",
                    "lws": {"size": 4},
                    "parallelism": {"gpus_per_node": 4, "tp": 1, "dp": 16, "ep": True},
                },
                {
                    "name": "prefill",
                    "lws": {"size": 2},
                    "parallelism": {"gpus_per_node": 4, "tp": 8, "dp": False, "ep": True},
                },
            ],
        }
    )

    assert spec.role("decode").routing_proxy is True
    assert spec.role("decode").dp_load_balancing == "external"
    assert spec.role("decode").backend_port_base == 8200
    assert spec.role("prefill").routing_proxy is False


def test_parallelism_gpus_alias_still_parses():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "decode",
                "lws": {"size": 4},
                "parallelism": {"gpus": 4, "tp": 1, "dp": 16, "ep": True},
            }
        )
    )

    role = spec.role("decode")
    assert role.lws.size == 4
    assert role.gpus_per_pod == 4


def test_unknown_role_keys_are_rejected():
    with pytest.raises(ValidationError):
        DeploymentSpec.model_validate(
            _spec_with_role({"name": "decode", "tensor_parallel_size": 4})
        )
    with pytest.raises(ValidationError):
        DeploymentSpec.model_validate(
            _spec_with_role({"name": "decode", "parallelism": {"tp": 1, "dp_load_balancing": "external"}})
        )


def test_warns_when_dp_replicas_fit_by_gpu_but_not_aggregate_cpu():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "prefill",
                "lws": {"size": 1, "replicas": 4},
                "parallelism": {"tp": 1, "dp": 2},
            }
        )
    )
    cluster = load_cluster(ROOT / "clusters" / "example-h200.yaml")
    cluster.model_server_resources.node_allocatable_cpu = "63"

    with pytest.warns(UserWarning, match=r"4 pods fit.*16 CPU each.*64 total.*allocatable CPU 63"):
        spec.apply_cluster_defaults(cluster)


def test_no_warning_when_dp_replicas_fit_aggregate_cpu():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "prefill",
                "lws": {"size": 1, "replicas": 4},
                "parallelism": {"tp": 1, "dp": 2},
            }
        )
    )
    cluster = load_cluster(ROOT / "clusters" / "example-h200.yaml")
    cluster.model_server_resources.node_allocatable_cpu = "64"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec.apply_cluster_defaults(cluster)


def test_warns_when_dp_replicas_fit_by_gpu_but_not_aggregate_memory():
    spec = DeploymentSpec.model_validate(
        _spec_with_role(
            {
                "name": "prefill",
                "lws": {"size": 1, "replicas": 4},
                "parallelism": {"tp": 1, "dp": 2},
            }
        )
    )
    cluster = load_cluster(ROOT / "clusters" / "example-h200.yaml")
    cluster.model_server_resources.node_allocatable_memory = "1023Gi"

    with pytest.warns(UserWarning, match=r"4 pods fit.*256Gi memory each.*1024Gi total"):
        spec.apply_cluster_defaults(cluster)
