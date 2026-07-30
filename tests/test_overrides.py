"""Tests for YAML spec inheritance and role override composition."""

from pathlib import Path

import pytest

from manifesto.cluster import load_cluster
from manifesto.images import DEFAULT_IMAGES
from manifesto.overrides import merge_overrides
from manifesto.parallelism import parallel_layout
from manifesto.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = load_cluster(ROOT / "clusters" / "example-gb200.yaml")


def test_role_map_overrides_merge_by_name():
    merged = merge_overrides(
        {
            "roles": [
                {"name": "decode", "lws": {"size": 2, "replicas": 1}, "vllm": {"moe_backend": "mega"}},
                {"name": "prefill", "lws": {"size": 2, "replicas": 2}},
            ]
        },
        {"roles": {"decode": {"lws": {"size": 4}, "vllm": {"all2all_backend": "flashinfer"}}}},
    )

    assert merged["roles"][0]["lws"] == {"size": 4, "replicas": 1}
    assert merged["roles"][0]["vllm"] == {
        "moe_backend": "mega",
        "all2all_backend": "flashinfer",
    }
    assert merged["roles"][1]["lws"] == {"size": 2, "replicas": 2}


def test_role_map_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown role"):
        merge_overrides({"roles": [{"name": "decode"}]}, {"roles": {"prefill": {"lws": {"replicas": 2}}}})


@pytest.mark.parametrize(
    ("config", "prefill_replicas", "decode_size", "decode_replicas", "decode_dp"),
    [
        ("1P-EP8-1D-EP8.yaml", 1, 2, 1, 8),
        ("2P-EP8-1D-EP8.yaml", 2, 2, 1, 8),
        ("3P-EP8-1D-EP8.yaml", 3, 2, 1, 8),
        ("3P-EP8-1D-EP16.yaml", 3, 4, 1, 16),
    ],
)
def test_deepseek_v4_llmd_guide_variants_expand(
    config: str,
    prefill_replicas: int,
    decode_size: int,
    decode_replicas: int,
    decode_dp: int,
):
    spec = load_spec(ROOT / "models" / "deepseek-v4" / config, CLUSTER)
    decode = spec.role("decode")
    prefill = spec.role("prefill")

    assert spec.topology == "pd"
    assert spec.model.id == "deepseek-ai/DeepSeek-V4-Pro"
    assert spec.model.image == DEFAULT_IMAGES.get("vllm.standard")

    assert prefill.lws.size == 2
    assert prefill.lws.replicas == prefill_replicas
    assert prefill.parallelism.tp == 1
    assert prefill.parallelism.dp_enabled is True
    assert parallel_layout(prefill).dp_local_size == 4
    assert prefill.kv_transfer_config["kv_role"] == "kv_both"

    assert decode.lws.size == decode_size
    assert decode.lws.replicas == decode_replicas
    assert decode.parallelism.tp == 1
    assert decode.parallelism.dp_enabled is True
    assert parallel_layout(decode).dp_local_size == decode_dp // decode_size
    assert decode.kv_transfer_config["kv_role"] == "kv_both"
    if decode_dp == 8:
        assert decode.vllm_args["moe_backend"] == "deep_gemm_mega_moe"
        assert "all2all_backend" not in decode.vllm_args
        assert decode.vllm_args["enable_eplb"] is False
        assert "eplb_config" not in decode.vllm_args
    else:
        assert decode.vllm_args["all2all_backend"] == "deepep_v2"
        assert decode.vllm_args["enable_eplb"] is True
        assert decode.vllm_args["eplb_config"]["num_redundant_experts"] == "32"


def test_nested_eplb_config_can_be_overridden_without_restatement():
    merged = merge_overrides(
        {
            "roles": [
                {
                    "name": "decode",
                    "vllm": {
                        "eplb_config": {
                            "window_size": "100",
                            "num_redundant_experts": "32",
                        }
                    },
                }
            ]
        },
        {"roles": {"decode": {"vllm": {"eplb_config": {"num_redundant_experts": "16"}}}}},
    )

    assert merged["roles"][0]["vllm"]["eplb_config"] == {
        "window_size": "100",
        "num_redundant_experts": "16",
    }


def test_override_can_delete_inherited_key():
    merged = merge_overrides(
        {"roles": [{"name": "decode", "vllm": {"moe_backend": "auto", "all2all_backend": "deepep_v2"}}]},
        {"roles": {"decode": {"vllm": {"moe_backend": "deep_gemm_mega_moe", "all2all_backend": "$delete"}}}},
    )

    assert merged["roles"][0]["vllm"] == {"moe_backend": "deep_gemm_mega_moe"}


def test_deepseek_v4_ep8_decode_uses_megagemm_without_all2all():
    spec = load_spec(ROOT / "models" / "deepseek-v4" / "1P-EP8-1D-EP8.yaml", CLUSTER)
    decode = spec.role("decode")

    assert decode.vllm_args["moe_backend"] == "deep_gemm_mega_moe"
    assert "all2all_backend" not in decode.vllm_args
    assert decode.vllm_args["enable_eplb"] is False
    assert "eplb_config" not in decode.vllm_args


def test_deepseek_v4_ep16_decode_inherits_deepep_v2():
    spec = load_spec(ROOT / "models" / "deepseek-v4" / "3P-EP8-1D-EP16.yaml", CLUSTER)
    decode = spec.role("decode")

    assert decode.vllm_args["all2all_backend"] == "deepep_v2"
    assert decode.vllm_args["enable_eplb"] is True
