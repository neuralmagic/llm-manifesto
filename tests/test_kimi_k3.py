"""Regression coverage for the aggregated GB200 Kimi K3 WideEP deployment."""

from pathlib import Path

import yaml

from manifesto.cluster import load_cluster
from manifesto.parallelism import parallel_layout
from manifesto.render import render
from manifesto.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "kimi-k3" / "aggregated-tp8-ep8.yaml"
CLUSTER = load_cluster(ROOT / "clusters" / "example-gb200.yaml")


def _workload(objects: list[dict], role: str) -> dict:
    return next(
        obj
        for obj in objects
        if obj["kind"] == "LeaderWorkerSet"
        and obj["metadata"]["labels"]["llm-d.ai/role"] == role
    )


def test_kimi_k3_aggregated_wide_ep_shape_and_backends():
    spec = load_spec(MODEL, CLUSTER)
    decode = spec.role("decode")

    assert spec.topology == "aggregated"
    assert spec.model.id == "moonshotai/Kimi-K3"
    assert spec.model.image == "vllm/vllm-openai:kimi-k3"

    assert decode.lws.size == 2
    assert decode.parallelism.tp == 8
    assert decode.parallelism.dp_enabled is False
    assert parallel_layout(decode).tp_local_size == 4
    assert parallel_layout(decode).cross_node_tp is True
    assert parallel_layout(decode).dp_local_size == 1
    assert "decode_context_parallel_size" not in decode.vllm_args

    assert decode.parallelism.ep is True
    assert decode.kv_transfer_config is None
    assert decode.vllm_args["all2all_backend"] == "deepep_v2"
    assert decode.vllm_args["block_size"] == 128


def test_kimi_k3_rendered_pods_request_full_gb200_nodes():
    objects = render(load_spec(MODEL, CLUSTER), user="tester", cluster=CLUSTER)
    workload = _workload(objects, "decode")
    pod_spec = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    container = next(item for item in pod_spec["containers"] if item["name"] == "vllm")

    assert container["resources"]["requests"]["nvidia.com/gpu"] == "4"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert not any(item["name"] == "VLLM_NIXL_SIDE_CHANNEL_HOST" for item in container["env"])

    decode_script = container["args"][0]
    assert "--tensor-parallel-size 8" in decode_script
    assert "--nnodes 2" in decode_script
    assert "--node-rank $LWS_WORKER_INDEX" in decode_script
    assert '--master-addr "${LWS_LEADER_ADDRESS}"' in decode_script
    assert 'HEADLESS_ARGS=(--headless)' in decode_script
    assert '"${HEADLESS_ARGS[@]}"' in decode_script
    assert "--decode-context-parallel-size" not in decode_script
    assert "--all2all-backend deepep_v2" in decode_script
    assert "--kv_transfer_config" not in decode_script
    assert "--data-parallel-size" not in decode_script
    assert "--data-parallel-multi-port-external-lb" not in decode_script
    assert "startupProbe" not in container

    readiness = container["readinessProbe"]["exec"]["command"][-1]
    assert '${LWS_WORKER_INDEX:-0}' in readiness
    assert "then exit 0" in readiness

    service = next(obj for obj in objects if obj["kind"] == "Service" and obj["metadata"]["name"].endswith("decode-svc"))
    assert service["spec"]["selector"]["leaderworkerset.sigs.k8s.io/worker-index"] == "0"

    pool = next(obj for obj in objects if obj["kind"] == "InferencePool")
    assert (
        "leaderworkerset.sigs.k8s.io/worker-index"
        not in pool["spec"]["selector"]["matchLabels"]
    )

    config = next(
        obj
        for obj in objects
        if obj["kind"] == "ConfigMap"
        and obj["metadata"]["name"].endswith("epp-config")
    )
    plugins = yaml.safe_load(config["data"]["plugins.yaml"])["plugins"]
    api_filter = next(
        plugin
        for plugin in plugins
        if plugin.get("name") == "manifesto-default-api-server-filter"
    )
    assert api_filter["parameters"]["validValues"] == ["0"]
