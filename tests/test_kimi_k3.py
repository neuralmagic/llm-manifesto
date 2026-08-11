"""Regression coverage for the GB200 Kimi K3 WideEP deployments."""

from pathlib import Path

import yaml

from manifesto.cluster import load_cluster
from manifesto.parallelism import parallel_layout
from manifesto.render import render
from manifesto.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "kimi-k3" / "aggregated-tp16-ep16.yaml"
PD_MODEL = ROOT / "models" / "kimi-k3" / "1P-1D-DP4-TP4.yaml"
CLUSTER = load_cluster(ROOT / "clusters" / "example-gb200.yaml")
ACTIVE_PORTS = "inference.networking.k8s.io/active-ports"
KIMI_MODELS = tuple(sorted((ROOT / "models" / "kimi-k3").glob("*.yaml")))


def _workload(objects: list[dict], role: str) -> dict:
    return next(
        obj
        for obj in objects
        if obj["kind"] == "LeaderWorkerSet"
        and obj["metadata"]["labels"]["llm-d.ai/role"] == role
    )


def test_all_kimi_k3_lws_roles_stay_within_one_gpu_clique():
    assert KIMI_MODELS

    for model in KIMI_MODELS:
        spec = load_spec(model, CLUSTER)
        objects = render(spec, user="tester", cluster=CLUSTER)

        for role in spec.roles:
            if role.lws.size == 1:
                continue

            assert role.lws.same_topology_key == "nvidia.com/gpu.clique", (
                f"{model.name} role {role.name} does not require clique co-location"
            )
            pod_spec = _workload(objects, role.name)["spec"]["leaderWorkerTemplate"][
                "workerTemplate"
            ]["spec"]
            required = pod_spec["affinity"]["podAffinity"][
                "requiredDuringSchedulingIgnoredDuringExecution"
            ]
            assert any(
                term["topologyKey"] == "nvidia.com/gpu.clique" for term in required
            )


def test_kimi_k3_aggregated_wide_ep_shape_and_backends():
    spec = load_spec(MODEL, CLUSTER)
    decode = spec.role("decode")

    assert spec.topology == "aggregated"
    assert spec.model.id == "moonshotai/Kimi-K3"
    assert spec.model.image == "vllm/vllm-openai:nightly"

    assert decode.lws.size == 4
    assert decode.lws.same_topology_key == "nvidia.com/gpu.clique"
    assert decode.parallelism.tp == 16
    assert decode.parallelism.dp_enabled is False
    assert parallel_layout(decode).tp_local_size == 4
    assert parallel_layout(decode).cross_node_tp is True
    assert parallel_layout(decode).dp_local_size == 1
    assert "decode_context_parallel_size" not in decode.vllm_args

    assert decode.parallelism.ep is True
    assert decode.kv_transfer_config is None
    assert "all2all_backend" not in decode.vllm_args
    assert decode.vllm_args["moe_backend"] == "deep_gemm_mega_moe"
    assert decode.vllm_args["block_size"] == 128


def test_kimi_k3_rendered_pods_request_full_gb200_nodes():
    objects = render(load_spec(MODEL, CLUSTER), user="tester", cluster=CLUSTER)
    workload = _workload(objects, "decode")
    pod_spec = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    container = next(item for item in pod_spec["containers"] if item["name"] == "vllm")

    assert container["resources"]["requests"]["nvidia.com/gpu"] == "4"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert container["resources"]["requests"]["cpu"] == "14"
    assert container["resources"]["requests"]["memory"] == "224Gi"
    assert container["resources"]["limits"]["memory"] == "224Gi"
    assert not any(item["name"] == "VLLM_NIXL_SIDE_CHANNEL_HOST" for item in container["env"])
    assert not any(item["name"] == "VLLM_USE_RUST_FRONTEND" for item in container["env"])

    affinity = pod_spec["affinity"]
    clique_term = affinity["podAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ][0]
    assert clique_term["topologyKey"] == "nvidia.com/gpu.clique"
    assert clique_term["labelSelector"]["matchLabels"] == {
        "app.kubernetes.io/instance": "kimi-k3-aggregated-tp16-ep16",
        "llm-d.ai/role": "decode",
    }
    # Co-location is per LWS group, not per role, so replicas stay independent.
    assert clique_term["matchLabelKeys"] == ["leaderworkerset.sigs.k8s.io/group-key"]

    decode_script = container["args"][0]
    assert "--tensor-parallel-size 16" in decode_script
    assert "--nnodes 4" in decode_script
    assert "--node-rank $LWS_WORKER_INDEX" in decode_script
    assert '--master-addr "${LWS_LEADER_ADDRESS}"' in decode_script
    assert 'HEADLESS_ARGS=(--headless)' in decode_script
    assert '"${HEADLESS_ARGS[@]}"' in decode_script
    assert "--decode-context-parallel-size" not in decode_script
    assert "--all2all-backend" not in decode_script
    assert "--moe-backend deep_gemm_mega_moe" in decode_script
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


def _pod_template(objects: list[dict], role: str) -> dict:
    return _workload(objects, role)["spec"]["leaderWorkerTemplate"]["workerTemplate"]


def _script(objects: list[dict], role: str) -> str:
    containers = _pod_template(objects, role)["spec"]["containers"]
    return next(item for item in containers if item["name"] == "vllm")["args"][0]


def _annotations(objects: list[dict], role: str) -> dict:
    return _pod_template(objects, role)["metadata"]["annotations"]


def _pd_objects() -> list[dict]:
    return render(load_spec(PD_MODEL, CLUSTER), user="tester", cluster=CLUSTER)


def test_kimi_k3_pd_dp4_tp4_shape():
    spec = load_spec(PD_MODEL, CLUSTER)
    prefill = spec.role("prefill")
    decode = spec.role("decode")

    assert spec.topology == "pd"
    assert spec.model.id == "moonshotai/Kimi-K3"

    # Prefill: TP4/DP4/EP16 across 4 four-GPU nodes. TP stays inside a node, so
    # there are no headless followers and every pod serves an API.
    prefill_layout = parallel_layout(prefill)
    assert prefill.lws.size == 4
    assert prefill.parallelism.tp == 4
    assert prefill.parallelism.dp_size == 4
    assert prefill_layout.cross_node_tp is False
    assert prefill_layout.tp_local_size == 4
    assert prefill_layout.dp_local_size == 1
    # TRTLLM-GEN MoE is unusable here: it has no batched-GEMM kernel for this
    # MXFP4 checkpoint, so both roles keep the recipe's backends.
    assert prefill.vllm_args["moe_backend"] == "auto"
    assert prefill.vllm_args["enforce_eager"] is True
    assert prefill.vllm_args["max_num_batched_tokens"] == 16384

    # Decode: identical TP4/DP4/EP16 shape.
    decode_layout = parallel_layout(decode)
    assert decode.lws.size == 4
    assert decode.parallelism.tp == 4
    assert decode.parallelism.dp_size == 4
    assert decode_layout.cross_node_tp is False
    assert decode_layout.dp_local_size == 1
    assert decode.vllm_args["moe_backend"] == "auto"
    assert decode.vllm_args["max_num_seqs"] == 32
    assert decode.vllm_args["max_num_batched_tokens"] == 32

    # 8 nodes / 32 GPUs total, and each role pins to one NVLink clique.
    assert prefill.lws.size + decode.lws.size == 8
    assert prefill.lws.same_topology_key == "nvidia.com/gpu.clique"
    assert decode.lws.same_topology_key == "nvidia.com/gpu.clique"

    # Both roles keep the repository's PD connector contract.
    for role in (prefill, decode):
        assert role.parallelism.ep is True
        assert role.kv_transfer_config["kv_connector"] == "NixlConnector"
        assert role.kv_transfer_config["kv_role"] == "kv_both"


def test_kimi_k3_pd_launch_scripts_match_recipe():
    objects = _pd_objects()
    prefill_script = _script(objects, "prefill")
    decode_script = _script(objects, "decode")

    # Prefill: intra-node TP4 with DP across nodes, so no headless followers.
    assert "--tensor-parallel-size 4" in prefill_script
    assert "--nnodes" not in prefill_script
    assert "HEADLESS_ARGS" not in prefill_script
    assert "--data-parallel-size $DP_SIZE" in prefill_script
    assert "--moe-backend auto" in prefill_script
    assert "--enforce-eager" in prefill_script
    assert "--no-enable-flashinfer-autotune" in prefill_script

    # Externally load-balanced decode: recipe hybrid-lb becomes Manifesto's
    # multi-port external LB, so coordinator flags are never copied verbatim.
    assert "--tensor-parallel-size 4" in decode_script
    assert "--data-parallel-size $DP_SIZE" in decode_script
    assert "--data-parallel-hybrid-lb" not in decode_script
    assert "--moe-backend auto" in decode_script
    assert "--no-enable-prefix-caching" in decode_script
    assert '--compilation-config \'{"cudagraph_mode":"FULL_DECODE_ONLY"}\'' in decode_script

    # The recipe enables no all-to-all backend; EP alone must not infer one.
    for script in (prefill_script, decode_script):
        assert "--all2all-backend" not in script
        assert "--enable-expert-parallel" in script
        assert '"kv_role":"kv_both"' in script
        assert "--tool-call-parser kimi_k3" in script
        assert "--reasoning-parser kimi_k3" in script


def test_kimi_k3_pd_pool_advertises_each_role_live_ports():
    objects = _pd_objects()

    # Both roles run one DP rank per pod, so the pool targets a single port.
    pool = next(obj for obj in objects if obj["kind"] == "InferencePool")
    assert pool["spec"]["targetPorts"] == [{"number": 8000}]

    # The endpoint picker assumes every target port is live on every selected
    # pod, so each role still declares exactly what it serves.
    assert _annotations(objects, "prefill")[ACTIVE_PORTS] == "8000"
    assert _annotations(objects, "decode")[ACTIVE_PORTS] == "8000"

    # Neither role uses cross-node TP now, so no leader-only filter is needed.
    config = next(
        obj
        for obj in objects
        if obj["kind"] == "ConfigMap" and obj["metadata"]["name"].endswith("epp-config")
    )
    plugins = yaml.safe_load(config["data"]["plugins.yaml"])
    assert not any(
        "api-server-filter" in plugin.get("name", "") for plugin in plugins["plugins"]
    )


def test_kimi_k3_pd_decode_enables_torch_profiler():
    objects = _pd_objects()
    decode_script = _script(objects, "decode")

    trace_dir = "/mnt/shared/tester/logs/decode/traces"
    assert f"mkdir -p {trace_dir}" in decode_script
    assert f'"torch_profiler_dir":"{trace_dir}"' in decode_script
    assert '"profiler":"torch"' in decode_script
    assert '"delay_iterations":200' in decode_script
    assert '"max_iterations":50' in decode_script
    assert '"ignore_frontend":true' in decode_script

    # Profiling is opt-in per role; prefill stays untouched.
    assert "--profiler-config" not in _script(objects, "prefill")


def test_kimi_k3_pd_clique_affinity_is_scoped_per_role():
    objects = _pd_objects()

    terms = {}
    for role in ("prefill", "decode"):
        pod_spec = _pod_template(objects, role)["spec"]
        required = pod_spec["affinity"]["podAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]
        assert len(required) == 1
        terms[role] = required[0]
        assert terms[role]["topologyKey"] == "nvidia.com/gpu.clique"

    # Each role packs into one NVLink domain...
    assert terms["prefill"]["labelSelector"]["matchLabels"] == {
        "app.kubernetes.io/instance": "kimi-k3-1p-1d-dp4tp4",
        "llm-d.ai/role": "prefill",
    }
    assert terms["decode"]["labelSelector"]["matchLabels"] == {
        "app.kubernetes.io/instance": "kimi-k3-1p-1d-dp4tp4",
        "llm-d.ai/role": "decode",
    }
    # ...but PD roles exchange KV over RDMA, so they must not be forced to
    # share one domain, which would need 6 nodes in a single clique.
    assert terms["prefill"]["labelSelector"] != terms["decode"]["labelSelector"]
    # And the domain is per LWS group, so a second replica of either role is
    # free to seed a different clique instead of piling onto the first one.
    for role in ("prefill", "decode"):
        assert terms[role]["matchLabelKeys"] == [
            "leaderworkerset.sigs.k8s.io/group-key"
        ]
