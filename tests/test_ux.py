"""UX-level tests for compact YAML syntax, equations, and generated manifests."""

from pathlib import Path

from manifesto.cluster import load_cluster
from manifesto.instance import Instance
from manifesto.parallelism import parallel_layout
from manifesto.render import render
from manifesto.resolve import resolve_role
from manifesto.spec import DeploymentSpec, DpLoadBalancing, RoleSpec, RoutingKind, load_spec


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = load_cluster(ROOT / "clusters" / "example-gb200.yaml")
DEEPSEEK = ROOT / "models" / "deepseek-v4" / "1P-EP8-1D-EP8.yaml"


def test_compact_parallelism_and_equations_resolve_to_runtime_values():
    spec = load_spec(DEEPSEEK, CLUSTER)
    role = spec.role("decode")
    layout = parallel_layout(role)
    resolved = resolve_role(spec, Instance("tester", spec.release), CLUSTER, role)

    assert role.gpus_per_pod == 4
    assert role.parallelism.tp == 1
    assert role.parallelism.dp_enabled is True
    assert layout.dp_local_size == 4
    assert role.parallelism.ep is True
    assert role.dp_load_balancing == DpLoadBalancing.EXTERNAL

    assert resolved.env["MAX_TOKENS"] == "1024"
    assert "UCX_NET_DEVICES" not in resolved.env
    assert resolved.env["NVSHMEM_QP_DEPTH"] == "2050"
    assert resolved.vllm_args["max_num_batched_tokens"] == 1024
    assert resolved.vllm_args["max_num_seqs"] == 1024
    assert resolved.vllm_args["max_cudagraph_capture_size"] == 1024
    assert resolved.vllm_args["disable_access_log_for_endpoints"] == "/health,/v1/models,/metrics"


def test_role_vllm_args_override_manifesto_defaults():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    role = spec.role("decode")
    role.vllm_args["disable_access_log_for_endpoints"] = "/health"

    resolved = resolve_role(spec, Instance("tester", spec.release), CLUSTER, role)

    assert resolved.vllm_args["disable_access_log_for_endpoints"] == "/health"


def test_role_raw_vllm_args_are_resolved_for_low_level_integrations():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    role = spec.role("decode")
    role.vllm_raw_args = ["--trust-remote-code", "--custom.flag=value"]

    resolved = resolve_role(spec, Instance("tester", spec.release), CLUSTER, role)

    assert resolved.vllm_raw_args == ["--trust-remote-code", "--custom.flag=value"]


def test_role_schema_accepts_raw_vllm_args():
    role = RoleSpec.model_validate(
        {
            "name": "decode",
            "vllm_raw_args": ["--trust-remote-code", "--custom.flag=value"],
        }
    )

    assert role.vllm_raw_args == ["--trust-remote-code", "--custom.flag=value"]


def test_fabric_profiles_are_cluster_config_driven():
    pd = load_spec(DEEPSEEK, CLUSTER)
    decode = resolve_role(pd, Instance("tester", pd.release), CLUSTER, pd.role("decode"))
    qwen = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    standard = resolve_role(qwen, Instance("tester", "qwen"), CLUSTER, qwen.role("decode"))

    assert decode.fabric_profile == "deepep_decode"
    assert decode.env["NCCL_MNNVL_ENABLE"] == "1"
    assert standard.fabric_profile == "standard"
    assert "NCCL_MNNVL_ENABLE" not in standard.env


def test_dp_is_global_and_local_dp_is_derived_from_lws_size():
    spec = load_spec(DEEPSEEK, CLUSTER)
    role = spec.role("decode")
    resolved = resolve_role(spec, Instance("tester", spec.release), CLUSTER, role)

    assert role.lws.size == 2
    assert parallel_layout(role).dp_local_size == 4
    assert role.routing_proxy is True
    assert role.serving_port_base == 8000
    assert role.backend_port_base == 8200
    assert resolved.env["MAX_TOKENS"] == "1024"


def test_pd_topology_adds_decode_routing_proxy_defaults():
    spec = load_spec(DEEPSEEK, CLUSTER)
    role = spec.role("decode")

    assert spec.routing.kind == RoutingKind.PD
    assert spec.routing.target_role == "decode"
    assert role.routing_proxy is True
    assert role.serving_port_base == 8000
    assert role.backend_port_base == 8200


def test_equations_get_explicit_dp_scopes():
    spec = load_spec(DEEPSEEK, CLUSTER)
    role = spec.role("decode")
    role.computed["env"] = {
        "DP_LOCAL": "dp_local_size",
        "DP_WORLD": "dp_world_size",
    }
    resolved = resolve_role(spec, Instance("tester", spec.release), CLUSTER, role)

    assert resolved.env["DP_LOCAL"] == "4"
    assert resolved.env["DP_WORLD"] == "8"


def test_prefill_tp_spans_lws_nodes():
    spec = load_spec(DEEPSEEK, CLUSTER)
    role = spec.role("prefill")
    resolved = resolve_role(spec, Instance("tester", spec.release), CLUSTER, role)

    assert role.parallelism.tp == 1
    assert role.parallelism.dp_enabled is True
    assert resolved.vllm_args["trust_remote_code"] is True


def test_single_gpu_no_dp_role_derives_one_gpu_from_tp():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    role = spec.role("decode")

    assert role.gpus_per_pod == 1
    assert role.resources.gpus == 1
    assert role.resources.cpu == "8"
    assert role.resources.memory == "64Gi"


def test_omitted_resources_scale_with_local_dp_gpu_shape():
    spec = DeploymentSpec.model_validate(
        {
            "release": "scaled",
            "topology": "aggregated",
            "model": {"id": "model", "image": "image"},
            "routing": {"kind": "disabled"},
            "roles": [
                {
                    "name": "prefill",
                    "lws": {"size": 1, "replicas": 4},
                    "parallelism": {"tp": 1, "dp": 2},
                }
            ],
        }
    )
    cluster = CLUSTER.model_copy(deep=True)
    cluster.gpus_per_node = 8
    cluster.model_server_resources.cpu_per_gpu = "8"
    cluster.model_server_resources.memory_per_gpu = "64Gi"

    spec.apply_cluster_defaults(cluster)

    role = spec.role("prefill")
    assert role.gpus_per_pod == 2
    assert role.resources.cpu == "16"
    assert role.resources.memory == "128Gi"


def test_explicit_cpu_and_memory_are_preserved_exactly():
    spec = DeploymentSpec.model_validate(
        {
            "release": "explicit",
            "topology": "aggregated",
            "model": {"id": "model", "image": "image"},
            "routing": {"kind": "disabled"},
            "roles": [
                {
                    "name": "decode",
                    "parallelism": {"tp": 1},
                    "resources": {"cpu": "3500m", "memory": "70Gi"},
                }
            ],
        }
    )

    spec.apply_cluster_defaults(CLUSTER)

    assert spec.role("decode").resources.cpu == "3500m"
    assert spec.role("decode").resources.memory == "70Gi"


def test_cache_key_comes_from_image_identity_unless_overridden():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)

    assert spec.cache_key == "latest"
    spec.model.image = "registry.example/vllm@sha256:abc123"
    assert spec.cache_key == "sha256-abc123"
    spec.cache.key = "dev/build 42"
    assert spec.cache_key == "dev-build-42"


def test_explicit_resource_gpu_request_overrides_inferred_request():
    spec = DeploymentSpec.model_validate(
        {
            "release": "gpus",
            "topology": "aggregated",
            "model": {"id": "model", "image": "image"},
            "routing": {"kind": "disabled"},
            "roles": [
                {
                    "name": "prefill",
                    "lws": {"size": 1},
                    "parallelism": {"tp": 1, "dp": 2},
                    "resources": {"gpus": 1},
                }
            ],
        }
    )
    spec.apply_cluster_defaults(CLUSTER.model_copy(update={"gpus_per_node": 8}))

    assert spec.role("prefill").gpus_per_pod == 2
    assert spec.role("prefill").resources.gpus == 1


def test_cluster_path_templates_feed_cache_dev_and_logs():
    cluster = CLUSTER.with_path_overrides(
        user_root="/vol/{user}",
        log_root="/logs/{user}/{release}",
        cache_root="/cache/{user}/{release}/{gpu_arch}/{cuda}/{cache_key}",
        dev_venv="/venvs/{user}/{release}",
        dev_source="/src/{user}",
    )
    spec = load_spec(DEEPSEEK, cluster)
    spec.runtime.dev = True
    role = spec.role("decode")
    instance = Instance("Tester.Name", spec.release)

    resolved = resolve_role(spec, instance, cluster, role)
    objects = render(spec, user="Tester.Name", cluster=cluster)
    lws = next(obj for obj in objects if obj["kind"] == "LeaderWorkerSet" and obj["metadata"]["name"].endswith("decode"))
    script = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]["args"][0]

    assert resolved.env["MANIFESTO_VLLM_DEV_VENV"] == "/venvs/tester-name/wide-ep-1p-ep8-1d-ep8"
    assert resolved.env["VLLM_CACHE_ROOT"] == "/cache/tester-name/wide-ep-1p-ep8-1d-ep8/gb200/cu13/latest/vllm"
    assert resolved.env["HOME"] == "/cache/tester-name/wide-ep-1p-ep8-1d-ep8/gb200/cu13/latest/home"
    assert "USER" not in resolved.env
    assert resolved.env["TRITON_CACHE_DIR"].endswith("/latest/triton")
    assert resolved.env["TORCHINDUCTOR_CACHE_DIR"].endswith("/latest/torchinductor")
    assert "LOG_DIR=/logs/tester-name/wide-ep-1p-ep8-1d-ep8/decode" in script
    assert "find /src/tester-name/vllm" in script
    assert "ucx-lib" not in script


def test_openshift_cluster_sets_stable_user_for_arbitrary_uid():
    cluster = CLUSTER.model_copy(update={"platform": "openshift"})
    spec = load_spec(DEEPSEEK, cluster)
    resolved = resolve_role(spec, Instance("tester", spec.release), cluster, spec.role("decode"))

    assert resolved.env["USER"] == "vllm"


def test_pre_launch_hooks_run_before_rank_launch_setup():
    spec = load_spec(DEEPSEEK, CLUSTER)
    spec.runtime.pre_launch.append("echo runtime-hook")
    role = spec.role("decode")
    role.pre_launch.append("echo role-hook")

    objects = render(spec, user="tester", cluster=CLUSTER)
    lws = next(obj for obj in objects if obj["kind"] == "LeaderWorkerSet" and obj["metadata"]["name"].endswith("decode"))
    script = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]["args"][0]

    assert script.index("source /opt/vllm/bin/activate") < script.index("echo runtime-hook")
    assert script.index("echo runtime-hook") < script.index("echo role-hook")
    assert script.index("echo role-hook") < script.index("DP_SIZE_LOCAL=4")
