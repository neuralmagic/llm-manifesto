"""Structural tests for rendered Kubernetes objects and YAML serialization."""

import json
import os
import subprocess
import urllib.request
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pytest
import yaml

from manifesto.cluster import load_cluster
from manifesto.images import DEFAULT_IMAGES
from manifesto.overrides import load_routing_profile
from manifesto.render import render, render_to_yaml
from manifesto.render.idle_shutdown import IDLE_SHUTDOWN_SCRIPT
from manifesto.spec import EppSpec, RuntimeSpec, load_spec


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = load_cluster(ROOT / "clusters" / "example-gb200.yaml")
EXAMPLE_H200 = load_cluster(ROOT / "clusters" / "example-h200.yaml")
DEEPSEEK = "deepseek-v4/1P-EP8-1D-EP8.yaml"


def _stateless_cluster():
    return load_cluster(ROOT / "clusters" / "example-stateless-b200.yaml")


def _objects(config: str) -> list[dict]:
    spec = load_spec(ROOT / "models" / config, CLUSTER)
    return render(spec, user="tester", cluster=CLUSTER)


def _objects_with_routing_profile(config: str, profile: str) -> list[dict]:
    spec = load_spec(ROOT / "models" / config, CLUSTER)
    path, plugin_config = load_routing_profile(profile)
    spec.routing.epp = EppSpec(
        plugins_config_file=path.name,
        plugin_configs={path.name: plugin_config},
    )
    return render(spec, user="tester", cluster=CLUSTER)


def _find(objects: list[dict], kind: str, name_suffix: str | None = None) -> dict:
    for obj in objects:
        if obj["kind"] != kind:
            continue
        if name_suffix is None or obj["metadata"]["name"].endswith(name_suffix):
            return obj
    raise AssertionError(f"missing {kind} {name_suffix or ''}")


def _idle_shutdown_functions(monkeypatch) -> dict:
    monkeypatch.setenv("NAMESPACE", "test")
    monkeypatch.setenv("POD_SELECTOR", "app=test")
    monkeypatch.setenv("TARGETS", "{}")
    monkeypatch.setenv("EXPECTED_TARGETS", "0")
    monkeypatch.setenv("TIMEOUT_SECONDS", "2700")
    monkeypatch.setenv("WORKLOADS", "[]")
    monkeypatch.setattr("ssl.create_default_context", lambda **_kwargs: None)
    definitions = IDLE_SHUTDOWN_SCRIPT.split("last_activity = time.monotonic()", 1)[0]
    namespace: dict = {}
    exec(compile(definitions, "idle_shutdown.py", "exec"), namespace)
    return namespace


def test_rendered_yaml_parses():
    objects = _objects(DEEPSEEK)
    parsed = list(yaml.safe_load_all(render_to_yaml(objects)))

    assert len(parsed) == len(objects)


def test_rendered_launch_script_uses_literal_yaml_block():
    rendered = render_to_yaml(_objects(DEEPSEEK))

    assert "args:\n          - |-" in rendered
    assert "vllm \\\n              serve \\" in rendered
    assert "\\nexec vllm serve" not in rendered


def test_stateless_single_rank_render_omits_filesystem_and_distributed_baggage():
    cluster = _stateless_cluster()
    spec = load_spec(ROOT / "models" / "qwen" / "qwen3-0.6b.yaml", cluster)
    objects = render(spec, user="tester", cluster=cluster)

    assert [obj["kind"] for obj in objects] == [
        "Deployment",
        "Service",
        "ServiceAccount",
        "Role",
        "RoleBinding",
        "ConfigMap",
        "Deployment",
    ]
    deployment = _find(objects, "Deployment", "decode")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    env_names = {item["name"] for item in container["env"]}
    script = container["args"][0]

    assert pod_spec["volumes"] == [
        {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}}
    ]
    assert container["volumeMounts"] == [{"name": "dshm", "mountPath": "/dev/shm"}]
    assert env_names == {"HF_TOKEN", "TQDM_DISABLE", "VLLM_NO_USAGE_STATS"}
    assert "LOG_DIR=" not in script
    assert "MANIFESTO_VLLM_ENV" not in script
    assert "for R in" not in script
    assert "DP_SIZE" not in script
    assert "CACHE_DIR" not in script
    assert ".manifesto-running" not in script
    assert "cleanup_compile_caches" not in script
    assert "MANIFESTO_POD_UID" not in env_names
    assert "--device-ids 0" in script
    assert "--port 8000" in script
    assert container["readinessProbe"]["httpGet"] == {
        "path": "/v1/models",
        "port": 8000,
    }
    for omitted in ("imagePullPolicy", "securityContext", "workingDir"):
        assert omitted not in container
    assert "serviceAccountName" not in pod_spec
    assert "terminationGracePeriodSeconds" not in pod_spec


def test_vllm_env_requires_an_absolute_mounted_path():
    cluster = _stateless_cluster()
    spec = load_spec(ROOT / "models" / "qwen" / "qwen3-0.6b.yaml", cluster)
    spec.runtime.vllm_env = "relative/worktree"

    with pytest.raises(ValueError, match="must be an absolute path"):
        render(spec, user="tester", cluster=cluster)

    spec.runtime.vllm_env = "/unmounted/worktree"
    with pytest.raises(ValueError, match="not covered by a model pod volume mount"):
        render(spec, user="tester", cluster=cluster)

    spec.runtime.vllm_env = "/mnt/shared/../unmounted/worktree"
    with pytest.raises(ValueError, match="not covered by a model pod volume mount"):
        render(spec, user="tester", cluster=CLUSTER)


def test_idle_shutdown_is_enabled_by_default_for_45_minutes():
    spec = load_spec(ROOT / "models" / DEEPSEEK, CLUSTER)
    objects = render(spec, user="tester", cluster=CLUSTER)
    controller = _find(objects, "Deployment", "idle-shutdown")
    container = controller["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}
    workloads = json.loads(env["WORKLOADS"]["value"])
    targets = json.loads(env["TARGETS"]["value"])

    assert controller["spec"]["replicas"] == 1
    assert env["TIMEOUT_SECONDS"]["value"] == "2700"
    assert env["EXPECTED_TARGETS"]["value"] == "16"
    assert workloads[-1]["name"].endswith("idle-shutdown")
    assert all(workload["replicas"] == 1 for workload in workloads)
    assert {workload["name"] for workload in workloads[:-1]} == {
        "vllm-ep8-decode",
        "vllm-ep8-prefill",
        "wide-ep-1p-ep8-1d-ep8-infpool-epp",
    }
    assert targets["decode"]["worker_indices"] is None
    assert targets["prefill"]["worker_indices"] is None
    role = _find(objects, "Role", "idle-shutdown-rbac")
    assert role["rules"][-1] == {
        "apiGroups": ["leaderworkerset.x-k8s.io"],
        "resources": ["leaderworkersets"],
        "resourceNames": ["vllm-ep8-decode", "vllm-ep8-prefill"],
        "verbs": ["get", "patch"],
    }
    compile(
        _find(objects, "ConfigMap", "idle-shutdown")["data"]["idle_shutdown.py"],
        "idle_shutdown.py",
        "exec",
    )


def test_idle_shutdown_reloads_service_account_token(monkeypatch, tmp_path):
    namespace = _idle_shutdown_functions(monkeypatch)
    token_path = tmp_path / "token"
    namespace["TOKEN_PATH"] = str(token_path)
    authorizations = []

    def urlopen(request, **_kwargs):
        authorizations.append(request.get_header("Authorization"))
        return BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    token_path.write_text("first-token")
    namespace["api_request"]("/api/v1/pods")
    token_path.write_text("rotated-token")
    namespace["api_request"]("/api/v1/pods")

    assert authorizations == ["Bearer first-token", "Bearer rotated-token"]


def test_idle_shutdown_rolls_back_successful_scale_patches(monkeypatch):
    namespace = _idle_shutdown_functions(monkeypatch)
    namespace["WORKLOADS"] = [
        {"name": "first", "path": "/first", "replicas": 2},
        {"name": "second", "path": "/second", "replicas": 3},
        {"name": "third", "path": "/third", "replicas": 4},
    ]
    patches = []

    def api_request(path, *, method="GET", body=None):
        patches.append((path, method, body))
        if path == "/third":
            raise OSError("transient API failure")

    namespace["api_request"] = api_request

    with pytest.raises(OSError, match="transient API failure"):
        namespace["scale_to_zero"]()

    assert patches == [
        ("/first", "PATCH", {"spec": {"replicas": 0}}),
        ("/second", "PATCH", {"spec": {"replicas": 0}}),
        ("/third", "PATCH", {"spec": {"replicas": 0}}),
        ("/second", "PATCH", {"spec": {"replicas": 3}}),
        ("/first", "PATCH", {"spec": {"replicas": 2}}),
    ]


def test_idle_shutdown_can_be_disabled_or_given_a_custom_timeout():
    spec = load_spec(ROOT / "models" / "qwen" / "qwen3-0.6b.yaml", _stateless_cluster())
    spec.runtime.idle_shutdown.enabled = False
    objects = render(spec, user="tester", cluster=_stateless_cluster())

    assert not any(
        obj["metadata"].get("labels", {}).get("app.kubernetes.io/component")
        == "idle-shutdown"
        for obj in objects
    )

    spec.runtime.idle_shutdown.enabled = True
    spec.runtime.idle_shutdown.timeout_minutes = 12
    objects = render(spec, user="tester", cluster=_stateless_cluster())
    controller = _find(objects, "Deployment", "idle-shutdown")
    env = {
        item["name"]: item
        for item in controller["spec"]["template"]["spec"]["containers"][0]["env"]
    }

    assert env["TIMEOUT_SECONDS"]["value"] == "720"


def test_idle_shutdown_only_scrapes_cross_node_tp_api_servers():
    spec = load_spec(ROOT / "models" / "kimi-k3" / "aggregated-tp16-ep16.yaml", CLUSTER)
    objects = render(spec, user="tester", cluster=CLUSTER)
    controller = _find(objects, "Deployment", "idle-shutdown")
    env = {
        item["name"]: item
        for item in controller["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    targets = json.loads(env["TARGETS"]["value"])

    assert targets["decode"]["worker_indices"] == ["0"]
    assert env["EXPECTED_TARGETS"]["value"] == "1"


def test_cluster_schema_rejects_removed_dev_configuration(tmp_path):
    data = yaml.safe_load((ROOT / "clusters" / "example-gb200.yaml").read_text())
    data["dev"] = {"venv": "/mnt/shared/vllm-venv"}
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="dev"):
        load_cluster(path)


@pytest.mark.parametrize("removed", [{"dev": True}, {"dev_venv": "/custom/venv"}])
def test_runtime_schema_rejects_removed_dev_configuration(removed):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RuntimeSpec.model_validate(removed)


def test_nixl_roles_advertise_their_pod_ip():
    objects = _objects(DEEPSEEK)

    for role in ("decode", "prefill"):
        workload = _find(objects, "LeaderWorkerSet", role)
        container = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"][
            "spec"
        ]["containers"][0]
        env = {item["name"]: item for item in container["env"]}

        assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == {
            "name": "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
        }


def test_explicit_nixl_side_channel_host_is_preserved():
    spec = load_spec(ROOT / "models" / DEEPSEEK, CLUSTER)
    spec.role("decode").env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "nixl.example.test"
    objects = render(spec, user="tester", cluster=CLUSTER)
    workload = _find(objects, "LeaderWorkerSet", "decode")
    container = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"][
        "containers"
    ][0]
    matching = [
        item
        for item in container["env"]
        if item["name"] == "VLLM_NIXL_SIDE_CHANNEL_HOST"
    ]

    assert matching == [{"name": "VLLM_NIXL_SIDE_CHANNEL_HOST", "value": "nixl.example.test"}]


def test_non_nixl_role_does_not_get_side_channel_host():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    spec.role("decode").kv_transfer_config = {
        "kv_connector": "LMCacheConnectorV1",
    }
    objects = render(spec, user="tester", cluster=CLUSTER)
    workload = _find(objects, "Deployment", "decode")
    container = workload["spec"]["template"]["spec"]["containers"][0]

    assert not any(
        item["name"] == "VLLM_NIXL_SIDE_CHANNEL_HOST"
        for item in container["env"]
    )


def test_dp_ports_feed_container_readiness_and_inferencepool():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]
    infpool = _find(objects, "InferencePool")

    assert [p["containerPort"] for p in container["ports"]] == [8100, 8200, 8201, 8202, 8203]
    assert container["resources"]["requests"]["cpu"] == "32"
    assert container["resources"]["requests"]["memory"] == "512Gi"
    readiness = container["readinessProbe"]["exec"]["command"][-1]
    assert "localhost:8000" in readiness
    assert "localhost:8003" in readiness
    assert container["startupProbe"]["httpGet"]["port"] == "dp-supervisor"
    assert infpool["apiVersion"] == "inference.networking.k8s.io/v1"
    assert infpool["spec"]["targetPorts"] == [{"number": 8000}, {"number": 8001}, {"number": 8002}, {"number": 8003}]
    assert infpool["spec"]["endpointPickerRef"]["name"] == "wide-ep-1p-ep8-1d-ep8-infpool-epp"
    script = container["args"][0]
    assert "DP_SIZE=8" in script
    assert "DP_SIZE=$((LWS_GROUP_SIZE * DP_SIZE_LOCAL))" not in script
    assert "--data-parallel-multi-port-external-lb" in script
    assert "--data-parallel-supervisor-port 8100" in script
    assert "--data-parallel-start-rank $START_RANK" in script
    assert "--data-parallel-rank" not in script
    assert "vllm \\\n  serve \\\n  deepseek-ai/DeepSeek-V4-Pro \\" in script
    assert "  --port 8200 \\" in script
    assert "  --max-num-seqs 1024 \\" in script
    assert "  --enable-eplb False" not in script


def test_crash_cleanup_clears_compilation_caches_but_preserves_autotuning():
    objects = _objects(DEEPSEEK)
    workload = _find(objects, "LeaderWorkerSet", "decode")
    script = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]["args"][0]

    assert '.manifesto-running-${HOSTNAME}-${MANIFESTO_POD_UID}' in script
    assert 'if [ "$STATUS" -ne 0 ]; then' in script
    assert 'find "$VLLM_CACHE_ROOT" -type d -name torch_compile_cache' in script
    assert '${FLASHINFER_CACHE_DIR:-}' in script
    assert '${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-}' in script
    assert '${TRITON_CACHE_DIR:-}' in script
    assert '${TORCHINDUCTOR_CACHE_DIR:-}' in script
    assert '${TILELANG_CACHE_DIR:-}' in script
    assert "flashinfer_autotune_cache" not in script
    env = {
        item["name"]: item
        for item in workload["spec"]["leaderWorkerTemplate"]["workerTemplate"][
            "spec"
        ]["containers"][0]["env"]
    }
    assert env["MANIFESTO_POD_UID"] == {
        "name": "MANIFESTO_POD_UID",
        "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
    }


def test_failed_launch_removes_compile_files_and_keeps_autotune_files(tmp_path):
    objects = _objects(DEEPSEEK)
    workload = _find(objects, "LeaderWorkerSet", "decode")
    script = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]["args"][0]
    cleanup_body = script.split("CRASH_MARKER=", 1)[1].split(
        "trap on_exit EXIT", 1
    )[0]
    cleanup_body += "trap on_exit EXIT"
    cleanup_preamble = f"set -euo pipefail\nCRASH_MARKER={cleanup_body}"

    cache_paths = {
        "VLLM_CACHE_ROOT": tmp_path / "vllm",
        "FLASHINFER_CACHE_DIR": tmp_path / "flashinfer",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": tmp_path / "cute-dsl",
        "TRITON_CACHE_DIR": tmp_path / "triton",
        "TORCHINDUCTOR_CACHE_DIR": tmp_path / "torchinductor",
        "TILELANG_CACHE_DIR": tmp_path / "tilelang",
    }
    compile_dirs = [
        cache_paths["VLLM_CACHE_ROOT"] / "rank0" / "torch_compile_cache",
        *(path for name, path in cache_paths.items() if name != "VLLM_CACHE_ROOT"),
    ]
    autotune_dir = cache_paths["VLLM_CACHE_ROOT"] / "flashinfer_autotune_cache"
    for path in [*compile_dirs, autotune_dir]:
        path.mkdir(parents=True)
        (path / "cached").write_text("data")

    env = os.environ | {name: str(path) for name, path in cache_paths.items()}
    env["HOSTNAME"] = "test-pod"
    env["MANIFESTO_POD_UID"] = "test-uid"
    result = subprocess.run(
        ["/bin/bash", "-c", f"{cleanup_preamble}\nexit 23"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    assert all(not path.exists() for path in compile_dirs)
    assert (autotune_dir / "cached").read_text() == "data"


def test_crash_cleanup_can_be_disabled():
    spec = load_spec(ROOT / "models" / DEEPSEEK, CLUSTER)
    spec.cache.cleanup_on_crash = False
    objects = render(spec, user="tester", cluster=CLUSTER)
    workload = _find(objects, "LeaderWorkerSet", "decode")
    script = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]["args"][0]
    env_names = {
        item["name"]
        for item in workload["spec"]["leaderWorkerTemplate"]["workerTemplate"][
            "spec"
        ]["containers"][0]["env"]
    }

    assert ".manifesto-running" not in script
    assert "cleanup_compile_caches" not in script
    assert "MANIFESTO_POD_UID" not in env_names


def test_deepseek_lws_uses_short_workload_names_with_full_instance_labels():
    objects = _objects(DEEPSEEK)
    decode = _find(objects, "LeaderWorkerSet", "decode")
    prefill = _find(objects, "LeaderWorkerSet", "prefill")

    assert decode["metadata"]["name"] == "vllm-ep8-decode"
    assert prefill["metadata"]["name"] == "vllm-ep8-prefill"
    assert decode["metadata"]["labels"]["app.kubernetes.io/instance"] == "wide-ep-1p-ep8-1d-ep8"
    assert prefill["metadata"]["labels"]["app.kubernetes.io/instance"] == "wide-ep-1p-ep8-1d-ep8"


def test_deepseek_ep16_decode_name_keeps_decode_width():
    spec = load_spec(ROOT / "models" / "deepseek-v4" / "3P-EP8-1D-EP16.yaml", CLUSTER)
    objects = render(spec, user="tester", cluster=CLUSTER)
    decode = _find(objects, "LeaderWorkerSet", "decode")
    prefill = _find(objects, "LeaderWorkerSet", "prefill")

    assert decode["metadata"]["name"] == "vllm-ep16-decode"
    assert prefill["metadata"]["name"] == "vllm-ep8-prefill"


def test_logs_persist_to_cluster_log_root():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    pod_spec = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    container = pod_spec["containers"][0]
    script = container["args"][0]
    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    mounts = {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}

    assert volumes["shared-storage"]["persistentVolumeClaim"]["claimName"] == "example-shared-cache"
    assert mounts["shared-storage"] == "/mnt/shared"
    assert "LOG_DIR=/mnt/shared/tester/logs/decode" in script


def test_shared_storage_accepts_non_pvc_volume_sources():
    cluster = CLUSTER.model_copy(deep=True)
    cluster.storage.shared_volume = {"emptyDir": {}}
    spec = load_spec(ROOT / "models" / DEEPSEEK, cluster)
    objects = render(spec, user="tester", cluster=cluster)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    pod_spec = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    volume = next(volume for volume in pod_spec["volumes"] if volume["name"] == "shared-storage")

    assert volume == {"name": "shared-storage", "emptyDir": {}}


def test_gateway_class_comes_from_cluster_profile():
    cluster = CLUSTER.model_copy(deep=True)
    cluster.gateway.class_name = "platform-gateway"
    spec = load_spec(ROOT / "models" / DEEPSEEK, cluster)
    objects = render(spec, user="tester", cluster=cluster)
    gateway = _find(objects, "Gateway")
    gateway_options = _find(objects, "ConfigMap", "gateway-options")

    assert gateway["spec"]["gatewayClassName"] == "platform-gateway"
    assert len(f"{gateway['metadata']['name']}-platform-gateway") <= 63
    assert not any(
        obj["kind"] == "Service" and obj["metadata"]["name"].startswith(gateway["metadata"]["name"])
        for obj in objects
    )
    assert yaml.safe_load(gateway_options["data"]["service"])["spec"]["type"] == "ClusterIP"


def test_dedicated_logging_pvc_is_mounted_when_configured():
    cluster = CLUSTER.model_copy(deep=True)
    cluster.logging.pvc = "logs-pvc"
    cluster.logging.mount_path = "/mnt/logs"
    cluster.logging.root = "/mnt/logs/{user}/{release}"
    spec = load_spec(ROOT / "models" / DEEPSEEK, cluster)
    objects = render(spec, user="tester", cluster=cluster)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    pod_spec = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    container = pod_spec["containers"][0]
    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    mounts = {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}

    assert volumes["logs"]["persistentVolumeClaim"]["claimName"] == "logs-pvc"
    assert mounts["logs"] == "/mnt/logs"
    assert "LOG_DIR=/mnt/logs/tester/wide-ep-1p-ep8-1d-ep8/decode" in container["args"][0]


def test_no_dp_qwen_uses_single_port_and_no_dp_flags():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    spec.role("decode").lws.replicas = 2
    objects = render(spec, user="tester", cluster=CLUSTER)
    deployment = _find(objects, "Deployment", "decode")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    script = container["args"][0]
    infpool = _find(objects, "InferencePool")

    assert not any(obj["kind"] == "LeaderWorkerSet" for obj in objects)
    assert deployment["spec"]["replicas"] == 2
    assert deployment["spec"]["selector"]["matchLabels"] == {
        "app.kubernetes.io/instance": "qwen",
        "llm-d.ai/role": "decode",
    }
    assert deployment["spec"]["template"]["metadata"]["labels"].items() >= deployment["spec"]["selector"][
        "matchLabels"
    ].items()
    assert [p["containerPort"] for p in container["ports"]] == [8000]
    hf_token = next(env for env in container["env"] if env["name"] == "HF_TOKEN")
    assert hf_token["valueFrom"]["secretKeyRef"] == {
        "name": "hf-secret",
        "key": "HF_TOKEN",
        "optional": False,
    }
    assert "--data-parallel-size" not in script
    assert "--disable-access-log-for-endpoints /health,/v1/models,/metrics" in script
    assert "--disable-uvicorn-access-log" not in script
    assert "startupProbe" not in container
    assert infpool["spec"]["targetPorts"] == [{"number": 8000}]


def test_llmd_data_parallelism_derives_external_dp_without_pd_proxy():
    spec = load_spec(ROOT / "models" / "qwen" / "h200-aggregated.yaml", EXAMPLE_H200)
    objects = render(spec, user="tester", cluster=EXAMPLE_H200)
    deployment = _find(objects, "Deployment", "decode")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    assert spec.role("decode").dp_load_balancing == "external"
    assert "initContainers" not in pod_spec
    assert [port["containerPort"] for port in container["ports"]] == [
        8100,
        8000,
        8001,
        8002,
        8003,
    ]
    assert "--data-parallel-multi-port-external-lb" in container["args"][0]


def test_null_role_vllm_arg_omits_manifesto_default():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    spec.role("decode").vllm_args["disable_access_log_for_endpoints"] = None

    objects = render(spec, user="tester", cluster=CLUSTER)
    deployment = _find(objects, "Deployment", "decode")
    script = deployment["spec"]["template"]["spec"]["containers"][0]["args"][0]

    assert "--disable-access-log-for-endpoints" not in script


def test_role_raw_vllm_args_are_appended_without_interpretation():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    role = spec.role("decode")
    role.vllm_raw_args = [
        "--trust-remote-code",
        "--attention-config.backend=FLASH_ATTN",
        "--override-generation-config={\"temperature\":0.5}",
    ]

    objects = render(spec, user="tester", cluster=CLUSTER)
    deployment = _find(objects, "Deployment", "decode")
    script = deployment["spec"]["template"]["spec"]["containers"][0]["args"][0]

    for raw_arg in role.vllm_raw_args[:-1]:
        assert f"  {raw_arg} \\" in script
    assert script.endswith(f"  {role.vllm_raw_args[-1]}")
    assert script.index("--disable-access-log-for-endpoints") < script.index("--trust-remote-code")


def test_single_node_dp_uses_deployment_without_lws_environment():
    spec = load_spec(ROOT / "models" / "qwen" / "h200-aggregated.yaml", EXAMPLE_H200)
    objects = render(spec, user="tester", cluster=EXAMPLE_H200)
    deployment = _find(objects, "Deployment", "decode")
    script = deployment["spec"]["template"]["spec"]["containers"][0]["args"][0]

    assert "LWS_" not in script
    assert "START_RANK=0" in script
    assert "--data-parallel-address 127.0.0.1" in script


def test_pd_inferencepool_selector_includes_prefill_and_decode_roles():
    objects = _objects(DEEPSEEK)
    infpool = _find(objects, "InferencePool")

    selector = infpool["spec"]["selector"]["matchLabels"]
    assert selector["app.kubernetes.io/instance"] == "wide-ep-1p-ep8-1d-ep8"
    assert selector["llm-d.ai/deployment"] == "pd"
    assert selector["llm-d.ai/inferenceServing"] == "true"
    assert "llm-d.ai/role" not in selector


def test_pd_cross_node_tp_filters_decode_leaders_in_epp_profile():
    spec = load_spec(ROOT / "models" / DEEPSEEK, CLUSTER)
    assert spec.role("prefill").parallelism.dp_enabled

    decode = spec.role("decode")
    decode.lws.size = 2
    decode.parallelism.tp = 8
    decode.parallelism.dp = False

    objects = render(spec, user="tester", cluster=CLUSTER)
    infpool = _find(objects, "InferencePool")
    selector = infpool["spec"]["selector"]["matchLabels"]
    assert "leaderworkerset.sigs.k8s.io/worker-index" not in selector

    configmap = _find(objects, "ConfigMap", "epp-config")
    config = yaml.safe_load(configmap["data"]["plugins.yaml"])
    leader_filter = next(
        plugin
        for plugin in config["plugins"]
        if plugin.get("name") == "manifesto-decode-api-server-filter"
    )
    assert leader_filter == {
        "type": "by-label",
        "name": "manifesto-decode-api-server-filter",
        "parameters": {
            "label": "leaderworkerset.sigs.k8s.io/worker-index",
            "validValues": ["0"],
            "allowsNoLabel": False,
        },
    }

    profiles = {profile["name"]: profile for profile in config["schedulingProfiles"]}
    assert {plugin["pluginRef"] for plugin in profiles["prefill"]["plugins"]}.isdisjoint(
        {"manifesto-decode-api-server-filter"}
    )
    assert [plugin["pluginRef"] for plugin in profiles["decode"]["plugins"][:2]] == [
        "decode-filter",
        "manifesto-decode-api-server-filter",
    ]


def test_pd_cross_node_tp_filters_each_role_profile():
    spec = load_spec(ROOT / "models" / DEEPSEEK, CLUSTER)
    for role_name in ("prefill", "decode"):
        role = spec.role(role_name)
        role.lws.size = 2
        role.parallelism.tp = 8
        role.parallelism.dp = False

    objects = render(spec, user="tester", cluster=CLUSTER)
    configmap = _find(objects, "ConfigMap", "epp-config")
    config = yaml.safe_load(configmap["data"]["plugins.yaml"])
    plugins = {plugin.get("name"): plugin for plugin in config["plugins"]}
    profiles = {profile["name"]: profile for profile in config["schedulingProfiles"]}

    for profile_name in ("prefill", "decode"):
        filter_name = f"manifesto-{profile_name}-api-server-filter"
        assert plugins[filter_name]["parameters"]["validValues"] == ["0"]
        assert [
            plugin["pluginRef"]
            for plugin in profiles[profile_name]["plugins"][:2]
        ] == [f"{profile_name}-filter", filter_name]


def test_pd_dp2_tp8_decode_uses_two_routable_two_node_tp_groups():
    spec = load_spec(ROOT / "models" / DEEPSEEK, CLUSTER)
    decode = spec.role("decode")
    decode.lws.size = 4
    decode.lws.replicas = 1
    decode.parallelism.tp = 8
    decode.parallelism.dp = 2

    objects = render(spec, user="tester", cluster=CLUSTER)
    workload = _find(objects, "LeaderWorkerSet", "decode")
    assert workload["spec"]["replicas"] == 1
    assert workload["spec"]["leaderWorkerTemplate"]["size"] == 4

    pod_spec = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    container = next(item for item in pod_spec["containers"] if item["name"] == "vllm")
    assert [port["containerPort"] for port in container["ports"]] == [8200, 5555]

    script = container["args"][0]
    assert "DP_SIZE_LOCAL=1" in script
    assert "DP_SIZE=2" in script
    assert "TP_NODES=2" in script
    assert "LWS_WORKER_INDEX % TP_NODES != 0" in script
    assert "LWS_GROUP_INDEX" not in script
    assert "--tensor-parallel-size 8" in script
    assert "--nnodes 4" in script
    assert "--node-rank $LWS_WORKER_INDEX" in script
    assert "--data-parallel-size $DP_SIZE" in script
    assert "--data-parallel-rank" not in script
    assert "--data-parallel-size-local 1" in script
    assert '--data-parallel-address "${LWS_LEADER_ADDRESS}"' in script
    assert "--data-parallel-rpc-port 5555" in script
    assert "--data-parallel-external-lb" in script
    assert "--data-parallel-multi-port-external-lb" not in script

    readiness = container["readinessProbe"]["exec"]["command"][-1]
    assert "LWS_WORKER_INDEX:-0} % 2 != 0" in readiness

    service = _find(objects, "Service", "decode-svc")
    assert service["spec"]["selector"]["leaderworkerset.sigs.k8s.io/worker-index"] == "0"

    configmap = _find(objects, "ConfigMap", "epp-config")
    config = yaml.safe_load(configmap["data"]["plugins.yaml"])
    leader_filter = next(
        plugin
        for plugin in config["plugins"]
        if plugin.get("name") == "manifesto-decode-api-server-filter"
    )
    assert leader_filter["parameters"]["validValues"] == ["0", "2"]


def test_routing_disabled_dp2_tp8_uses_internal_vllm_load_balancing():
    spec = load_spec(ROOT / "models" / "qwen" / "qwen3-0.6b.yaml", CLUSTER)
    decode = spec.role("decode")
    decode.lws.size = 4
    decode.parallelism.tp = 8
    decode.parallelism.dp = 2
    decode.parallelism.gpus = 4
    decode.resources.gpus = 4

    objects = render(spec, user="tester", cluster=CLUSTER)
    workload = _find(objects, "LeaderWorkerSet", "decode")
    container = workload["spec"]["leaderWorkerTemplate"]["workerTemplate"][
        "spec"
    ]["containers"][0]
    script = container["args"][0]

    assert workload["spec"]["leaderWorkerTemplate"]["size"] == 4
    assert "--tensor-parallel-size 8" in script
    assert "--nnodes 4" in script
    assert "--node-rank $LWS_WORKER_INDEX" in script
    assert "--data-parallel-size $DP_SIZE" in script
    assert "--data-parallel-size-local 1" in script
    assert "--data-parallel-external-lb" not in script
    assert "--data-parallel-rank" not in script
    assert 'if [ "$LWS_WORKER_INDEX" -gt 0 ]; then' in script
    assert "LWS_WORKER_INDEX % TP_NODES" not in script

    readiness = container["readinessProbe"]["exec"]["command"][-1]
    assert 'if [ "${LWS_WORKER_INDEX:-0}" -gt 0 ]' in readiness
    service = _find(objects, "Service", "decode-svc")
    assert service["spec"]["selector"]["leaderworkerset.sigs.k8s.io/worker-index"] == "0"
    assert not any(obj["kind"] == "InferencePool" for obj in objects)


def test_cross_node_tp_custom_epp_render_is_repeatable_and_non_mutating():
    spec = load_spec(ROOT / "models" / DEEPSEEK, CLUSTER)
    base_objects = render(spec, user="tester", cluster=CLUSTER)
    base_configmap = _find(base_objects, "ConfigMap", "epp-config")
    selected_config = yaml.safe_load(base_configmap["data"]["plugins.yaml"])
    unselected_config = {"kind": "EndpointPickerConfig", "plugins": []}
    spec.routing.epp = EppSpec(
        plugins_config_file="selected.yaml",
        plugin_configs={
            "unselected.yaml": unselected_config,
            "selected.yaml": selected_config,
        },
    )
    original_configs = deepcopy(spec.routing.epp.plugin_configs)

    decode = spec.role("decode")
    decode.lws.size = 2
    decode.parallelism.tp = 8
    decode.parallelism.dp = False
    first_objects = render(spec, user="tester", cluster=CLUSTER)
    first_configmap = _find(first_objects, "ConfigMap", "epp-config")
    first_selected = yaml.safe_load(first_configmap["data"]["selected.yaml"])
    assert yaml.safe_load(first_configmap["data"]["unselected.yaml"]) == unselected_config
    assert spec.routing.epp.plugin_configs == original_configs
    first_filter = next(
        plugin
        for plugin in first_selected["plugins"]
        if plugin.get("name") == "manifesto-decode-api-server-filter"
    )
    assert first_filter["parameters"]["validValues"] == ["0"]

    decode.lws.size = 4
    decode.parallelism.dp = 2
    second_objects = render(spec, user="tester", cluster=CLUSTER)
    second_configmap = _find(second_objects, "ConfigMap", "epp-config")
    second_selected = yaml.safe_load(second_configmap["data"]["selected.yaml"])
    second_filter = next(
        plugin
        for plugin in second_selected["plugins"]
        if plugin.get("name") == "manifesto-decode-api-server-filter"
    )
    assert second_filter["parameters"]["validValues"] == ["0", "2"]
    assert spec.routing.epp.plugin_configs == original_configs


def test_non_pd_inferencepool_selector_targets_decode_role():
    objects = _objects("qwen/aggregated.yaml")
    infpool = _find(objects, "InferencePool")

    selector = infpool["spec"]["selector"]["matchLabels"]
    assert selector["app.kubernetes.io/instance"] == "qwen"
    assert selector["llm-d.ai/role"] == "decode"


def test_inferencepool_references_epp_service():
    objects = _objects(DEEPSEEK)
    service = _find(objects, "Service", "infpool-epp")

    assert service["spec"]["selector"]["app.kubernetes.io/component"] == "epp"
    assert service["spec"]["ports"][0]["targetPort"] == 9002


def test_epp_uses_current_config_file_flag():
    objects = _objects(DEEPSEEK)
    deployment = _find(objects, "Deployment", "infpool-epp")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    args = container["args"]

    assert container["image"] == DEFAULT_IMAGES.get("llm_d.epp", release=DEFAULT_IMAGES.get("llm_d.release"))
    assert "--config-file=/etc/epp/plugins.yaml" in args
    assert "--pool-name=wide-ep-1p-ep8-1d-ep8-infpool" in args
    assert "--pool-namespace=default" in args
    assert not any(arg.startswith("--plugins-config-file") for arg in args)


def test_wide_ep_uses_kv_aware_epp_scheduling():
    objects = _objects_with_routing_profile(DEEPSEEK, "wide-ep-lws-config")
    config_map = _find(objects, "ConfigMap", "epp-config")
    deployment = _find(objects, "Deployment", "infpool-epp")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert list(config_map["data"]) == ["wide-ep-lws-config.yaml"]
    config = yaml.safe_load(config_map["data"]["wide-ep-lws-config.yaml"])
    plugins = {(plugin["type"], plugin.get("name")) for plugin in config["plugins"]}
    assert ("approx-prefix-cache-producer", "gpu-prefix-cache-producer") in plugins
    assert ("approx-prefix-cache-producer", "cpu-prefix-cache-producer") in plugins
    assert ("prefix-cache-scorer", "gpu-prefix-cache-scorer") in plugins
    assert ("prefix-cache-scorer", "cpu-prefix-cache-scorer") in plugins
    assert config["schedulingProfiles"][0]["plugins"] == [
        {"pluginRef": "prefill-filter"},
        {"pluginRef": "gpu-prefix-cache-scorer", "weight": 5},
        {"pluginRef": "cpu-prefix-cache-scorer", "weight": 2},
        {"pluginRef": "active-request-scorer", "weight": 1},
    ]
    assert container["volumeMounts"] == [
        {
            "name": "config",
            "mountPath": "/etc/epp/wide-ep-lws-config.yaml",
            "subPath": "wide-ep-lws-config.yaml",
        }
    ]


def test_epp_uses_dedicated_service_account_and_rbac():
    objects = _objects(DEEPSEEK)
    service_account = _find(objects, "ServiceAccount", "infpool-epp")
    role = _find(objects, "Role", "infpool-epp-rbac")
    binding = _find(objects, "RoleBinding", "infpool-epp-rbac")
    deployment = _find(objects, "Deployment", "infpool-epp")

    assert deployment["spec"]["template"]["spec"]["serviceAccountName"] == service_account["metadata"]["name"]
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": service_account["metadata"]["name"], "namespace": "default"}
    ]
    assert binding["roleRef"]["name"] == role["metadata"]["name"]

    rules = {(tuple(rule["apiGroups"]), tuple(rule["resources"])): rule["verbs"] for rule in role["rules"]}
    assert rules[(("",), ("pods",))] == ["get", "list", "watch"]
    assert rules[(("inference.networking.k8s.io",), ("inferencepools",))] == ["get", "list", "watch"]
    assert rules[
        (
            ("inference.networking.x-k8s.io",),
            ("inferencemodelrewrites", "inferencemodels", "inferenceobjectives", "inferencepoolimports"),
        )
    ] == ["get", "list", "watch"]


def test_example_h200_cluster_uses_generic_cache_and_rdma_settings():
    spec = load_spec(ROOT / "models" / "qwen" / "h200-aggregated.yaml", EXAMPLE_H200)
    assert spec.model.hf_home == "/var/cache/huggingface"
    objects = render(spec, user="tester", cluster=EXAMPLE_H200)
    deployment = _find(objects, "Deployment", "decode")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    script = container["args"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}

    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    mounts = {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}

    assert volumes["hf-cache"]["hostPath"]["path"] == "/var/cache/manifesto/huggingface"
    assert volumes["jit-cache"]["hostPath"]["path"] == "/var/cache/manifesto/jit"
    assert mounts["hf-cache"] == "/var/cache/huggingface"
    assert mounts["jit-cache"] == "/var/cache/vllm"
    assert "NCCL_IB_HCA" not in env
    assert "NVSHMEM_HCA_PREFIX" not in env
    assert env["HF_HUB_CACHE"] == "/var/cache/huggingface"
    assert env["FLASHINFER_WORKSPACE_BASE"] == "/var/cache/manifesto/flashinfer"
    assert "MAX_TOKENS" not in env
    assert "--max-num-batched-tokens" not in script
    assert "--max-num-seqs" not in script
    assert "--max-cudagraph-capture-size" not in script
    assert container["resources"]["requests"]["example.com/rdma"] == "1"
    assert container["resources"]["limits"]["example.com/rdma"] == "1"


def test_lws_uses_cluster_routing_sidecar_image():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    init_container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["initContainers"][0]

    assert init_container["image"] == DEFAULT_IMAGES.get(
        "llm_d.routing_sidecar",
        release=DEFAULT_IMAGES.get("llm_d.release"),
    )


def test_routing_plugin_config_can_be_inline_override():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    spec.routing.plugin_config = {
        "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
        "kind": "EndpointPickerConfig",
        "plugins": [{"type": "weighted-random-picker", "name": "custom-picker"}],
    }
    objects = render(spec, user="tester", cluster=CLUSTER)
    config = _find(objects, "ConfigMap", "epp-config")

    assert "custom-picker" in config["data"]["plugins.yaml"]
    assert "active-request-scorer" not in config["data"]["plugins.yaml"]


def test_routing_epp_can_select_one_of_multiple_plugin_configs():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    spec.routing.epp = EppSpec(
        replicas=2,
        plugins_config_file="kv-aware.yaml",
        plugin_configs={
            "default.yaml": {"kind": "EndpointPickerConfig", "plugins": []},
            "kv-aware.yaml": {
                "kind": "EndpointPickerConfig",
                "plugins": [{"type": "prefix-cache-scorer"}],
            },
        },
    )
    objects = render(spec, user="tester", cluster=CLUSTER)
    config = _find(objects, "ConfigMap", "epp-config")
    deployment = _find(objects, "Deployment", "infpool-epp")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert set(config["data"]) == {"default.yaml", "kv-aware.yaml"}
    assert deployment["spec"]["replicas"] == 2
    assert "--config-file=/etc/epp/kv-aware.yaml" in container["args"]


def test_prefill_launch_uses_global_tp_and_local_gpu_span():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "prefill")
    container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]
    script = container["args"][0]

    assert "--tensor-parallel-size 1" in script
    assert "DP_SIZE=8" in script
    assert "--data-parallel-multi-port-external-lb" in script
    assert "--data-parallel-start-rank $START_RANK" in script
    assert "--data-parallel-rank" not in script


def test_routing_disabled_still_emits_model_server_service():
    objects = _objects("qwen/qwen3-0.6b.yaml")
    service = _find(objects, "Service", "decode-svc")

    assert service["spec"]["selector"]["llm-d.ai/role"] == "decode"
    assert service["spec"]["ports"] == [
        {"name": "vllm-0", "port": 8000, "targetPort": 8000}
    ]
    assert not any(obj["kind"] == "InferencePool" for obj in objects)


def test_pd_spec_emits_service_for_each_role():
    objects = _objects(DEEPSEEK)
    decode_svc = _find(objects, "Service", "decode-svc")
    prefill_svc = _find(objects, "Service", "prefill-svc")

    assert decode_svc["spec"]["selector"]["llm-d.ai/role"] == "decode"
    assert prefill_svc["spec"]["selector"]["llm-d.ai/role"] == "prefill"
    assert len(decode_svc["spec"]["ports"]) > 0
    assert len(prefill_svc["spec"]["ports"]) > 0


def test_deepseek_v4_nested_attention_config_preserves_official_flag_spelling():
    objects = _objects(DEEPSEEK)
    for role in ("decode", "prefill"):
        lws = _find(objects, "LeaderWorkerSet", role)
        script = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]["args"][0]

        assert "--attention_config.use_fp4_indexer_cache=True" in script
        assert "attention-config.use-fp4-indexer-cache" not in script
