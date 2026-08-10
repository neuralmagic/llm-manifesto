"""Kueue admission metadata for GPU serving workloads."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from manifesto.cluster import KueueConfig, load_cluster
from manifesto.e2e import render_probe_job
from manifesto.instance import Instance
from manifesto.render import render
from manifesto.render.lws import KUEUE_QUEUE_LABEL
from manifesto.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_PATH = ROOT / "clusters" / "example-gb200.yaml"
QUEUE = "example-gpu-queue"


def _render(model: str, *, queue: str | None) -> list[dict]:
    cluster = load_cluster(CLUSTER_PATH)
    cluster.kueue.local_queue = queue
    spec = load_spec(ROOT / "models" / model, cluster)
    return render(spec, user="tester", cluster=cluster)


def _lws_pod_spec(workload: dict) -> dict:
    return workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def _find(objects: list[dict], kind: str, name_suffix: str) -> dict:
    return next(
        obj
        for obj in objects
        if obj["kind"] == kind and obj["metadata"]["name"].endswith(name_suffix)
    )


def test_aggregate_lws_uses_declarative_local_queue_without_resource_changes():
    model = "kimi-k3/aggregated-tp16-ep16.yaml"
    queued = _render(model, queue=QUEUE)
    unqueued = _render(model, queue=None)
    queued_lws = next(obj for obj in queued if obj["kind"] == "LeaderWorkerSet")
    unqueued_lws = next(obj for obj in unqueued if obj["kind"] == "LeaderWorkerSet")

    assert queued_lws["metadata"]["labels"][KUEUE_QUEUE_LABEL] == QUEUE
    assert (
        KUEUE_QUEUE_LABEL
        not in queued_lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["metadata"][
            "labels"
        ]
    )
    assert KUEUE_QUEUE_LABEL not in unqueued_lws["metadata"]["labels"]
    assert _lws_pod_spec(queued_lws) == _lws_pod_spec(unqueued_lws)

    containers = _lws_pod_spec(queued_lws)["containers"]
    vllm = next(container for container in containers if container["name"] == "vllm")
    assert vllm["resources"] == {
        "requests": {
            "cpu": "14",
            "memory": "224Gi",
            "nvidia.com/gpu": "4",
            "ephemeral-storage": "128Gi",
        },
        "limits": {
            "memory": "224Gi",
            "nvidia.com/gpu": "4",
            "ephemeral-storage": "128Gi",
        },
    }
    assert all("resources" in container for container in containers)


def test_pd_lws_roles_share_queue_and_preserve_complete_pod_specs():
    model = "deepseek-v4/1P-EP8-1D-EP8.yaml"
    queued = _render(model, queue=QUEUE)
    unqueued = _render(model, queue=None)
    queued_lws = {
        obj["metadata"]["labels"]["llm-d.ai/role"]: obj
        for obj in queued
        if obj["kind"] == "LeaderWorkerSet"
    }
    unqueued_lws = {
        obj["metadata"]["labels"]["llm-d.ai/role"]: obj
        for obj in unqueued
        if obj["kind"] == "LeaderWorkerSet"
    }

    assert queued_lws.keys() == {"prefill", "decode"}
    for role, workload in queued_lws.items():
        assert workload["metadata"]["labels"][KUEUE_QUEUE_LABEL] == QUEUE
        assert _lws_pod_spec(workload) == _lws_pod_spec(unqueued_lws[role])
        pod_spec = _lws_pod_spec(workload)
        assert all("resources" in container for container in pod_spec["containers"])
        assert all(
            "resources" in container for container in pod_spec.get("initContainers", [])
        )
    assert _lws_pod_spec(queued_lws["decode"])["initContainers"]


def test_single_node_aggregate_remains_deployment_with_independent_pod_admission():
    model = "qwen/aggregated.yaml"
    queued = _render(model, queue=QUEUE)
    unqueued = _render(model, queue=None)
    deployment = _find(queued, "Deployment", "decode")
    unqueued_deployment = _find(unqueued, "Deployment", "decode")

    assert deployment["metadata"]["labels"][KUEUE_QUEUE_LABEL] == QUEUE
    assert (
        deployment["spec"]["template"]["metadata"]["labels"][KUEUE_QUEUE_LABEL] == QUEUE
    )
    assert KUEUE_QUEUE_LABEL not in unqueued_deployment["metadata"]["labels"]
    assert (
        KUEUE_QUEUE_LABEL
        not in unqueued_deployment["spec"]["template"]["metadata"]["labels"]
    )
    assert (
        deployment["spec"]["template"]["spec"]
        == unqueued_deployment["spec"]["template"]["spec"]
    )
    assert (
        _find(queued, "Service", "decode-svc")["spec"]
        == _find(unqueued, "Service", "decode-svc")["spec"]
    )
    assert (
        _find(queued, "InferencePool", "infpool")["spec"]
        == _find(unqueued, "InferencePool", "infpool")["spec"]
    )

    controller = _find(queued, "Deployment", "idle-shutdown")
    env = {
        item["name"]: item
        for item in controller["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    workloads = json.loads(env["WORKLOADS"]["value"])
    model_workload = next(item for item in workloads if item["name"].endswith("decode"))
    assert "/deployments/" in model_workload["path"]
    role = _find(queued, "Role", "idle-shutdown-rbac")
    assert any(
        rule["apiGroups"] == ["apps"]
        and model_workload["name"] in rule["resourceNames"]
        for rule in role["rules"]
    )


def test_single_node_pd_roles_remain_independently_admitted_deployments():
    model = ROOT / "models" / "deepseek-v4/1P-EP8-1D-EP8.yaml"
    renders = {}
    for queue in (None, QUEUE):
        cluster = load_cluster(CLUSTER_PATH)
        cluster.kueue.local_queue = queue
        spec = load_spec(model, cluster)
        for role in spec.roles:
            role.lws.size = 1
            role.parallelism.dp = 4
        renders[queue] = render(spec, user="tester", cluster=cluster)

    queued = renders[QUEUE]
    unqueued = renders[None]
    for role in ("prefill", "decode"):
        deployment = _find(queued, "Deployment", role)
        unqueued_deployment = _find(unqueued, "Deployment", role)
        assert deployment["metadata"]["labels"][KUEUE_QUEUE_LABEL] == QUEUE
        assert (
            deployment["spec"]["template"]["metadata"]["labels"][KUEUE_QUEUE_LABEL]
            == QUEUE
        )
        assert (
            deployment["spec"]["template"]["spec"]
            == unqueued_deployment["spec"]["template"]["spec"]
        )
        assert (
            _find(queued, "Service", f"{role}-svc")["spec"]
            == _find(unqueued, "Service", f"{role}-svc")["spec"]
        )

    assert _find(queued, "Deployment", "decode")["spec"]["template"]["spec"][
        "initContainers"
    ]


def test_queue_metadata_is_omitted_from_non_lws_resources():
    objects = _render("deepseek-v4/1P-EP8-1D-EP8.yaml", queue=QUEUE)

    for obj in objects:
        if obj["kind"] == "LeaderWorkerSet":
            continue
        assert KUEUE_QUEUE_LABEL not in obj.get("metadata", {}).get("labels", {})
        template_labels = (
            obj.get("spec", {})
            .get("template", {})
            .get("metadata", {})
            .get("labels", {})
        )
        assert KUEUE_QUEUE_LABEL not in template_labels


def test_zero_gpu_native_lws_is_unqueued_and_omits_accelerator_resources():
    cluster = load_cluster(CLUSTER_PATH)
    cluster.kueue.local_queue = QUEUE
    spec = load_spec(ROOT / "models" / "kimi-k3/aggregated-tp16-ep16.yaml", cluster)
    spec.role("decode").resources.gpus = 0
    objects = render(spec, user="tester", cluster=cluster)
    lws = next(obj for obj in objects if obj["kind"] == "LeaderWorkerSet")
    vllm = next(
        container
        for container in _lws_pod_spec(lws)["containers"]
        if container["name"] == "vllm"
    )

    assert KUEUE_QUEUE_LABEL not in lws["metadata"]["labels"]
    assert "nvidia.com/gpu" not in vllm["resources"]["requests"]
    assert "nvidia.com/gpu" not in vllm["resources"]["limits"]


def test_zero_gpu_native_deployment_is_unqueued_and_omits_accelerator_resources():
    cluster = load_cluster(CLUSTER_PATH)
    cluster.kueue.local_queue = QUEUE
    spec = load_spec(ROOT / "models" / "qwen/aggregated.yaml", cluster)
    spec.role("decode").resources.gpus = 0
    objects = render(spec, user="tester", cluster=cluster)
    deployment = _find(objects, "Deployment", "decode")
    vllm = deployment["spec"]["template"]["spec"]["containers"][0]

    assert KUEUE_QUEUE_LABEL not in deployment["metadata"]["labels"]
    assert KUEUE_QUEUE_LABEL not in deployment["spec"]["template"]["metadata"]["labels"]
    assert "nvidia.com/gpu" not in vllm["resources"]["requests"]
    assert "nvidia.com/gpu" not in vllm["resources"]["limits"]


def test_cpu_only_probe_job_is_not_queued_or_suspended():
    cluster = load_cluster(CLUSTER_PATH)
    cluster.kueue.local_queue = QUEUE
    job = render_probe_job(
        name="example-probe",
        instance=Instance(user="tester", release="model"),
        url="http://model.example:8000/v1",
        image="python:3.12-alpine",
        timeout=300,
        cluster=cluster,
    )

    assert KUEUE_QUEUE_LABEL not in job["metadata"]["labels"]
    assert "suspend" not in job["spec"]
    assert KUEUE_QUEUE_LABEL not in job["spec"]["template"]["metadata"]["labels"]
    requests = job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
    assert requests == {"cpu": "50m", "memory": "64Mi"}


@pytest.mark.parametrize(
    "queue",
    [
        "",
        "-bad",
        "bad-",
        "contains/slash",
        "UPPERCASE",
        "under_score",
        "x" * 64,
    ],
)
def test_local_queue_must_be_a_kubernetes_label_value(queue):
    with pytest.raises(ValidationError, match="kueue.local_queue"):
        KueueConfig(local_queue=queue)
