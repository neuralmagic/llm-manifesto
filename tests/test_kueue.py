"""Kueue admission metadata for GPU serving workloads."""

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


def test_aggregate_lws_uses_declarative_local_queue_without_resource_changes():
    model = "kimi-k3/aggregated-tp16-ep16.yaml"
    queued = _render(model, queue=QUEUE)
    unqueued = _render(model, queue=None)
    queued_lws = next(obj for obj in queued if obj["kind"] == "LeaderWorkerSet")
    unqueued_lws = next(obj for obj in unqueued if obj["kind"] == "LeaderWorkerSet")

    assert queued_lws["metadata"]["labels"][KUEUE_QUEUE_LABEL] == QUEUE
    assert KUEUE_QUEUE_LABEL not in unqueued_lws["metadata"]["labels"]
    assert _lws_pod_spec(queued_lws) == _lws_pod_spec(unqueued_lws)

    containers = _lws_pod_spec(queued_lws)["containers"]
    vllm = next(container for container in containers if container["name"] == "vllm")
    assert vllm["resources"] == {
        "requests": {
            "cpu": "32",
            "memory": "512Gi",
            "nvidia.com/gpu": "4",
            "ephemeral-storage": "128Gi",
        },
        "limits": {
            "memory": "512Gi",
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
    ["", "-bad", "bad-", "contains/slash", "x" * 64],
)
def test_local_queue_must_be_a_kubernetes_label_value(queue):
    with pytest.raises(ValidationError, match="kueue.local_queue"):
        KueueConfig(local_queue=queue)
