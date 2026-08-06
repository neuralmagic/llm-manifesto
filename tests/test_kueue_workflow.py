"""Preflight and migration ordering for Kueue-managed model workloads."""

import json
from pathlib import Path

import pytest
import yaml

from manifesto import workflow

QUEUE = "gpu-queue"
CONFIG = workflow.RuntimeConfig(
    user="tester",
    namespace="workload-ns",
    cluster_path="cluster.yaml",
    render_out=Path("/tmp/manifest.yaml"),
)


def _lws(queue: str | None = QUEUE) -> dict:
    labels = {
        "app.kubernetes.io/name": "manifesto",
        "llm-d.ai/inferenceServing": "true",
    }
    if queue:
        labels[workflow.KUEUE_QUEUE_LABEL] = queue
    return {
        "apiVersion": "leaderworkerset.x-k8s.io/v1",
        "kind": "LeaderWorkerSet",
        "metadata": {"name": "tester-model", "labels": labels},
        "spec": {
            "replicas": 1,
            "leaderWorkerTemplate": {
                "size": 1,
                "workerTemplate": {
                    "spec": {
                        "initContainers": [
                            {
                                "name": "proxy",
                                "resources": {
                                    "requests": {"cpu": "1", "memory": "1Gi"}
                                },
                            }
                        ],
                        "containers": [
                            {
                                "name": "vllm",
                                "resources": {
                                    "requests": {
                                        "cpu": "8",
                                        "memory": "64Gi",
                                        "nvidia.com/gpu": "1",
                                    }
                                },
                            }
                        ],
                    }
                },
            },
        },
    }


def _deployment() -> dict:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "tester-model",
            "labels": {
                "app.kubernetes.io/name": "manifesto",
                "llm-d.ai/inferenceServing": "true",
            },
        },
        "spec": {},
    }


def _active() -> list[dict]:
    return [{"type": "Active", "status": "True"}]


def _install_fakes(
    monkeypatch,
    desired,
    *,
    deployment=None,
    lws=None,
    local_active=True,
    local_exists=True,
    cluster_active=True,
    kueue_apis=True,
    covered_resources=("cpu", "memory", "nvidia.com/gpu"),
):
    events = []
    manifest = yaml.safe_dump_all([desired])
    monkeypatch.setenv("HF_TOKEN", "hf_read_test")
    monkeypatch.setattr(workflow, "render_manifest", lambda *_args, **_kwargs: manifest)

    def fake_capture(cmd, **_kwargs):
        events.append(("capture", cmd))
        if "--api-group=leaderworkerset.x-k8s.io" in cmd:
            return workflow.LWS_RESOURCE_TYPE + "\n"
        if "--api-group=kueue.x-k8s.io" in cmd:
            return (
                "\n".join(sorted(workflow.KUEUE_REQUIRED_RESOURCE_TYPES)) + "\n"
                if kueue_apis
                else ""
            )
        if "localqueue" in cmd:
            if not local_exists:
                return ""
            return json.dumps(
                {
                    "spec": {"clusterQueue": "gpu-cluster-queue"},
                    "status": {"conditions": _active() if local_active else []},
                }
            )
        if "clusterqueue" in cmd:
            return json.dumps(
                {
                    "spec": {
                        "resourceGroups": [
                            {
                                "coveredResources": list(covered_resources),
                                "flavors": [
                                    {
                                        "name": "gpu-flavor",
                                        "resources": [
                                            {"name": "cpu", "nominalQuota": "100"},
                                            {"name": "memory", "nominalQuota": "1Ti"},
                                            {
                                                "name": "nvidia.com/gpu",
                                                "nominalQuota": "8",
                                            },
                                        ],
                                    }
                                ],
                            }
                        ]
                    },
                    "status": {"conditions": _active() if cluster_active else []},
                }
            )
        if "deployments.apps" in cmd:
            return json.dumps(deployment) if deployment else ""
        if workflow.LWS_RESOURCE_TYPE in cmd:
            return json.dumps(lws) if lws else ""
        raise AssertionError(cmd)

    def fake_run(cmd, *, input_text=None):
        events.append(("run", cmd, input_text))
        return 0

    monkeypatch.setattr(workflow, "capture", fake_capture)
    monkeypatch.setattr(workflow, "run", fake_run)
    return events


def test_deployment_to_lws_deletes_obsolete_kind_before_apply(monkeypatch):
    events = _install_fakes(monkeypatch, _lws(), deployment=_deployment())

    assert workflow.deploy(object(), config=CONFIG) == 0

    runs = [event for event in events if event[0] == "run"]
    assert runs[0][1][-3:] == ["apply", "-f", "-"]
    assert runs[1][1][-4:] == [
        "delete",
        "deployments.apps/tester-model",
        "--ignore-not-found=true",
        "--wait=true",
    ]
    assert runs[1][1][3] == "delete"
    assert runs[2][1][-3:] == ["apply", "-f", "-"]


def test_lws_to_deployment_deletes_obsolete_kind_before_apply(monkeypatch):
    events = _install_fakes(monkeypatch, _deployment(), lws=_lws())

    assert workflow.deploy(object(), config=CONFIG) == 0

    runs = [event for event in events if event[0] == "run"]
    assert runs[1][1][3:5] == [
        "delete",
        f"{workflow.LWS_RESOURCE_TYPE}/tester-model",
    ]
    assert runs[2][1][3:] == ["apply", "-f", "-"]


@pytest.mark.parametrize(
    ("live_queue", "desired_queue"),
    [("old-queue", QUEUE), (QUEUE, None)],
)
def test_lws_queue_change_recreates_before_apply(
    monkeypatch, live_queue, desired_queue
):
    events = _install_fakes(
        monkeypatch,
        _lws(desired_queue),
        lws=_lws(live_queue),
    )

    assert workflow.deploy(object(), config=CONFIG) == 0

    runs = [event for event in events if event[0] == "run"]
    assert runs[1][1][3] == "delete"
    assert runs[2][1][3] == "apply"


def test_inactive_local_queue_fails_before_any_mutation(monkeypatch):
    events = _install_fakes(monkeypatch, _lws(), local_active=False)

    with pytest.raises(workflow.WorkflowError, match="LocalQueue.*not Active=True"):
        workflow.deploy(object(), config=CONFIG)

    assert not any(event[0] == "run" for event in events)


def test_missing_kueue_apis_fail_before_any_mutation(monkeypatch):
    events = _install_fakes(monkeypatch, _lws(), kueue_apis=False)

    with pytest.raises(workflow.WorkflowError, match="required APIs are not served"):
        workflow.deploy(object(), config=CONFIG)

    assert not any(event[0] == "run" for event in events)


def test_missing_local_queue_fails_before_any_mutation(monkeypatch):
    events = _install_fakes(monkeypatch, _lws(), local_exists=False)

    with pytest.raises(workflow.WorkflowError, match="cannot read LocalQueue"):
        workflow.deploy(object(), config=CONFIG)

    assert not any(event[0] == "run" for event in events)


def test_missing_lws_api_fails_before_any_mutation(monkeypatch):
    events = _install_fakes(monkeypatch, _lws())

    def missing_api(cmd, **_kwargs):
        events.append(("capture", cmd))
        return ""

    monkeypatch.setattr(workflow, "capture", missing_api)
    with pytest.raises(workflow.WorkflowError, match="LeaderWorkerSet API"):
        workflow.deploy(object(), config=CONFIG)

    assert not any(event[0] == "run" for event in events)


def test_inactive_cluster_queue_fails_before_any_mutation(monkeypatch):
    events = _install_fakes(monkeypatch, _lws(), cluster_active=False)

    with pytest.raises(workflow.WorkflowError, match="ClusterQueue.*not Active=True"):
        workflow.deploy(object(), config=CONFIG)

    assert not any(event[0] == "run" for event in events)


def test_uncovered_podset_resource_fails_before_any_mutation(monkeypatch):
    events = _install_fakes(
        monkeypatch,
        _lws(),
        covered_resources=("cpu", "memory"),
    )

    with pytest.raises(
        workflow.WorkflowError,
        match=r"does not cover.*nvidia.com/gpu",
    ):
        workflow.deploy(object(), config=CONFIG)

    assert not any(event[0] == "run" for event in events)


def test_preflight_surfaces_all_podset_container_requests(monkeypatch, capsys):
    events = _install_fakes(monkeypatch, _lws())

    workflow.preflight_workloads(CONFIG, [_lws()])

    output = capsys.readouterr().err
    assert "replicas=1, pods-per-replica=1" in output
    assert "worker/initContainers/proxy: requests=cpu=1, memory=1Gi" in output
    assert (
        "worker/containers/vllm: requests=cpu=8, memory=64Gi, nvidia.com/gpu=1"
        in output
    )
    assert events
