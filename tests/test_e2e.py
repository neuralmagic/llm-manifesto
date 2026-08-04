"""Regression tests for the fresh-namespace in-cluster workflow."""

import json
from pathlib import Path

import pytest
import yaml

from manifesto.cli import main
from manifesto.cluster import load_cluster
from manifesto.instance import Instance
import manifesto.e2e as e2e_workflow
import manifesto.workflow as workflow


ROOT = Path(__file__).resolve().parents[1]
ROUTED_MODEL = ROOT / "models" / "qwen" / "aggregated.yaml"
DIRECT_MODEL = ROOT / "models" / "qwen" / "qwen3-0.6b.yaml"
CLUSTER = ROOT / "clusters" / "example-gb200.yaml"


def test_e2e_runs_in_cluster_job_against_routed_gateway(monkeypatch):
    calls = []
    lifecycle = []

    def fake_run(cmd, *, input_text=None):
        calls.append((cmd, input_text))
        return 0

    def fake_capture(cmd, **_kwargs):
        if cmd[:3] == ["kubectl", "get", "namespace/workload-ns"]:
            return ""
        if "job/tester-qwen-e2e" in cmd:
            return json.dumps({"status": {"succeeded": 1}})
        raise AssertionError(cmd)

    def fake_deploy(args, *, config, cluster):
        lifecycle.append("deploy")
        rendered = workflow.render_manifest(args, config, cluster=cluster)
        assert "persistentVolumeClaim" not in rendered
        assert "emptyDir" in rendered
        return 0

    monkeypatch.setattr(workflow, "run", fake_run)
    monkeypatch.setattr(workflow, "capture", fake_capture)
    monkeypatch.setattr(workflow, "deploy", fake_deploy)
    monkeypatch.setattr(
        workflow,
        "ready",
        lambda _args, **_kwargs: lifecycle.append("ready") or 0,
    )

    rc = main(
        [
            "test",
            "e2e",
            str(ROUTED_MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "workload-ns",
            "--user",
            "tester",
            "--image",
            "registry.test/python:3.12",
        ]
    )

    assert rc == 0
    assert lifecycle == ["deploy", "ready"]
    assert calls[0][0] == ["kubectl", "create", "namespace", "workload-ns"]
    create_cmd, manifest = calls[1]
    assert create_cmd == ["kubectl", "-n", "workload-ns", "create", "-f", "-"]
    job = yaml.safe_load(manifest)
    assert job["kind"] == "Job"
    assert job["spec"]["backoffLimit"] == 0
    pod_spec = job["spec"]["template"]["spec"]
    assert pod_spec["automountServiceAccountToken"] is False
    assert pod_spec["securityContext"]["runAsUser"] == 65532
    container = pod_spec["containers"][0]
    assert container["image"] == "registry.test/python:3.12"
    assert container["env"] == [
        {
            "name": "MANIFESTO_E2E_URL",
            "value": (
                "http://tester-qwen-gateway-istio.workload-ns.svc.cluster.local:80/v1"
            ),
        },
        {"name": "MANIFESTO_E2E_TIMEOUT", "value": "300"},
    ]
    assert "/models" in container["command"][2]
    assert "/completions" in container["command"][2]
    assert "except urllib.error.URLError" in container["command"][2]
    assert calls[2][0][-2:] == ["logs", "job/tester-qwen-e2e"]
    assert calls[3][0][-3:] == [
        "delete",
        "job/tester-qwen-e2e",
        "--ignore-not-found=true",
    ]
    assert calls[4][0] == [
        "kubectl",
        "delete",
        "namespace/workload-ns",
        "--wait=true",
        "--timeout=120s",
    ]


def test_e2e_can_use_external_env_with_cluster_storage_and_keep_namespace(monkeypatch):
    calls = []
    lifecycle = []
    monkeypatch.setattr(
        workflow,
        "run",
        lambda cmd, *, input_text=None: calls.append((cmd, input_text)) or 0,
    )
    monkeypatch.setattr(
        workflow,
        "capture",
        lambda cmd, **_kwargs: (
            "" if "namespace/workload-ns" in cmd else json.dumps({"status": {"succeeded": 1}})
        ),
    )
    def fake_deploy(args, *, config, cluster):
        lifecycle.append("deploy")
        rendered = workflow.render_manifest(args, config, cluster=cluster)
        assert "persistentVolumeClaim" in rendered
        assert "name: MANIFESTO_VLLM_ENV" in rendered
        assert "value: /mnt/shared/tester/vllm-envs/feature" in rendered
        return 0

    monkeypatch.setattr(workflow, "deploy", fake_deploy)
    monkeypatch.setattr(workflow, "ready", lambda _args, **_kwargs: 0)

    rc = main(
        [
            "test",
            "e2e",
            str(DIRECT_MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "workload-ns",
            "--user",
            "tester",
            "--vllm-env",
            "/mnt/shared/tester/vllm-envs/feature",
            "--keep-namespace",
        ]
    )

    assert rc == 0
    assert lifecycle == ["deploy"]
    job = yaml.safe_load(calls[1][1])
    assert job["spec"]["template"]["spec"]["containers"][0]["env"][0]["value"] == (
        "http://tester-qwen3-0-6b-decode-svc.workload-ns.svc.cluster.local:8000/v1"
    )
    assert not any("namespace/workload-ns" in cmd for cmd, _ in calls[2:])


def test_e2e_refuses_to_reuse_an_existing_namespace(monkeypatch, capsys):
    monkeypatch.setattr(
        workflow,
        "capture",
        lambda cmd, **_kwargs: (
            "namespace/workload-ns\n" if "namespace/workload-ns" in cmd else ""
        ),
    )
    monkeypatch.setattr(
        workflow,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not mutate")),
    )

    rc = main(
        [
            "test",
            "e2e",
            str(ROUTED_MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "workload-ns",
            "--user",
            "tester",
        ]
    )

    assert rc == 2
    assert "already exists; e2e requires a fresh namespace" in capsys.readouterr().err


def test_e2e_job_lets_openshift_assign_namespace_uid():
    cluster = load_cluster(CLUSTER)
    cluster.platform = "openshift"
    job = e2e_workflow.render_probe_job(
        name="test-e2e",
        instance=Instance(user="tester", release="model"),
        url="http://model:8000/v1",
        image="python:3.12-alpine",
        timeout=300,
        cluster=cluster,
    )

    security_context = job["spec"]["template"]["spec"]["securityContext"]
    assert security_context == {
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }


def test_e2e_probe_inherits_cluster_placement_and_dns():
    cluster = load_cluster(CLUSTER)
    cluster.pod_defaults.affinity = {"nodeAffinity": {"required": "gpu-workers"}}
    cluster.pod_defaults.tolerations = [{"key": "gpu", "operator": "Exists"}]
    cluster.pod_defaults.dns_policy = "ClusterFirst"
    cluster.pod_defaults.dns_config = {"options": [{"name": "ndots", "value": "2"}]}

    job = e2e_workflow.render_probe_job(
        name="test-e2e",
        instance=Instance(user="tester", release="model"),
        url="http://model:8000/v1",
        image="python:3.12-alpine",
        timeout=300,
        cluster=cluster,
    )

    pod_spec = job["spec"]["template"]["spec"]
    assert pod_spec["affinity"] == cluster.pod_defaults.affinity
    assert pod_spec["tolerations"] == cluster.pod_defaults.tolerations
    assert pod_spec["dnsPolicy"] == "ClusterFirst"
    assert pod_spec["dnsConfig"] == cluster.pod_defaults.dns_config


def test_e2e_rejects_pvc_extra_volumes():
    cluster = load_cluster(CLUSTER)
    cluster.pod_defaults.extra_volumes = [
        {"name": "private-data", "persistentVolumeClaim": {"claimName": "private-data"}}
    ]

    with pytest.raises(workflow.WorkflowError, match="extra_volumes PVCs: private-data"):
        e2e_workflow.ephemeral_cluster(cluster)


def test_e2e_detects_job_admission_failure_without_waiting(monkeypatch):
    config = workflow.RuntimeConfig(
        user="tester",
        namespace="workload-ns",
        cluster_path=None,
        render_out=Path("/tmp/tester-manifesto.yaml"),
    )

    def fake_capture(cmd, **_kwargs):
        if "job/test-e2e" in cmd:
            return json.dumps({"status": {}})
        if "events" in cmd:
            return json.dumps(
                {
                    "items": [
                        {
                            "type": "Warning",
                            "reason": "FailedCreate",
                            "message": "pod rejected by policy",
                        }
                    ]
                }
            )
        raise AssertionError(cmd)

    monkeypatch.setattr(workflow, "capture", fake_capture)

    assert e2e_workflow._job_state(config, "test-e2e") == (
        "failed-create",
        "pod rejected by policy",
    )
