"""Public cluster projection shared with non-serving GPU workloads."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from manifesto.cli import main
from manifesto.cluster import AcceleratorConfig, load_cluster
from manifesto.workload import (
    DeploymentPolicy,
    JobPolicy,
    LeaderWorkerSetPolicy,
    PodTemplate,
    ServicePort,
    Workload,
    WorkloadBackend,
    WorkloadMetadata,
    WorkloadService,
    render_workload,
    workload_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def test_workload_settings_project_only_portable_cluster_policy():
    cluster = load_cluster(ROOT / "clusters" / "example-gb200.yaml")
    cluster.kueue.local_queue = "benchmark-queue"
    cluster.pod_defaults.annotations = {"example.com/pool": "gpu"}
    cluster.pod_defaults.tolerations = [{"key": "gpu", "operator": "Exists"}]
    cluster.pod_defaults.image_pull_secrets = ["example-registry-credentials"]
    cluster.accelerators.profiles["gb200"].node_selector["gpu.product"] = "GB200"

    settings = workload_settings(cluster)

    assert settings.local_queue == "benchmark-queue"
    assert settings.accelerator().resource_name == "nvidia.com/gpu"
    assert settings.accelerator("any").node_selector == {"gpu.product": "GB200"}
    assert settings.pod.annotations == {"example.com/pool": "gpu"}
    assert settings.pod.tolerations == [{"key": "gpu", "operator": "Exists"}]
    assert settings.pod.image_pull_secrets == ["example-registry-credentials"]
    assert "storage" not in settings.model_dump()
    assert "fabric" not in settings.model_dump()


def test_workload_settings_reject_unknown_accelerator():
    settings = workload_settings(load_cluster(ROOT / "clusters" / "example-gb200.yaml"))
    with pytest.raises(ValueError, match="unknown accelerator"):
        settings.accelerator("quantum-gpu")


def test_accelerator_resource_must_be_an_extended_resource():
    with pytest.raises(ValidationError, match="Kubernetes extended resource"):
        AcceleratorConfig(
            resource_name="gpu",
            presence_label="example.com/gpu.present",
            gpu_arch="test",
            torch_cuda_arch_list="10.0",
        )


def test_job_renderer_owns_queue_placement_and_optional_headless_service():
    cluster = load_cluster(ROOT / "clusters" / "example-gb200.yaml")
    cluster.pod_defaults.annotations = {"example.com/pool": "gpu"}
    cluster.pod_defaults.tolerations = [{"key": "gpu", "operator": "Exists"}]
    cluster.pod_defaults.image_pull_secrets = ["example-registry-credentials"]
    cluster.accelerators.profiles["gb200"].node_selector["gpu.product"] = "GB200"
    workload = Workload(
        name="benchmark-run",
        backend=WorkloadBackend.JOB,
        metadata=WorkloadMetadata(labels={"app": "benchmark"}),
        pod_template=PodTemplate(
            metadata=WorkloadMetadata(labels={"app": "benchmark", "run": "one"}),
            spec={
                "restartPolicy": "Never",
                "imagePullSecrets": [{"name": "workload-registry-credentials"}],
                "containers": [{"name": "worker", "image": "worker:test"}],
            },
        ),
        accelerator_count=2,
        accelerator_container="worker",
        queue_name="benchmark-queue",
        workload_priority_class="batch",
        job=JobPolicy(
            suspend=True,
            backoff_limit=0,
            active_deadline_seconds=600,
            ttl_seconds_after_finished=3600,
        ),
        service=WorkloadService(
            name="benchmark-run-headless",
            selector={"run": "one"},
            headless=True,
            publish_not_ready_addresses=True,
            ports=[ServicePort(name="control", port=29500, target_port=29500)],
        ),
    )

    service, job = render_workload(
        workload,
        settings=workload_settings(cluster),
        accelerator="gb200",
    )

    assert service["kind"] == "Service"
    assert service["spec"]["clusterIP"] == "None"
    assert service["spec"]["publishNotReadyAddresses"] is True
    assert job["kind"] == "Job"
    assert job["spec"]["suspend"] is True
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["activeDeadlineSeconds"] == 600
    assert job["spec"]["ttlSecondsAfterFinished"] == 3600
    assert "completionMode" not in job["spec"]
    assert "completions" not in job["spec"]
    assert "parallelism" not in job["spec"]
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == ("benchmark-queue")
    pod_template = job["spec"]["template"]
    assert (
        pod_template["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "batch"
    )
    assert pod_template["metadata"]["annotations"] == {"example.com/pool": "gpu"}
    assert pod_template["spec"]["nodeSelector"] == {"gpu.product": "GB200"}
    assert pod_template["spec"]["tolerations"] == [{"key": "gpu", "operator": "Exists"}]
    assert pod_template["spec"]["imagePullSecrets"] == [
        {"name": "example-registry-credentials"},
        {"name": "workload-registry-credentials"},
    ]
    assert pod_template["spec"]["containers"][0]["resources"] == {
        "requests": {"nvidia.com/gpu": "2"},
        "limits": {"nvidia.com/gpu": "2"},
    }


@pytest.mark.parametrize(
    ("backend", "policy", "kind"),
    [
        (WorkloadBackend.DEPLOYMENT, DeploymentPolicy(replicas=2), "Deployment"),
        (
            WorkloadBackend.LEADER_WORKER_SET,
            LeaderWorkerSetPolicy(replicas=2, size=4),
            "LeaderWorkerSet",
        ),
    ],
)
def test_shared_ir_dispatches_controller_backends(backend, policy, kind):
    kwargs = {
        "deployment": policy if backend == WorkloadBackend.DEPLOYMENT else None,
        "leader_worker_set": (
            policy if backend == WorkloadBackend.LEADER_WORKER_SET else None
        ),
    }
    workload = Workload(
        name="model",
        backend=backend,
        selector={"app": "model"},
        pod_template=PodTemplate(
            metadata=WorkloadMetadata(labels={"app": "model"}),
            spec={"containers": [{"name": "model", "image": "model:test"}]},
        ),
        **kwargs,
    )
    assert render_workload(workload)[0]["kind"] == kind


def test_render_workload_cli_emits_exact_non_indexed_job(tmp_path, capsys):
    source = tmp_path / "benchmark-job.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "name": "benchmark-run",
                "backend": "job",
                "metadata": {"labels": {"app": "benchmark"}},
                "pod_template": {
                    "metadata": {"labels": {"app": "benchmark"}},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{"name": "worker", "image": "worker:test"}],
                    },
                },
                "queue_name": "benchmark-queue",
                "accelerator_count": 1,
                "accelerator_container": "worker",
                "job": {
                    "suspend": True,
                    "backoff_limit": 0,
                    "active_deadline_seconds": 600,
                },
            }
        )
    )

    assert (
        main(
            [
                "render",
                "workload",
                str(source),
                "--cluster",
                str(ROOT / "clusters" / "example-gb200.yaml"),
            ]
        )
        == 0
    )
    (job,) = yaml.safe_load_all(capsys.readouterr().out)
    assert job["kind"] == "Job"
    assert job["spec"]["suspend"] is True
    assert "completionMode" not in job["spec"]
    assert job["spec"]["template"]["spec"]["containers"][0]["resources"] == {
        "requests": {"nvidia.com/gpu": "1"},
        "limits": {"nvidia.com/gpu": "1"},
    }
