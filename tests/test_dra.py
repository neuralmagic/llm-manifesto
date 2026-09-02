"""Swappable accelerator allocation through Kubernetes DRA."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from manifesto.cluster import AcceleratorConfig, Cluster
from manifesto.dra import DRA_CLAIM_NAME
from manifesto.instance import Instance
from manifesto.render import render
from manifesto.render.lws import render_workload
from manifesto.spec import load_spec
from manifesto.workload import (
    JobPolicy,
    PodTemplate,
    Workload,
    WorkloadBackend,
    WorkloadMetadata,
    render_workload as render_shared_workload,
    workload_settings,
)


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_PATH = ROOT / "clusters" / "example-gb200.yaml"


def _dra_cluster() -> Cluster:
    data = yaml.safe_load(CLUSTER_PATH.read_text())
    profile = data["accelerators"]["profiles"]["gb200"]
    profile["allocation"] = {
        "dra": {"device_class_name": "gpu.nvidia.com"}
    }
    return Cluster.model_validate(data)


def _pod_spec(workload: dict) -> dict:
    if workload["kind"] == "Deployment":
        return workload["spec"]["template"]["spec"]
    return workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def _assert_dra_pair(objects: list[dict], workload_kind: str, count: int) -> None:
    template = next(obj for obj in objects if obj["kind"] == "ResourceClaimTemplate")
    workload = next(obj for obj in objects if obj["kind"] == workload_kind)
    request = template["spec"]["spec"]["devices"]["requests"][0]
    assert request == {
        "name": "gpu",
        "exactly": {
            "deviceClassName": "gpu.nvidia.com",
            "allocationMode": "ExactCount",
            "count": count,
        },
    }
    assert len(template["metadata"]["name"]) <= 63

    pod_spec = _pod_spec(workload)
    assert pod_spec["resourceClaims"] == [
        {
            "name": DRA_CLAIM_NAME,
            "resourceClaimTemplateName": template["metadata"]["name"],
        }
    ]
    vllm = next(item for item in pod_spec["containers"] if item["name"] == "vllm")
    assert vllm["resources"]["claims"] == [{"name": DRA_CLAIM_NAME}]
    assert "nvidia.com/gpu" not in vllm["resources"]["requests"]
    assert "nvidia.com/gpu" not in vllm["resources"]["limits"]
    assert vllm["resources"]["requests"]["cpu"]
    assert vllm["resources"]["requests"]["memory"]


def test_accelerator_requires_exactly_one_allocation_backend():
    common = {
        "presence_label": "nvidia.com/gpu.present",
        "gpu_arch": "test",
        "torch_cuda_arch_list": "10.0",
    }
    with pytest.raises(ValidationError, match="exactly one"):
        AcceleratorConfig(**common, allocation={})
    with pytest.raises(ValidationError, match="exactly one"):
        AcceleratorConfig(
            **common,
            allocation={
                "extended_resource": {"resource_name": "nvidia.com/gpu"},
                "dra": {"device_class_name": "gpu.nvidia.com"},
            },
        )


@pytest.mark.parametrize(
    ("model", "workload_kind", "count"),
    [
        ("models/qwen/aggregated.yaml", "Deployment", 1),
        ("models/kimi-k3/aggregated-tp16-ep16.yaml", "LeaderWorkerSet", 4),
    ],
)
def test_serving_workloads_swap_extended_resources_for_dra_claims(
    model, workload_kind, count
):
    cluster = _dra_cluster()
    spec = load_spec(ROOT / model, cluster)
    _assert_dra_pair(
        render(spec, user="tester", cluster=cluster), workload_kind, count
    )


def test_zero_gpu_dra_role_emits_no_template_or_claim():
    cluster = _dra_cluster()
    spec = load_spec(ROOT / "models/qwen/aggregated.yaml", cluster)
    spec.role("decode").resources.gpus = 0
    objects = render(spec, user="tester", cluster=cluster)
    workload = next(
        obj
        for obj in objects
        if obj["kind"] == "Deployment"
        and obj["metadata"]["name"].endswith("decode")
    )
    assert not any(obj["kind"] == "ResourceClaimTemplate" for obj in objects)
    assert "resourceClaims" not in _pod_spec(workload)


def test_dra_accelerator_claim_coexists_with_imex_claim():
    cluster = _dra_cluster()
    cluster.fabric.imex_resource_claim_template = "compute-domain-template"
    spec = load_spec(ROOT / "models/qwen/aggregated.yaml", cluster)
    workload = render_workload(
        spec,
        Instance(user="tester", release=spec.release),
        cluster,
        spec.roles[0],
    )
    pod_spec = _pod_spec(workload)
    assert {claim["name"] for claim in pod_spec["resourceClaims"]} == {
        "compute-domain-channel",
        DRA_CLAIM_NAME,
    }
    assert {
        claim["name"]
        for claim in pod_spec["containers"][0]["resources"]["claims"]
    } == {"compute-domain-channel", DRA_CLAIM_NAME}


def test_reusable_workload_renderer_emits_dra_template_and_job():
    cluster = _dra_cluster()
    cluster.fabric.imex_resource_claim_template = "compute-domain-template"
    workload = Workload(
        name="benchmark",
        backend=WorkloadBackend.JOB,
        metadata=WorkloadMetadata(labels={"app": "benchmark"}),
        pod_template=PodTemplate(
            metadata=WorkloadMetadata(labels={"app": "benchmark"}),
            spec={
                "restartPolicy": "Never",
                "containers": [{"name": "worker", "image": "worker:test"}],
            },
        ),
        accelerator_count=2,
        accelerator_container="worker",
        job=JobPolicy(),
    )
    objects = render_shared_workload(
        workload,
        settings=workload_settings(cluster),
        accelerator="gb200",
    )
    assert [obj["kind"] for obj in objects] == ["ResourceClaimTemplate", "Job"]
    template, job = objects
    assert template["spec"]["spec"]["devices"]["requests"][0]["exactly"][
        "count"
    ] == 2
    assert job["spec"]["template"]["spec"]["containers"][0]["resources"] == {
        "claims": [
            {"name": "compute-domain-channel"},
            {"name": DRA_CLAIM_NAME},
        ]
    }
    assert job["spec"]["template"]["spec"]["resourceClaims"] == [
        {
            "name": "compute-domain-channel",
            "resourceClaimTemplateName": "compute-domain-template",
        },
        {
            "name": DRA_CLAIM_NAME,
            "resourceClaimTemplateName": template["metadata"]["name"],
        },
    ]


def test_immutable_template_name_changes_with_count():
    cluster = _dra_cluster()
    settings = workload_settings(cluster)

    def template_name(count: int) -> str:
        workload = Workload(
            name="x" * 80,
            backend=WorkloadBackend.JOB,
            pod_template=PodTemplate(
                spec={"containers": [{"name": "worker", "image": "worker:test"}]}
            ),
            accelerator_count=count,
            job=JobPolicy(),
        )
        return render_shared_workload(workload, settings=settings)[0]["metadata"][
            "name"
        ]

    one = template_name(1)
    two = template_name(2)
    assert one != two
    assert len(one) <= 63


def test_immutable_template_name_changes_with_claim_labels():
    cluster = _dra_cluster()
    settings = workload_settings(cluster)

    def template_name(release: str) -> str:
        workload = Workload(
            name="stable-workload-name",
            backend=WorkloadBackend.JOB,
            metadata=WorkloadMetadata(labels={"app": "benchmark", "release": release}),
            pod_template=PodTemplate(
                spec={"containers": [{"name": "worker", "image": "worker:test"}]}
            ),
            accelerator_count=1,
            job=JobPolicy(),
        )
        return render_shared_workload(workload, settings=settings)[0]["metadata"][
            "name"
        ]

    assert template_name("one") != template_name("two")
