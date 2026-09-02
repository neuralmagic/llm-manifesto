"""Portable cluster settings for GPU workloads outside Manifesto's renderer.

This module is the intentionally small integration surface for controllers
that need to schedule GPU work alongside Manifesto deployments without taking
on model topology, routing, or launch-script concerns.
"""

from __future__ import annotations

import copy
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .cluster import AcceleratorAllocationConfig, Cluster
from .dra import (
    accelerator_claim_template_name,
    attach_accelerator_claim,
    attach_resource_claim,
    render_accelerator_claim_template,
)

KUEUE_QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
KUEUE_PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"
_DNS_LABEL = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")


class WorkloadBackend(StrEnum):
    """Kubernetes workload controllers supported by the shared render IR."""

    JOB = "job"
    DEPLOYMENT = "deployment"
    LEADER_WORKER_SET = "leaderworkerset"
    GROVE = "grove"


class WorkloadMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class PodTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: WorkloadMetadata = Field(default_factory=WorkloadMetadata)
    spec: dict[str, Any]


class JobPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    suspend: bool = False
    backoff_limit: int = Field(default=0, ge=0)
    active_deadline_seconds: int | None = Field(default=None, ge=1)
    ttl_seconds_after_finished: int | None = Field(default=None, ge=0)
    completion_mode: Literal["NonIndexed", "Indexed"] | None = None
    completions: int | None = Field(default=None, ge=1)
    parallelism: int | None = Field(default=None, ge=1)
    pod_failure_policy: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_valid_completion_policy(self) -> JobPolicy:
        if self.completion_mode == "Indexed" and self.completions is None:
            raise ValueError("Indexed Jobs require completions")
        if self.pod_failure_policy:
            rules = self.pod_failure_policy.get("rules")
            if not isinstance(rules, list) or not rules:
                raise ValueError("pod_failure_policy requires at least one rule")
            matchers = {"onExitCodes", "onPodConditions", "onPodStatusPatterns"}
            actions = {"FailJob", "FailIndex", "Ignore", "Count"}
            for rule in rules:
                if (
                    not isinstance(rule, dict)
                    or rule.get("action") not in actions
                ):
                    raise ValueError(
                        "pod_failure_policy rules require a supported action"
                    )
                selected = matchers.intersection(rule)
                if len(selected) != 1:
                    raise ValueError(
                        "pod_failure_policy rules require exactly one matcher"
                    )
                matcher = rule[next(iter(selected))]
                if not isinstance(matcher, (dict, list)) or not matcher:
                    raise ValueError(
                        "pod_failure_policy rule matcher must be non-empty"
                    )
        return self


class DeploymentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    replicas: int = Field(default=1, ge=0)
    strategy: dict[str, Any] = Field(default_factory=dict)


class LeaderWorkerSetPolicy(BaseModel):
    """LWS controller policy with an optional same-shaped explicit leader.

    Manifesto applies the Workload's accelerator count, accelerator container,
    and cluster-owned claims to both templates. Controllers needing a CPU-only
    or differently sized leader must use separate workload intent rather than
    this same-shaped leader contract.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    replicas: int = Field(default=1, ge=0)
    size: int = Field(default=1, ge=1)
    startup_policy: Literal["LeaderCreated", "LeaderReady"] | None = None
    restart_policy: Literal["None", "RecreateGroupOnPodRestart"] | None = None
    network_config: dict[str, Any] = Field(default_factory=dict)
    leader_template: PodTemplate | None = None
    rollout_strategy: dict[str, Any] = Field(default_factory=dict)


class ServicePort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    port: int = Field(ge=1, le=65535)
    target_port: int | str
    protocol: Literal["TCP", "UDP", "SCTP"] = "TCP"


class WorkloadService(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ports: list[ServicePort] = Field(min_length=1)
    selector: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    headless: bool = False
    publish_not_ready_addresses: bool = False


class Workload(BaseModel):
    """Controller-neutral workload intent lowered to raw Kubernetes objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    backend: WorkloadBackend
    metadata: WorkloadMetadata = Field(default_factory=WorkloadMetadata)
    pod_template: PodTemplate
    selector: dict[str, str] = Field(default_factory=dict)
    accelerator_count: int = Field(default=0, ge=0)
    accelerator_container: str | None = None
    queue_name: str | None = None
    workload_priority_class: str | None = None
    job: JobPolicy | None = None
    deployment: DeploymentPolicy | None = None
    leader_worker_set: LeaderWorkerSetPolicy | None = None
    service: WorkloadService | None = None

    @model_validator(mode="after")
    def require_backend_policy(self) -> Workload:
        policies = {
            WorkloadBackend.JOB: self.job,
            WorkloadBackend.DEPLOYMENT: self.deployment,
            WorkloadBackend.LEADER_WORKER_SET: self.leader_worker_set,
        }
        if self.backend in policies and policies[self.backend] is None:
            raise ValueError(f"{self.backend.value} backend requires its policy")
        if (
            self.backend == WorkloadBackend.JOB
            and self.job is not None
            and self.job.pod_failure_policy
            and self.pod_template.spec.get("restartPolicy") != "Never"
        ):
            raise ValueError(
                "Jobs with pod_failure_policy require restartPolicy: Never"
            )
        return self


def load_workload(path: str | Path) -> Workload:
    """Load controller-neutral workload intent from YAML."""

    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"workload must be a YAML mapping: {path}")
    return Workload.model_validate(data)


class WorkloadAccelerator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allocation: AcceleratorAllocationConfig
    node_selector: dict[str, str] = Field(default_factory=dict)

    @property
    def resource_name(self) -> str | None:
        backend = self.allocation.extended_resource
        return backend.resource_name if backend is not None else None

    @property
    def device_class_name(self) -> str | None:
        backend = self.allocation.dra
        return backend.device_class_name if backend is not None else None


class WorkloadPodDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annotations: dict[str, str] = Field(default_factory=dict)
    affinity: dict[str, Any] = Field(default_factory=dict)
    tolerations: list[dict[str, Any]] = Field(default_factory=list)
    dns_policy: (
        Literal["ClusterFirst", "Default", "ClusterFirstWithHostNet", "None"] | None
    ) = None
    dns_config: dict[str, Any] = Field(default_factory=dict)
    image_pull_secrets: list[str] = Field(default_factory=list)
    image_pull_policy: Literal["Always", "IfNotPresent", "Never"] | None = None


class WorkloadResourceClaim(BaseModel):
    """Cluster-owned ResourceClaimTemplate attached to accelerator containers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    template_name: str

    @field_validator("name")
    @classmethod
    def require_claim_name(cls, value: str) -> str:
        if len(value) > 63 or _DNS_LABEL.fullmatch(value) is None:
            raise ValueError("resource claim name must be a DNS-1123 label")
        return value

    @field_validator("template_name")
    @classmethod
    def require_template_name(cls, value: str) -> str:
        labels = value.split(".")
        if (
            len(value) > 253
            or not labels
            or any(_DNS_LABEL.fullmatch(label) is None for label in labels)
        ):
            raise ValueError(
                "resource claim template name must be a DNS-1123 subdomain"
            )
        return value


class WorkloadSettings(BaseModel):
    """Cluster-owned settings that are safe to reuse for arbitrary GPU Jobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_accelerator: str
    accelerators: dict[str, WorkloadAccelerator]
    local_queue: str | None = None
    pod: WorkloadPodDefaults = Field(default_factory=WorkloadPodDefaults)
    accelerator_resource_claims: tuple[WorkloadResourceClaim, ...] = ()

    @model_validator(mode="after")
    def require_unique_resource_claim_names(self) -> WorkloadSettings:
        names = [claim.name for claim in self.accelerator_resource_claims]
        if len(names) != len(set(names)):
            raise ValueError("accelerator resource claim names must be unique")
        return self

    def accelerator(self, name: str | None = None) -> WorkloadAccelerator:
        selected = name if name and name != "any" else self.default_accelerator
        try:
            return self.accelerators[selected]
        except KeyError as exc:
            choices = ", ".join(sorted(self.accelerators))
            raise ValueError(
                f"unknown accelerator {selected!r} for this cluster; choose one of: {choices}"
            ) from exc


def workload_settings(cluster: Cluster) -> WorkloadSettings:
    """Project a full cluster profile onto the reusable workload contract."""

    pod = cluster.pod_defaults
    return WorkloadSettings(
        default_accelerator=cluster.accelerators.default,
        accelerators={
            name: WorkloadAccelerator(
                allocation=accelerator.allocation,
                node_selector=copy.deepcopy(accelerator.node_selector),
            )
            for name, accelerator in cluster.accelerators.profiles.items()
        },
        local_queue=cluster.kueue.local_queue,
        pod=WorkloadPodDefaults(
            annotations=copy.deepcopy(pod.annotations),
            affinity=copy.deepcopy(pod.affinity),
            tolerations=copy.deepcopy(pod.tolerations),
            dns_policy=pod.dns_policy,
            dns_config=copy.deepcopy(pod.dns_config),
            image_pull_secrets=list(pod.image_pull_secrets),
            image_pull_policy=pod.image_pull_policy,
        ),
        accelerator_resource_claims=(
            (
                WorkloadResourceClaim(
                    name="compute-domain-channel",
                    template_name=cluster.fabric.imex_resource_claim_template,
                ),
            )
            if cluster.fabric.imex_resource_claim_template
            else ()
        ),
    )


def render_workload(
    workload: Workload,
    *,
    settings: WorkloadSettings | None = None,
    accelerator: str | None = None,
) -> list[dict[str, Any]]:
    """Lower workload intent to Kubernetes objects without contacting a cluster."""

    pod_template = workload.pod_template.model_dump(mode="python")
    accelerator_claim_template = None
    leader_template = (
        workload.leader_worker_set.leader_template.model_dump(mode="python")
        if workload.leader_worker_set
        and workload.leader_worker_set.leader_template is not None
        else None
    )
    if workload.backend == WorkloadBackend.LEADER_WORKER_SET:
        for label in (KUEUE_QUEUE_LABEL, KUEUE_PRIORITY_LABEL):
            pod_template["metadata"]["labels"].pop(label, None)
        if leader_template is not None:
            for label in (KUEUE_QUEUE_LABEL, KUEUE_PRIORITY_LABEL):
                leader_template["metadata"]["labels"].pop(label, None)
    if settings is not None:
        accelerator_claim_template = _apply_settings(
            pod_template, settings, accelerator, workload
        )
        if leader_template is not None:
            leader_claim_template = _apply_settings(
                leader_template, settings, accelerator, workload
            )
            if leader_claim_template != accelerator_claim_template:
                raise ValueError(
                    "LeaderWorkerSet leader and worker templates resolved to "
                    "different accelerator claims"
                )
    if not pod_template["metadata"].get("annotations"):
        pod_template["metadata"].pop("annotations", None)

    workload_labels = dict(workload.metadata.labels)
    pod_labels = pod_template["metadata"]["labels"]
    resolved_selector = workload.selector or dict(pod_labels)
    queue_name = workload.queue_name or (settings.local_queue if settings else None)
    if queue_name:
        workload_labels[KUEUE_QUEUE_LABEL] = queue_name
        if workload.backend in {WorkloadBackend.JOB, WorkloadBackend.DEPLOYMENT}:
            pod_labels[KUEUE_QUEUE_LABEL] = queue_name
    if workload.workload_priority_class:
        workload_labels[KUEUE_PRIORITY_LABEL] = workload.workload_priority_class
        if workload.backend in {WorkloadBackend.JOB, WorkloadBackend.DEPLOYMENT}:
            pod_labels[KUEUE_PRIORITY_LABEL] = workload.workload_priority_class

    objects: list[dict[str, Any]] = []
    if workload.service is not None:
        objects.append(
            render_service(
                workload.service,
                default_selector=resolved_selector,
            )
        )
    if accelerator_claim_template is not None:
        objects.append(accelerator_claim_template)

    if workload.backend == WorkloadBackend.JOB:
        objects.append(_render_job(workload, workload_labels, pod_template))
    elif workload.backend == WorkloadBackend.DEPLOYMENT:
        objects.append(
            _render_deployment(
                workload,
                workload_labels,
                pod_template,
                resolved_selector,
            )
        )
    elif workload.backend == WorkloadBackend.LEADER_WORKER_SET:
        objects.append(
            _render_leader_worker_set(
                workload,
                workload_labels,
                pod_template,
                leader_template,
            )
        )
    else:
        raise NotImplementedError(
            "Grove is reserved as a workload backend but does not yet have a "
            "stable Manifesto renderer"
        )
    return objects


def render_service(
    service: WorkloadService, *, default_selector: dict[str, str]
) -> dict[str, Any]:
    """Render an optional stable or headless Service for a workload."""

    spec: dict[str, Any] = {
        "selector": dict(service.selector or default_selector),
        "ports": [
            {
                "name": port.name,
                "port": port.port,
                "targetPort": port.target_port,
                "protocol": port.protocol,
            }
            for port in service.ports
        ],
    }
    if service.headless:
        spec["clusterIP"] = "None"
    if service.publish_not_ready_addresses:
        spec["publishNotReadyAddresses"] = True
    metadata: dict[str, Any] = {"name": service.name}
    if service.labels:
        metadata["labels"] = dict(service.labels)
    if service.annotations:
        metadata["annotations"] = dict(service.annotations)
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": metadata,
        "spec": spec,
    }


def _apply_settings(
    pod_template: dict[str, Any],
    settings: WorkloadSettings,
    accelerator: str | None,
    workload: Workload,
) -> dict[str, Any] | None:
    pod_metadata = pod_template["metadata"]
    annotations = dict(settings.pod.annotations) | pod_metadata["annotations"]
    if annotations:
        pod_metadata["annotations"] = annotations
    else:
        pod_metadata.pop("annotations", None)
    pod_spec = pod_template["spec"]
    selected = settings.accelerator(accelerator)
    node_selector = dict(selected.node_selector) | pod_spec.get("nodeSelector", {})
    if node_selector:
        pod_spec["nodeSelector"] = node_selector
    if settings.pod.affinity:
        pod_spec.setdefault("affinity", copy.deepcopy(settings.pod.affinity))
    if settings.pod.tolerations:
        pod_spec.setdefault("tolerations", copy.deepcopy(settings.pod.tolerations))
    if settings.pod.dns_policy:
        pod_spec.setdefault("dnsPolicy", settings.pod.dns_policy)
    if settings.pod.dns_config:
        pod_spec.setdefault("dnsConfig", copy.deepcopy(settings.pod.dns_config))
    if settings.pod.image_pull_secrets:
        configured = [{"name": name} for name in settings.pod.image_pull_secrets]
        existing = pod_spec.get("imagePullSecrets", [])
        seen = {item.get("name") for item in configured}
        pod_spec["imagePullSecrets"] = [
            *configured,
            *(item for item in existing if item.get("name") not in seen),
        ]
    if settings.pod.image_pull_policy:
        for field_name in ("initContainers", "containers"):
            for container in pod_spec.get(field_name, []):
                container.setdefault("imagePullPolicy", settings.pod.image_pull_policy)
    if workload.accelerator_count:
        containers = pod_spec.get("containers", [])
        if workload.accelerator_container:
            container = next(
                (
                    item
                    for item in containers
                    if item.get("name") == workload.accelerator_container
                ),
                None,
            )
            if container is None:
                raise ValueError(
                    "accelerator_container does not name a pod container: "
                    f"{workload.accelerator_container!r}"
                )
        elif len(containers) == 1:
            container = containers[0]
        else:
            raise ValueError(
                "accelerator_container is required when a GPU workload has "
                "multiple containers"
            )
        for claim in settings.accelerator_resource_claims:
            attach_resource_claim(
                pod_spec,
                container,
                name=claim.name,
                template_name=claim.template_name,
            )
        if selected.resource_name is not None:
            resources = container.setdefault("resources", {})
            value = str(workload.accelerator_count)
            resources.setdefault("requests", {})[selected.resource_name] = value
            resources.setdefault("limits", {})[selected.resource_name] = value
        else:
            assert selected.device_class_name is not None
            template_name = accelerator_claim_template_name(
                workload.name,
                labels=workload.metadata.labels,
                device_class_name=selected.device_class_name,
                count=workload.accelerator_count,
            )
            attach_accelerator_claim(
                pod_spec,
                container,
                template_name=template_name,
            )
            return render_accelerator_claim_template(
                name=template_name,
                labels=workload.metadata.labels,
                device_class_name=selected.device_class_name,
                count=workload.accelerator_count,
            )
    return None


def _metadata(workload: Workload, labels: dict[str, str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": workload.name, "labels": labels}
    if workload.metadata.annotations:
        metadata["annotations"] = dict(workload.metadata.annotations)
    return metadata


def _render_job(
    workload: Workload,
    labels: dict[str, str],
    pod_template: dict[str, Any],
) -> dict[str, Any]:
    policy = workload.job
    assert policy is not None
    spec: dict[str, Any] = {
        "suspend": policy.suspend,
        "backoffLimit": policy.backoff_limit,
        "template": pod_template,
    }
    if policy.active_deadline_seconds is not None:
        spec["activeDeadlineSeconds"] = policy.active_deadline_seconds
    if policy.ttl_seconds_after_finished is not None:
        spec["ttlSecondsAfterFinished"] = policy.ttl_seconds_after_finished
    if policy.completion_mode is not None:
        spec["completionMode"] = policy.completion_mode
    if policy.completions is not None:
        spec["completions"] = policy.completions
    if policy.parallelism is not None:
        spec["parallelism"] = policy.parallelism
    if policy.pod_failure_policy:
        spec["podFailurePolicy"] = copy.deepcopy(policy.pod_failure_policy)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": _metadata(workload, labels),
        "spec": spec,
    }


def _render_deployment(
    workload: Workload,
    labels: dict[str, str],
    pod_template: dict[str, Any],
    selector: dict[str, str],
) -> dict[str, Any]:
    policy = workload.deployment
    assert policy is not None
    spec: dict[str, Any] = {
        "replicas": policy.replicas,
        "selector": {"matchLabels": dict(selector)},
        "template": pod_template,
    }
    if policy.strategy:
        spec["strategy"] = copy.deepcopy(policy.strategy)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": _metadata(workload, labels),
        "spec": spec,
    }


def _render_leader_worker_set(
    workload: Workload,
    labels: dict[str, str],
    pod_template: dict[str, Any],
    leader_template: dict[str, Any] | None,
) -> dict[str, Any]:
    policy = workload.leader_worker_set
    assert policy is not None
    spec: dict[str, Any] = {
        "replicas": policy.replicas,
        "leaderWorkerTemplate": {
            "size": policy.size,
            "workerTemplate": pod_template,
        },
    }
    if leader_template is not None:
        spec["leaderWorkerTemplate"]["leaderTemplate"] = leader_template
    if policy.restart_policy is not None:
        spec["leaderWorkerTemplate"]["restartPolicy"] = policy.restart_policy
    if policy.startup_policy is not None:
        spec["startupPolicy"] = policy.startup_policy
    if policy.network_config:
        spec["networkConfig"] = copy.deepcopy(policy.network_config)
    if policy.rollout_strategy:
        spec["rolloutStrategy"] = copy.deepcopy(policy.rollout_strategy)
    return {
        "apiVersion": "leaderworkerset.x-k8s.io/v1",
        "kind": "LeaderWorkerSet",
        "metadata": _metadata(workload, labels),
        "spec": spec,
    }
