"""Workload renderer for vLLM roles, including sidecars and fabric mounts."""

from __future__ import annotations

import copy

from .common import env_list, field_ref_env, secret_env
from .sidecars import sidecars
from ..cluster import Cluster
from ..features import Feature, WorkloadKind
from ..instance import Instance
from ..launch import build_launch_script
from ..parallelism import parallel_layout
from ..resolve import resolve_role
from ..spec import DeploymentSpec, RoleSpec
from ..workload import (
    KUEUE_QUEUE_LABEL as KUEUE_QUEUE_LABEL,
    DeploymentPolicy,
    LeaderWorkerSetPolicy,
    PodTemplate,
    Workload,
    WorkloadBackend,
    WorkloadMetadata,
    render_workload as render_controller_workload,
)

# Endpoint-picker hint listing which InferencePool target ports a pod really serves.
ACTIVE_PORTS_ANNOTATION = "inference.networking.k8s.io/active-ports"

# LWS stamps this on the leader and every worker of a group; the value is a hash
# of the leader's namespaced name, so it is unique per replica cluster-wide.
LWS_GROUP_KEY_LABEL = "leaderworkerset.sigs.k8s.io/group-key"


def render_workload(spec: DeploymentSpec, instance: Instance, cluster: Cluster, role: RoleSpec) -> dict:
    resolved = resolve_role(spec, instance, cluster, role)
    accelerator = spec.accelerator_config(cluster)
    external_dp = resolved.features.external_dp
    multi_port_external_dp = external_dp and resolved.ports.rank_count > 1
    layout = parallel_layout(role)
    cross_node_tp = layout.cross_node_tp
    distributed_dp = layout.distributed_dp
    workload_name = instance.user_scoped_name(role.workload_name) if role.workload_name else instance.name(role.name)

    containers, extra_volumes = sidecars(
        spec.runtime.sidecars,
        dcgm_config_name=instance.name("dcgm-metrics"),
    )
    volumes = cluster.base_volumes()
    if role.shm_size:
        volumes[0]["emptyDir"]["sizeLimit"] = role.shm_size
    volumes.extend(extra_volumes)

    container_ports = [
        {"containerPort": port, "name": f"vllm-{idx}"}
        for idx, port in enumerate(resolved.ports.backend)
    ]
    if multi_port_external_dp:
        container_ports.insert(0, {"containerPort": 8100, "name": "dp-supervisor"})
    if distributed_dp:
        container_ports.append({"containerPort": 5555, "name": "dp-rpc"})
    readiness_ports = resolved.ports.public if resolved.features.routing_proxy else resolved.ports.backend

    init_containers = []
    if resolved.features.routing_proxy:
        init_containers.append(
            {
                "name": "routing-proxy",
                "image": cluster.llm_d.routing_sidecar,
                "imagePullPolicy": "Always",
                "args": [
                    f"--port={resolved.ports.public[0]}",
                    f"--vllm-port={resolved.ports.backend[0]}",
                    f"--data-parallel-size={resolved.ports.rank_count}",
                    "--secure-proxy=false",
                    "--connector=nixlv2",
                ],
                "ports": [
                    {"containerPort": port, "name": f"rank{idx}", "protocol": "TCP"}
                    for idx, port in enumerate(resolved.ports.public)
                ],
                "restartPolicy": "Always",
                "resources": {
                    "requests": {"cpu": 8, "memory": "16Gi"},
                    "limits": {"cpu": 8, "memory": "16Gi"},
                },
                "securityContext": {"allowPrivilegeEscalation": False},
            }
        )

    security_context = cluster.pod_defaults.container_security_context
    container_env = [
        secret_env("HF_TOKEN", "hf-secret", "HF_TOKEN"),
        *env_list(resolved.env),
        *(
            field_ref_env(contribution.name, contribution.field_path)
            for contribution in resolved.features.field_ref_env
        ),
    ]
    if resolved.persistent_cache and spec.cache.cleanup_on_crash:
        container_env.append(field_ref_env("MANIFESTO_POD_UID", "metadata.uid"))

    vllm_requests = {
        "cpu": role.resources.cpu,
        "memory": role.resources.memory,
    }
    vllm_limits = {"memory": role.resources.memory}
    if role.resources.gpus > 0:
        vllm_requests[accelerator.resource_name] = str(role.resources.gpus)
        vllm_limits[accelerator.resource_name] = str(role.resources.gpus)

    vllm_container = {
        "name": "vllm",
        "image": spec.model.image,
        "command": ["/bin/bash", "-c"],
        "args": [
            build_launch_script(
                spec,
                role,
                resolved.ports,
                log_dir=resolved.log_dir,
                trace_dir=resolved.trace_dir,
                vllm_env=resolved.vllm_env,
                persistent_cache=resolved.persistent_cache,
                vllm_args=resolved.vllm_args,
                external_dp=external_dp,
                multi_port_external_dp=multi_port_external_dp,
                distributed_dp=distributed_dp,
                vllm_raw_args=resolved.vllm_raw_args,
            )
        ],
        "env": container_env,
        "ports": container_ports,
        "resources": {
            "requests": vllm_requests,
            "limits": vllm_limits,
        },
        "volumeMounts": cluster.volume_mounts(),
    }
    if cluster.pod_defaults.image_pull_policy:
        vllm_container["imagePullPolicy"] = cluster.pod_defaults.image_pull_policy
    if security_context is not None:
        vllm_container["securityContext"] = security_context
    if cluster.pod_defaults.working_dir:
        vllm_container["workingDir"] = cluster.pod_defaults.working_dir
    if role.resources.ephemeral_storage:
        for resource_kind in ("requests", "limits"):
            vllm_container["resources"][resource_kind]["ephemeral-storage"] = (
                role.resources.ephemeral_storage
            )
    if cross_node_tp:
        leader_readiness = " && ".join(
            f"curl -sf http://localhost:{port}/v1/models | grep -q '\"id\"'"
            for port in readiness_ports
        )
        if distributed_dp and external_dp:
            readiness_guard = (
                f"if (( ${{LWS_WORKER_INDEX:-0}} % {layout.tp_node_count} != 0 )); "
                "then exit 0; fi"
            )
        else:
            readiness_guard = (
                'if [ "${LWS_WORKER_INDEX:-0}" -gt 0 ]; then exit 0; fi'
            )
        vllm_container["readinessProbe"] = {
            "exec": {
                "command": [
                    "/bin/bash",
                    "-c",
                    f"{readiness_guard}; {leader_readiness}",
                ]
            },
            "periodSeconds": 5,
            "failureThreshold": 120,
        }
    elif len(readiness_ports) == 1:
        vllm_container["readinessProbe"] = {
            "httpGet": {"path": "/v1/models", "port": readiness_ports[0]},
            "periodSeconds": 5,
            "failureThreshold": 120,
        }
    else:
        vllm_container["readinessProbe"] = {
            "exec": {
                "command": [
                    "/bin/bash",
                    "-c",
                    " && ".join(
                        f"curl -sf http://localhost:{port}/v1/models | grep -q '\"id\"'"
                        for port in readiness_ports
                    ),
                ]
            },
            "periodSeconds": 5,
            "failureThreshold": 120,
        }
    if multi_port_external_dp:
        vllm_container["startupProbe"] = {
            "httpGet": {"path": "/health", "port": "dp-supervisor"},
            "periodSeconds": 1,
            "timeoutSeconds": 5,
            "failureThreshold": 1800,
        }
    if cluster.rdma.resource_name:
        for resources in ("requests", "limits"):
            vllm_container["resources"][resources][cluster.rdma.resource_name] = cluster.rdma.value
    if resolved.resource_claims:
        vllm_container["resources"]["claims"] = [{"name": claim["name"]} for claim in resolved.resource_claims]

    pod_labels = instance.labels("model-server", role.name) | {
        "llm-d.ai/inferenceServing": "true",
        "llm-d.ai/model": spec.model.label_value,
        "llm-d.ai/deployment": spec.topology.value,
    }
    pod_metadata = {"labels": pod_labels}
    annotations = dict(cluster.pod_defaults.annotations)
    if Feature.LLM_D in resolved.features.enabled:
        # A shared PD InferencePool advertises the union of both roles' ports.
        # Without this, the endpoint picker assumes every target port is live on
        # every pod and routes to ports a role with fewer ranks never opens.
        annotations[ACTIVE_PORTS_ANNOTATION] = ",".join(
            str(port) for port in resolved.ports.public
        )
    if annotations:
        pod_metadata["annotations"] = annotations

    pod_spec = {"volumes": volumes, "containers": [vllm_container, *containers]}
    if accelerator.node_selector:
        pod_spec["nodeSelector"] = dict(accelerator.node_selector)
    if cluster.openshift.scc:
        pod_spec["serviceAccountName"] = instance.name("model-server")
    if cluster.pod_defaults.termination_grace_period_seconds is not None:
        pod_spec["terminationGracePeriodSeconds"] = (
            cluster.pod_defaults.termination_grace_period_seconds
        )
    affinity = copy.deepcopy(cluster.pod_defaults.affinity)
    if role.lws.same_topology_key:
        required = affinity.setdefault("podAffinity", {}).setdefault(
            "requiredDuringSchedulingIgnoredDuringExecution", []
        )
        # Scoped to the role, not the instance: a role's own collectives share
        # an NVLink domain, but PD roles only exchange KV over RDMA, so forcing
        # both into one domain would reject placements that work fine.
        #
        # matchLabelKeys narrows the term to the pod's own LWS group, which is
        # what makes this correct at replicas > 1. The scheduler reads each
        # listed key off the incoming pod and ANDs key=value into the selector
        # above, so a decode pod in group abc123 effectively requires
        # {instance, role=decode, group-key=abc123}. We cannot put group-key in
        # matchLabels ourselves: one pod template is shared by every replica and
        # LWS mints the value per group at admission, so there is nothing to
        # hardcode. Without it the selector matches *any* replica of the role,
        # which reads as "join a domain that already holds one of my role's
        # pods" -- replica 0 seeds a domain and every later replica is then
        # required to pile into that same one, going Pending once it fills
        # rather than starting a second domain. With it, each group's leader
        # finds no pod carrying its own group-key, hits the scheduler's
        # empty-selector case (a required term with zero matches cluster-wide is
        # satisfied), and is free to seed whichever domain fits; its workers
        # then must follow it. Net: each replica stays domain-whole and replicas
        # pack independently. group-key is globally unique, so the instance and
        # role labels are strictly redundant once this is set -- they are kept
        # because they cost nothing and state the intent.
        #
        # Requires the podAffinity matchLabelKeys beta (Kubernetes 1.31+,
        # OpenShift 4.18+). Older API servers prune the field silently and fall
        # back to the single-domain behavior described above -- no error, so
        # check the server version before relying on this on a new cluster.
        #
        # This is placement, not gang scheduling: a leader picks its domain by
        # where it alone fits, not where all lws.size pods fit. If domains start
        # running near-full, reach for Kueue topology-aware scheduling rather
        # than more affinity terms.
        required.append(
            {
                "labelSelector": {
                    "matchLabels": instance.pod_selector(role.name),
                },
                "matchLabelKeys": [LWS_GROUP_KEY_LABEL],
                "topologyKey": role.lws.same_topology_key,
            }
        )
    if affinity:
        pod_spec["affinity"] = affinity
    if cluster.pod_defaults.tolerations:
        pod_spec["tolerations"] = cluster.pod_defaults.tolerations
    if cluster.pod_defaults.dns_policy:
        pod_spec["dnsPolicy"] = cluster.pod_defaults.dns_policy
    if cluster.pod_defaults.dns_config:
        pod_spec["dnsConfig"] = cluster.pod_defaults.dns_config
    if init_containers:
        pod_spec["initContainers"] = init_containers
    if resolved.resource_claims:
        pod_spec["resourceClaims"] = resolved.resource_claims

    if resolved.features.workload_kind == WorkloadKind.DEPLOYMENT:
        selector = instance.pod_selector(role.name)
        workload = Workload(
            name=workload_name,
            backend=WorkloadBackend.DEPLOYMENT,
            metadata=WorkloadMetadata(
                labels=instance.labels("model-server", role.name)
            ),
            pod_template=PodTemplate(
                metadata=WorkloadMetadata(**copy.deepcopy(pod_metadata)),
                spec=pod_spec,
            ),
            selector=selector,
            queue_name=(
                cluster.kueue.local_queue if role.resources.gpus > 0 else None
            ),
            deployment=DeploymentPolicy(
                replicas=role.lws.replicas,
                strategy={
                    "type": "RollingUpdate",
                    "rollingUpdate": {"maxSurge": 0, "maxUnavailable": "100%"},
                },
            ),
        )
        return render_controller_workload(workload)[0]

    workload_labels = instance.labels("lws", role.name) | {
        "llm-d.ai/inferenceServing": "true",
        "llm-d.ai/model": spec.model.label_value,
        "llm-d.ai/deployment": spec.topology.value,
    }
    workload = Workload(
        name=workload_name,
        backend=WorkloadBackend.LEADER_WORKER_SET,
        metadata=WorkloadMetadata(labels=workload_labels),
        pod_template=PodTemplate(
            metadata=WorkloadMetadata(**pod_metadata),
            spec=pod_spec,
        ),
        queue_name=(
            cluster.kueue.local_queue if role.resources.gpus > 0 else None
        ),
        leader_worker_set=LeaderWorkerSetPolicy(
            replicas=role.lws.replicas,
            size=role.lws.size,
            rollout_strategy={
                "type": "RollingUpdate",
                "rollingUpdateConfiguration": {
                    "maxUnavailable": "100%"
                },
            },
        ),
    )
    return render_controller_workload(workload)[0]
