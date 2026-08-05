"""Instance-wide idle detector that scales Manifesto workloads to zero."""

from __future__ import annotations

import hashlib
import json

from ..cluster import Cluster
from ..images import DEFAULT_IMAGES
from ..instance import Instance
from ..parallelism import parallel_layout
from ..resolve import resolve_role
from ..spec import DeploymentSpec, RoutingKind
from .lws import uses_leader_worker_set


IDLE_SHUTDOWN_SCRIPT = r'''import json
import os
import ssl
import time
import urllib.parse
import urllib.request

API = "https://kubernetes.default.svc"
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
NAMESPACE = os.environ["NAMESPACE"]
POD_SELECTOR = os.environ["POD_SELECTOR"]
TARGETS = json.loads(os.environ["TARGETS"])
EXPECTED_TARGETS = int(os.environ["EXPECTED_TARGETS"])
TIMEOUT_SECONDS = int(os.environ["TIMEOUT_SECONDS"])
POLL_SECONDS = 60
WORKLOADS = json.loads(os.environ["WORKLOADS"])

CONTEXT = ssl.create_default_context(cafile=CA_PATH)


def api_request(path, *, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    with open(TOKEN_PATH, encoding="utf-8") as token_file:
        token = token_file.read().strip()
    request = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/merge-patch+json",
        },
    )
    with urllib.request.urlopen(request, context=CONTEXT, timeout=15) as response:
        return json.load(response)


def ready_vllm_targets():
    query = urllib.parse.urlencode({"labelSelector": POD_SELECTOR})
    pods = api_request(f"/api/v1/namespaces/{NAMESPACE}/pods?{query}")
    targets = []
    for pod in pods.get("items", []):
        conditions = pod.get("status", {}).get("conditions", [])
        if not any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions):
            continue
        pod_ip = pod.get("status", {}).get("podIP")
        labels = pod.get("metadata", {}).get("labels", {})
        target = TARGETS.get(labels.get("llm-d.ai/role"))
        if not target:
            continue
        worker_indices = target["worker_indices"]
        if worker_indices is not None:
            worker_index = labels.get("leaderworkerset.sigs.k8s.io/worker-index")
            if worker_index not in worker_indices:
                continue
        for port in target["ports"]:
            targets.append((pod_ip, port))
    return targets


def metric_value(line):
    try:
        return float(line.rsplit(None, 1)[1])
    except (IndexError, ValueError):
        return 0.0


def scrape_activity(targets):
    completed = 0.0
    running = 0.0
    recognized = False
    for host, port in targets:
        with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=15) as response:
            metrics = response.read().decode("utf-8", errors="replace")
        for line in metrics.splitlines():
            if (
                line.startswith("vllm:request_success_total{")
                or line.startswith("vllm:request_success_total ")
                or line.startswith("vllm:request_failure_total{")
                or line.startswith("vllm:request_failure_total ")
            ):
                completed += metric_value(line)
                recognized = True
            elif (
                line.startswith("vllm:num_requests_running{")
                or line.startswith("vllm:num_requests_running ")
                or line.startswith("vllm:num_requests_waiting{")
                or line.startswith("vllm:num_requests_waiting ")
            ):
                running += metric_value(line)
                recognized = True
    return completed, running, recognized


def scale_to_zero():
    scaled = []
    try:
        for workload in WORKLOADS:
            api_request(workload["path"], method="PATCH", body={"spec": {"replicas": 0}})
            scaled.append(workload)
            print(f"scaled {workload['name']} to zero", flush=True)
    except Exception:
        for workload in reversed(scaled):
            try:
                api_request(
                    workload["path"],
                    method="PATCH",
                    body={"spec": {"replicas": workload["replicas"]}},
                )
                print(
                    f"restored {workload['name']} to {workload['replicas']} replicas",
                    flush=True,
                )
            except Exception as rollback_error:
                print(
                    f"failed to restore {workload['name']}: {rollback_error}",
                    flush=True,
                )
        raise


last_activity = time.monotonic()
previous_completed = None
while True:
    time.sleep(POLL_SECONDS)
    try:
        targets = ready_vllm_targets()
        if len(targets) != EXPECTED_TARGETS:
            last_activity = time.monotonic()
            previous_completed = None
            continue
        completed, running, recognized = scrape_activity(targets)
        if not recognized:
            raise RuntimeError("vLLM activity metrics were not found")
        if previous_completed is None or completed != previous_completed or running > 0:
            last_activity = time.monotonic()
        previous_completed = completed
        if time.monotonic() - last_activity >= TIMEOUT_SECONDS:
            scale_to_zero()
            break
    except Exception as error:
        # Fail open: API, network, and metric-format failures must never tear down
        # a deployment whose idleness cannot be established.
        print(f"idle check failed: {error}", flush=True)
        last_activity = time.monotonic()
        previous_completed = None
'''


def render_idle_shutdown(
    spec: DeploymentSpec,
    instance: Instance,
    cluster: Cluster,
) -> list[dict]:
    """Render a small controller that watches vLLM request metrics instance-wide."""
    if not spec.runtime.idle_shutdown.enabled or not spec.roles:
        return []

    name = instance.name("idle-shutdown")
    role_name = instance.name("idle-shutdown-rbac")
    labels = instance.labels("idle-shutdown")
    pod_annotations = {
        "manifesto.llm-d.ai/idle-shutdown-script-sha256": hashlib.sha256(
            IDLE_SHUTDOWN_SCRIPT.encode()
        ).hexdigest()
    }
    rules: list[dict] = []
    model_workloads: list[dict[str, str | int]] = []
    deployment_names: list[str] = []
    lws_names: list[str] = []
    targets: dict[str, dict] = {}
    expected_targets = 0
    for role in spec.roles:
        workload_name = (
            instance.user_scoped_name(role.workload_name)
            if role.workload_name
            else instance.name(role.name)
        )
        resolved = resolve_role(spec, instance, cluster, role)
        layout = parallel_layout(role)
        targets[instance.labels(role=role.name)["llm-d.ai/role"]] = {
            "ports": list(resolved.ports.backend),
            "worker_indices": (
                [str(index) for index in layout.serving_worker_indices]
                if layout.cross_node_tp
                else None
            ),
        }
        serving_pods_per_replica = (
            len(layout.serving_worker_indices)
            if layout.cross_node_tp
            else role.lws.size
        )
        expected_targets += (
            role.lws.replicas
            * serving_pods_per_replica
            * len(resolved.ports.backend)
        )
        if not uses_leader_worker_set(
            cluster,
            role,
            resolved.features.workload_kind,
        ):
            deployment_names.append(workload_name)
            model_workloads.append(
                {
                    "name": workload_name,
                    "path": f"/apis/apps/v1/namespaces/{spec.namespace}/deployments/{workload_name}",
                    "replicas": role.lws.replicas,
                }
            )
        else:
            lws_names.append(workload_name)
            model_workloads.append(
                {
                    "name": workload_name,
                    "path": f"/apis/leaderworkerset.x-k8s.io/v1/namespaces/{spec.namespace}/leaderworkersets/{workload_name}",
                    "replicas": role.lws.replicas,
                }
            )
    workloads = model_workloads
    if spec.routing.kind != RoutingKind.DISABLED:
        epp_name = instance.name("infpool-epp")
        deployment_names.append(epp_name)
        workloads.append(
            {
                "name": epp_name,
                "path": f"/apis/apps/v1/namespaces/{spec.namespace}/deployments/{epp_name}",
                "replicas": (
                    spec.routing.epp.replicas
                    if spec.routing.epp is not None
                    else spec.routing.replicas
                ),
            }
        )

    controller_path = f"/apis/apps/v1/namespaces/{spec.namespace}/deployments/{name}"
    # Stop the detector last. Re-applying the manifest restores every desired replica count.
    workloads.append({"name": name, "path": controller_path, "replicas": 1})

    rules.append(
        {
            "apiGroups": [""],
            "resources": ["pods"],
            "verbs": ["get", "list"],
        }
    )
    rules.append(
        {
            "apiGroups": ["apps"],
            "resources": ["deployments"],
            "resourceNames": [*deployment_names, name],
            "verbs": ["get", "patch"],
        }
    )
    if lws_names:
        rules.append(
            {
                "apiGroups": ["leaderworkerset.x-k8s.io"],
                "resources": ["leaderworkersets"],
                "resourceNames": lws_names,
                "verbs": ["get", "patch"],
            }
        )

    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": name, "labels": labels},
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {"name": role_name, "labels": labels},
            "rules": rules,
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {"name": role_name, "labels": labels},
            "subjects": [
                {"kind": "ServiceAccount", "name": name, "namespace": spec.namespace}
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": role_name,
            },
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": name, "labels": labels},
            "data": {"idle_shutdown.py": IDLE_SHUTDOWN_SCRIPT},
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": name, "labels": labels},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels, "annotations": pod_annotations},
                    "spec": {
                        "serviceAccountName": name,
                        "containers": [
                            {
                                "name": "idle-shutdown",
                                "image": DEFAULT_IMAGES.get("sidecars.idle_shutdown"),
                                "imagePullPolicy": "IfNotPresent",
                                "command": ["python", "/opt/manifesto/idle_shutdown.py"],
                                "env": [
                                    {
                                        "name": "NAMESPACE",
                                        "valueFrom": {
                                            "fieldRef": {"fieldPath": "metadata.namespace"}
                                        },
                                    },
                                    {
                                        "name": "POD_SELECTOR",
                                        "value": ",".join(
                                            f"{key}={value}"
                                            for key, value in (
                                                instance.pod_selector()
                                                | {"app.kubernetes.io/component": "model-server"}
                                            ).items()
                                        ),
                                    },
                                    {
                                        "name": "TIMEOUT_SECONDS",
                                        "value": str(spec.runtime.idle_shutdown.timeout_minutes * 60),
                                    },
                                    {
                                        "name": "EXPECTED_TARGETS",
                                        "value": str(expected_targets),
                                    },
                                    {
                                        "name": "TARGETS",
                                        "value": json.dumps(targets, separators=(",", ":")),
                                    },
                                    {
                                        "name": "WORKLOADS",
                                        "value": json.dumps(workloads, separators=(",", ":")),
                                    },
                                ],
                                "resources": {
                                    "requests": {"cpu": "10m", "memory": "32Mi"},
                                    "limits": {"cpu": "100m", "memory": "64Mi"},
                                },
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": True,
                                    "runAsNonRoot": True,
                                    **(
                                        {}
                                        if cluster.platform == "openshift"
                                        else {"runAsUser": 65532}
                                    ),
                                },
                                "volumeMounts": [
                                    {
                                        "name": "script",
                                        "mountPath": "/opt/manifesto/idle_shutdown.py",
                                        "subPath": "idle_shutdown.py",
                                        "readOnly": True,
                                    }
                                ],
                            }
                        ],
                        "volumes": [{"name": "script", "configMap": {"name": name}}],
                    },
                },
            },
        },
    ]
