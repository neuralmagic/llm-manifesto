"""Base Kubernetes objects shared by full deployment renders."""

from __future__ import annotations

from ..dp_ports import RolePorts
from ..instance import Instance


def render_service_account(instance: Instance) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": instance.name("model-server"), "labels": instance.labels("model-server")},
    }


def render_openshift_scc_binding(instance: Instance, *, namespace: str, scc: str) -> dict:
    service_account_name = instance.name("model-server")
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {
            "name": instance.name("model-server-scc"),
            "namespace": namespace,
            "labels": instance.labels("model-server"),
        },
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "ClusterRole",
            "name": f"system:openshift:scc:{scc}",
        },
        "subjects": [
            {
                "kind": "ServiceAccount",
                "name": service_account_name,
                "namespace": namespace,
            }
        ],
    }


def render_model_server_service(
    instance: Instance,
    role_name: str,
    ports: RolePorts,
    *,
    leader_only: bool = False,
) -> dict:
    selector = instance.pod_selector(role_name)
    if leader_only:
        selector["leaderworkerset.sigs.k8s.io/worker-index"] = "0"
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": instance.name(f"{role_name}-svc"),
            "labels": instance.labels("model-server", role_name),
        },
        "spec": {
            "selector": selector,
            "ports": [
                {"name": f"vllm-{idx}", "port": port, "targetPort": port}
                for idx, port in enumerate(ports.public)
            ],
        },
    }


def render_dcgm_metrics_configmap(instance: Instance) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": instance.name("dcgm-metrics"), "labels": instance.labels("monitoring")},
        "data": {
            "custom-counters.csv": "\n".join(
                [
                    "DCGM_FI_PROF_NVLINK_TX_BYTES, gauge, NVLink transmit bytes per second",
                    "DCGM_FI_PROF_NVLINK_RX_BYTES, gauge, NVLink receive bytes per second",
                    "DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization",
                    "DCGM_FI_DEV_FB_USED, gauge, Framebuffer memory used",
                    "DCGM_FI_DEV_FB_FREE, gauge, Framebuffer memory free",
                ]
            )
        },
    }
