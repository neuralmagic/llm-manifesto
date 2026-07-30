"""Render the persistent per-user dev pod for building vLLM from source."""

from __future__ import annotations

from .common import env_list, secret_env
from ..cluster import AcceleratorConfig, Cluster
from ..instance import Instance


def render_dev_pod(
    cluster: Cluster,
    user: str,
    accelerator: AcceleratorConfig | None = None,
    *,
    image: str | None = None,
    cpu: str | None = None,
    memory: str | None = None,
    gpus: int | None = None,
    run_as_user: int | None = None,
) -> dict:
    accelerator = accelerator or cluster.accelerators.get()
    instance = Instance(user=user, release="dev")
    name = instance.user_scoped_name("vllm-dev")
    user_root = cluster.user_root(user=instance.user_slug, release="")
    cpu = cpu or cluster.dev.cpu
    memory = memory or cluster.dev.memory
    gpus = cluster.dev.gpus if gpus is None else gpus
    resources = {
        "requests": {"cpu": cpu, "memory": memory},
        "limits": {"cpu": cpu, "memory": memory},
    }
    if gpus:
        for resource_kind in ("requests", "limits"):
            resources[resource_kind][accelerator.resource_name] = str(gpus)

    volumes = [
        {
            "name": "dshm",
            "emptyDir": {"medium": "Memory", "sizeLimit": cluster.pod_defaults.shm_size},
        }
    ]
    mounts = [{"name": "dshm", "mountPath": "/dev/shm"}]
    if cluster.storage.shared_volume:
        volumes.append({"name": "shared-storage", **cluster.storage.shared_volume})
        mounts.append({"name": "shared-storage", "mountPath": cluster.storage.shared_mount_path})
        hf_home = f"{cluster.storage.shared_mount_path}/hf_cache"
    elif cluster.cache.hf_host_path and cluster.cache.jit_host_path:
        volumes.extend(
            [
                {"name": "hf-cache", "hostPath": {"path": cluster.cache.hf_host_path, "type": "DirectoryOrCreate"}},
                {"name": "jit-cache", "hostPath": {"path": cluster.cache.jit_host_path, "type": "DirectoryOrCreate"}},
            ]
        )
        mounts.extend(
            [
                {"name": "hf-cache", "mountPath": "/var/cache/huggingface"},
                {"name": "jit-cache", "mountPath": "/var/cache/vllm"},
            ]
        )
        hf_home = cluster.cache.hf_home
    else:
        raise ValueError("cluster profile has neither shared storage nor host caches for the dev pod")

    env = {
        "HF_HOME": hf_home,
        "VLLM_CACHE_ROOT": f"{user_root}/dev-caches/vllm",
        "FLASHINFER_CACHE_DIR": f"{user_root}/dev-caches/flashinfer",
        "TORCH_CUDA_ARCH_LIST": accelerator.torch_cuda_arch_list,
        "CCACHE_DIR": f"{user_root}/ccache",
        "UV_CACHE_DIR": f"{user_root}/dev-caches/uv",
        "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
        "CMAKE_C_COMPILER_LAUNCHER": "ccache",
    }

    affinity = cluster.pod_defaults.affinity or {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {"key": accelerator.presence_label, "operator": "Exists"}
                        ]
                    }
                ]
            }
        }
    }
    container_security_context = cluster.pod_defaults.container_security_context
    if cluster.platform == "openshift":
        container_security_context = {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        } | (container_security_context or {})
        if run_as_user is not None:
            container_security_context["runAsUser"] = run_as_user

    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "labels": {"app": name, **instance.labels("dev")},
        },
        "spec": {
            "restartPolicy": "Always",
            "affinity": affinity,
            "containers": [
                {
                    "name": "dev",
                    "image": image or cluster.dev.image,
                    "imagePullPolicy": "Always",
                    "command": ["sleep", "infinity"],
                    "resources": resources,
                    "env": [secret_env("HF_TOKEN", "hf-secret", "HF_TOKEN"), *env_list(env)],
                    "volumeMounts": mounts,
                    # Let dev_init create the configured source path as the
                    # container UID. A runtime-created workingDir on a mounted
                    # volume can otherwise be owned by root on OpenShift.
                    "workingDir": "/tmp",
                }
            ],
            "volumes": volumes,
        },
    }
    if cluster.pod_defaults.annotations:
        pod["metadata"]["annotations"] = cluster.pod_defaults.annotations
    if cluster.pod_defaults.tolerations:
        pod["spec"]["tolerations"] = cluster.pod_defaults.tolerations
    if container_security_context:
        pod["spec"]["containers"][0]["securityContext"] = container_security_context
    else:
        pod["spec"]["securityContext"] = {"runAsUser": 0, "runAsGroup": 0}
    if cluster.platform == "openshift" and "runAsUser" in container_security_context:
        fs_group = container_security_context["runAsUser"]
        pod["spec"]["securityContext"] = {
            "fsGroup": fs_group,
            "fsGroupChangePolicy": "OnRootMismatch",
        }
    dns_policy = cluster.pod_defaults.dns_policy or cluster.dev.dns_policy
    if dns_policy:
        pod["spec"]["dnsPolicy"] = dns_policy
    if cluster.pod_defaults.dns_config:
        pod["spec"]["dnsConfig"] = cluster.pod_defaults.dns_config
    return pod
