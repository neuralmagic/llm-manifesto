"""Generic cluster-profile customization rendering tests."""

from manifesto.cluster import Cluster
from manifesto.instance import Instance
from manifesto.render import render
from manifesto.resolve import resolve_role
from manifesto.spec import DeploymentSpec


def _custom_cluster(*, scc: str | None = None) -> Cluster:
    return Cluster.model_validate(
        {
            "name": "synthetic-cluster",
            "platform": "openshift",
            "gpus_per_node": 2,
            "accelerators": {
                "default": "test",
                "profiles": {
                    "test": {
                        "resource_name": "example.com/gpu",
                        "presence_label": "example.com/gpu.present",
                        "gpu_arch": "test",
                        "torch_cuda_arch_list": "12.3",
                    }
                },
            },
            "storage": {
                "shared_volume": {"emptyDir": {}},
                "shared_mount_path": "/mnt/shared",
                "local_nvme_path": None,
            },
            "rdma": {"resource_name": "example.com/rdma", "value": "2"},
            "pod_defaults": {
                "dns_policy": "None",
                "dns_config": {"nameservers": ["192.0.2.53"]},
                "annotations": {"example.com/network": "secondary-net"},
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "example.com/accelerator",
                                            "operator": "In",
                                            "values": ["accelerator-v1"],
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                },
                "tolerations": [
                    {"key": "example.com/accelerator", "operator": "Exists", "effect": "NoSchedule"}
                ],
                "extra_volumes": [
                    {
                        "name": "rdma-devices",
                        "hostPath": {"path": "/dev/rdma", "type": "Directory"},
                    }
                ],
                "extra_volume_mounts": [{"name": "rdma-devices", "mountPath": "/dev/rdma"}],
                "container_security_context": {
                    "capabilities": {"add": ["IPC_LOCK", "SYS_PTRACE"]},
                    "runAsGroup": 0,
                    "runAsUser": 0,
                },
            },
            "model_server_resources": {"cpu_per_gpu": "2", "memory_per_gpu": "8Gi"},
            "fabric": {
                "ucx_net_devices": "rdma0:1",
                "default_profile": "standard",
                "profiles": {"standard": {}},
            },
            "openshift": {"scc": scc},
        }
    )


def _spec(cluster: Cluster) -> DeploymentSpec:
    spec = DeploymentSpec.model_validate(
        {
            "release": "synthetic-release",
            "namespace": "synthetic-namespace",
            "topology": "aggregated",
            "model": {
                "id": "example/model",
                "label": "example-model",
                "image": "example.com/vllm:test",
            },
            "routing": {"kind": "disabled"},
            "runtime": {"sidecars": []},
            "roles": [
                {
                    "name": "decode",
                    "lws": {"size": 2, "replicas": 1},
                    "parallelism": {"tp": 1, "dp": 4, "ep": True, "gpus": 2},
                    "resources": {"cpu": "4", "memory": "16Gi", "gpus": 2},
                }
            ],
        }
    )
    spec.apply_cluster_defaults(cluster)
    return spec


def test_pod_defaults_render_metadata_scheduling_resources_and_security():
    cluster = _custom_cluster()
    spec = _spec(cluster)
    objects = render(spec, user="tester", cluster=cluster)
    lws = next(obj for obj in objects if obj["kind"] == "LeaderWorkerSet")
    template = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]
    pod_spec = template["spec"]
    container = pod_spec["containers"][0]

    assert template["metadata"]["annotations"] == {"example.com/network": "secondary-net"}
    assert pod_spec["affinity"] == cluster.pod_defaults.affinity
    assert pod_spec["dnsPolicy"] == "None"
    assert pod_spec["dnsConfig"] == {"nameservers": ["192.0.2.53"]}
    assert pod_spec["tolerations"] == cluster.pod_defaults.tolerations
    assert container["securityContext"] == cluster.pod_defaults.container_security_context

    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    mounts = {mount["name"]: mount for mount in container["volumeMounts"]}
    assert "local-nvme" not in volumes
    assert volumes["rdma-devices"]["hostPath"]["path"] == "/dev/rdma"
    assert mounts["rdma-devices"]["mountPath"] == "/dev/rdma"
    assert container["resources"]["requests"]["example.com/rdma"] == "2"
    assert container["resources"]["limits"]["example.com/rdma"] == "2"
    assert "ephemeral-storage" not in container["resources"]["requests"]
    assert "ephemeral-storage" not in container["resources"]["limits"]


def test_openshift_scc_binding_targets_release_service_account():
    cluster = _custom_cluster(scc="custom-driver")
    spec = _spec(cluster)
    objects = render(spec, user="tester", cluster=cluster)
    service_account = next(obj for obj in objects if obj["kind"] == "ServiceAccount")
    binding = next(obj for obj in objects if obj["kind"] == "RoleBinding")

    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "ClusterRole",
        "name": "system:openshift:scc:custom-driver",
    }
    assert binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": service_account["metadata"]["name"],
            "namespace": spec.namespace,
        }
    ]


def test_role_can_override_cluster_fabric_profile():
    cluster = _custom_cluster()
    cluster.fabric.profiles["custom_ep"] = cluster.fabric.profiles["standard"].model_copy(
        update={"env": {"CUSTOM_FABRIC_MODE": "enabled"}}
    )
    spec = _spec(cluster)
    role = spec.roles[0]
    role.fabric_profile = "custom_ep"

    resolved = resolve_role(spec, Instance(user="tester", release=spec.release), cluster, role)

    assert resolved.fabric_profile == "custom_ep"
    assert resolved.env["CUSTOM_FABRIC_MODE"] == "enabled"


def test_shared_storage_defaults_hf_home_without_local_nvme():
    cluster = _custom_cluster()
    spec = _spec(cluster)

    assert cluster.cache.hf_home == "/mnt/shared/hf_cache"
    assert spec.model.hf_home == "/mnt/shared/hf_cache"


def test_empty_container_security_context_is_preserved():
    cluster = _custom_cluster()
    cluster.pod_defaults.container_security_context = {}
    spec = _spec(cluster)
    objects = render(spec, user="tester", cluster=cluster)
    lws = next(obj for obj in objects if obj["kind"] == "LeaderWorkerSet")
    container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]

    assert container["securityContext"] == {}


def test_optional_pod_defaults_are_emitted_only_when_configured():
    cluster = _custom_cluster()
    cluster.pod_defaults.image_pull_policy = "IfNotPresent"
    cluster.pod_defaults.termination_grace_period_seconds = 30
    cluster.pod_defaults.working_dir = "/workspace"
    spec = _spec(cluster)

    objects = render(spec, user="tester", cluster=cluster)
    lws = next(obj for obj in objects if obj["kind"] == "LeaderWorkerSet")
    pod_spec = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    container = pod_spec["containers"][0]

    assert pod_spec["terminationGracePeriodSeconds"] == 30
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert container["workingDir"] == "/workspace"
