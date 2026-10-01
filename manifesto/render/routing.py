"""Standalone or Gateway API routing manifests for one Manifesto instance."""

from __future__ import annotations

import yaml

from ..cluster import Cluster
from ..instance import Instance
from ..resolve import ResolvedRole
from ..routing import (
    ENVOY_CONFIG,
    epp_image,
    plugin_configs as resolve_plugin_configs,
    plugins_config_file as selected_config_file,
)
from ..spec import DeploymentSpec, RoutingFrontend, RoutingKind


def gateway_name(instance: Instance, cluster: Cluster) -> str:
    """Return the Gateway name while reserving space for its class suffix."""
    return instance.name(
        "gateway",
        max_length=63 - len(cluster.gateway.class_name) - 1,
    )


def standalone_service_name(instance: Instance) -> str:
    """Return the Service that exposes the standalone Envoy frontend."""
    return instance.name("infpool-epp")


def _envoy_container(cluster: Cluster) -> dict:
    return {
        "name": "envoy-proxy",
        "image": cluster.llm_d.envoy,
        "imagePullPolicy": "IfNotPresent",
        "args": [
            "--service-node",
            "envoy-sidecar",
            "--log-level",
            "warn",
            "--concurrency",
            "8",
            "--drain-strategy",
            "immediate",
            "--drain-time-s",
            "60",
            "-c",
            "/etc/envoy/envoy.yaml",
        ],
        "ports": [
            {"containerPort": 8081, "name": "http"},
            {"containerPort": 19001, "name": "envoy-ready"},
        ],
        "readinessProbe": {
            "failureThreshold": 1,
            "httpGet": {"path": "/ready", "port": 19001, "scheme": "HTTP"},
            "periodSeconds": 5,
            "successThreshold": 1,
            "timeoutSeconds": 1,
        },
        "resources": {
            "requests": {"cpu": "4", "memory": "8Gi"},
            "limits": {"memory": "16Gi"},
        },
        "volumeMounts": [
            {
                "name": "envoy-config",
                "mountPath": "/etc/envoy/envoy.yaml",
                "subPath": "envoy.yaml",
                "readOnly": True,
            }
        ],
    }


def _profile_worker_indices(
    spec: DeploymentSpec,
    target_role: str,
    resolved_roles: dict[str, ResolvedRole],
) -> dict[str, tuple[int, ...]]:
    profile_roles = (
        {"prefill": "prefill", "decode": "decode"}
        if spec.routing.kind == RoutingKind.PD
        else {"default": target_role}
    )
    result: dict[str, tuple[int, ...]] = {}
    for profile_name, role_name in profile_roles.items():
        resolved = resolved_roles[role_name]
        if resolved.has_headless_nodes:
            result[profile_name] = resolved.api_nodes
    return result


def render_routing(
    spec: DeploymentSpec, instance: Instance, cluster: Cluster, resolved_roles: dict[str, ResolvedRole],
) -> list[dict]:
    if spec.routing.kind is None:
        raise ValueError("routing kind must be resolved before rendering")
    if spec.routing.target_role is None:
        raise ValueError("routing target role must be resolved before rendering")
    if spec.routing.kind == RoutingKind.DISABLED:
        return []

    target_role = spec.routing.target_role
    ports = resolved_roles[target_role].ports
    infpool_name = instance.name("infpool")
    epp_name = instance.name("infpool-epp")
    epp_role_name = instance.name("infpool-epp-rbac")
    plugin_configs = resolve_plugin_configs(
        spec.routing,
        profile_worker_indices=_profile_worker_indices(spec, target_role, resolved_roles),
    )
    plugins_config_file = selected_config_file(spec.routing)
    standalone = spec.routing.frontend == RoutingFrontend.STANDALONE

    selector = instance.pod_selector(None if spec.routing.kind == RoutingKind.PD else spec.routing.target_role) | {
        "llm-d.ai/inferenceServing": "true",
        "llm-d.ai/deployment": spec.topology.value,
    }
    pool_selector: dict = {"matchLabels": selector}

    epp_container = {
        "name": "epp",
        "image": epp_image(spec.routing, cluster),
        "imagePullPolicy": "Always",
        "args": [
            f"--config-file=/etc/epp/{plugins_config_file}",
            "--grpc-port=9002",
            f"--pool-name={infpool_name}",
            f"--pool-namespace={spec.namespace}",
        ],
        "ports": [{"containerPort": 9002, "name": "grpc"}],
        "volumeMounts": [
            {
                "name": "config",
                "mountPath": f"/etc/epp/{plugins_config_file}",
                "subPath": plugins_config_file,
            }
        ],
        "resources": {
            "requests": {"cpu": "8", "memory": "16Gi"},
            "limits": {"cpu": "8", "memory": "16Gi"},
        },
    }
    containers = [epp_container]
    volumes = [{"name": "config", "configMap": {"name": instance.name("epp-config")}}]
    service_ports = [
        {"name": "grpc", "port": 9002, "protocol": "TCP", "targetPort": 9002}
    ]
    if standalone:
        containers.insert(0, _envoy_container(cluster))
        volumes.append(
            {
                "name": "envoy-config",
                "configMap": {"name": instance.name("envoy-config")},
            }
        )
        service_ports.append(
            {"name": "http", "port": 80, "protocol": "TCP", "targetPort": 8081}
        )

    deployment_spec = {
        "replicas": (
            spec.routing.epp.replicas
            if spec.routing.epp is not None
            else spec.routing.replicas
        ),
        "selector": {"matchLabels": instance.labels("epp")},
        "template": {
            "metadata": {
                "labels": instance.labels("epp") | {"inferencepool": epp_name}
            },
            "spec": {
                "serviceAccountName": epp_name,
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "kubernetes.io/arch",
                                            "operator": "In",
                                            "values": ["amd64"],
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                },
                "containers": containers,
                "volumes": volumes,
            },
        },
    }
    if cluster.pod_defaults.image_pull_secrets:
        deployment_spec["template"]["spec"]["imagePullSecrets"] = (
            cluster.pod_defaults.image_pull_secret_refs()
        )
    if standalone:
        deployment_spec["template"]["spec"]["terminationGracePeriodSeconds"] = 130

    objects = [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": epp_name, "labels": instance.labels("epp")},
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {"name": epp_role_name, "labels": instance.labels("epp")},
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["pods"],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": ["inference.networking.k8s.io"],
                    "resources": ["inferencepools"],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": ["llm-d.ai"],
                    "resources": ["inferenceobjectives", "inferencemodelrewrites"],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": ["inference.networking.x-k8s.io"],
                    "resources": [
                        "inferencemodelrewrites",
                        "inferencemodels",
                        "inferenceobjectives",
                        "inferencepoolimports",
                    ],
                    "verbs": ["get", "list", "watch"],
                },
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {"name": epp_role_name, "labels": instance.labels("epp")},
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": epp_name,
                    "namespace": spec.namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": epp_role_name,
            },
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": instance.name("epp-config"), "labels": instance.labels("routing")},
            "data": {name: yaml.safe_dump(config, sort_keys=False) for name, config in plugin_configs.items()},
        },
        *(
            [
                {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": instance.name("envoy-config"),
                        "labels": instance.labels("envoy"),
                    },
                    "data": {"envoy.yaml": ENVOY_CONFIG},
                }
            ]
            if standalone
            else []
        ),
        {
            "apiVersion": "inference.networking.k8s.io/v1",
            "kind": "InferencePool",
            "metadata": {"name": infpool_name, "labels": instance.labels("routing")},
            "spec": {
                "targetPorts": [{"number": port} for port in ports.public],
                "selector": pool_selector,
                "endpointPickerRef": {"name": epp_name, "kind": "Service", "port": {"number": 9002}},
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": epp_name, "labels": instance.labels("epp")},
            "spec": {
                "selector": instance.labels("epp"),
                "ports": service_ports,
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": epp_name, "labels": instance.labels("epp")},
            "spec": deployment_spec,
        },
    ]

    if standalone:
        return objects

    gateway_resource_name = gateway_name(instance, cluster)
    gateway_pod_spec = {
        "containers": [
            {
                "name": "istio-proxy",
                "resources": {
                    "requests": {"cpu": "8", "memory": "64Gi"},
                    "limits": {"cpu": "8", "memory": "64Gi"},
                },
            }
        ]
    }
    if cluster.pod_defaults.image_pull_secrets:
        gateway_pod_spec["imagePullSecrets"] = (
            cluster.pod_defaults.image_pull_secret_refs()
        )

    objects.extend(
        [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": instance.name("gateway-options"), "labels": instance.labels("gateway")},
            "data": {
                "deployment": yaml.safe_dump(
                    {
                        "spec": {
                            "template": {
                                "spec": gateway_pod_spec
                            }
                        }
                    },
                    sort_keys=False,
                ),
                "service": yaml.safe_dump({"spec": {"type": cluster.gateway.service_type}}, sort_keys=False),
            },
        },
        {
            "apiVersion": "gateway.networking.k8s.io/v1",
            "kind": "Gateway",
            "metadata": {
                "name": gateway_resource_name,
                "labels": instance.labels("gateway") | {"istio.io/enable-inference-extproc": "true"},
            },
            "spec": {
                "infrastructure": {
                    "parametersRef": {
                        "group": "",
                        "kind": "ConfigMap",
                        "name": instance.name("gateway-options"),
                    }
                },
                "gatewayClassName": cluster.gateway.class_name,
                "listeners": [
                    {
                        "name": "default",
                        "port": 80,
                        "protocol": "HTTP",
                        "allowedRoutes": {"namespaces": {"from": "Same"}},
                    }
                ],
            },
        },
        {
            "apiVersion": "gateway.networking.k8s.io/v1",
            "kind": "HTTPRoute",
            "metadata": {"name": instance.name("route"), "labels": instance.labels("route")},
            "spec": {
                "parentRefs": [
                    {
                        "group": "gateway.networking.k8s.io",
                        "kind": "Gateway",
                        "name": gateway_resource_name,
                    }
                ],
                "rules": [
                    {
                        "backendRefs": [
                            {
                                "group": "inference.networking.k8s.io",
                                "kind": "InferencePool",
                                "name": infpool_name,
                                "port": ports.public[0],
                                "weight": 1,
                            }
                        ],
                        "matches": [{"path": {"type": "PathPrefix", "value": "/"}}],
                        "timeouts": {"backendRequest": "0s", "request": "0s"},
                    }
                ],
            },
        },
        {
            "apiVersion": "networking.istio.io/v1",
            "kind": "DestinationRule",
            "metadata": {"name": epp_name, "labels": instance.labels("epp")},
            "spec": {
                "host": epp_name,
                "trafficPolicy": {
                    "connectionPool": {
                        "tcp": {
                            "connectTimeout": "900s",
                            "maxConnectionDuration": "1800s",
                            "maxConnections": 256000,
                        },
                        "http": {
                            "http1MaxPendingRequests": 256000,
                            "http2MaxRequests": 256000,
                            "idleTimeout": "900s",
                            "maxRequestsPerConnection": 256000,
                        },
                    },
                    "tls": {"insecureSkipVerify": True, "mode": "SIMPLE"},
                },
            },
        },
        {
            "apiVersion": "networking.istio.io/v1",
            "kind": "DestinationRule",
            "metadata": {"name": instance.name("infpool-backend"), "labels": instance.labels("routing")},
            "spec": {
                "host": f"{infpool_name}-ip",
                "trafficPolicy": {
                    "connectionPool": {
                        "tcp": {"maxConnections": 256000},
                        "http": {"idleTimeout": "300s"},
                    }
                },
            },
        },
        ]
    )
    return objects
