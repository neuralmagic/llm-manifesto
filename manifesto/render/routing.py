"""Standalone or Gateway API routing manifests for one Manifesto instance."""

from __future__ import annotations

from copy import deepcopy

import yaml

from ..cluster import Cluster
from ..instance import Instance
from ..parallelism import parallel_layout
from ..resolve import resolve_role
from ..spec import DeploymentSpec, RoutingFrontend, RoutingKind, RoutingSpec

_LWS_WORKER_INDEX_LABEL = "leaderworkerset.sigs.k8s.io/worker-index"

_ENVOY_CONFIG = """\
admin:
  address:
    socket_address: {address: 127.0.0.1, port_value: 19000}
static_resources:
  listeners:
    - name: ready
      address:
        socket_address: {address: 0.0.0.0, port_value: 19001}
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: envoy-ready-http
                route_config:
                  name: ready
                  virtual_hosts:
                    - name: ready
                      domains: ["*"]
                      routes:
                        - match: {prefix: /}
                          direct_response: {status: 200}
                http_filters:
                  - name: envoy.filters.http.health_check
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.health_check.v3.HealthCheck
                      pass_through_mode: false
                      headers:
                        - name: ":path"
                          string_match: {exact: /ready}
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
    - name: vllm
      address:
        socket_address: {address: 0.0.0.0, port_value: 8081}
      per_connection_buffer_limit_bytes: 32768
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: http-8081
                route_config:
                  name: vllm
                  virtual_hosts:
                    - name: vllm
                      domains: ["*"]
                      routes:
                        - match: {prefix: /}
                          route:
                            cluster: original_destination_cluster
                            timeout: 86400s
                            idle_timeout: 86400s
                            upgrade_configs:
                              - upgrade_type: websocket
                          typed_per_filter_config:
                            envoy.filters.http.ext_proc:
                              "@type": type.googleapis.com/envoy.config.route.v3.FilterConfig
                              config: {}
                http_filters:
                  - name: envoy.filters.http.ext_proc
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
                      failure_mode_allow: true
                      grpc_service:
                        envoy_grpc: {cluster_name: ext_proc, authority: localhost:9002}
                        timeout: 10s
                      processing_mode:
                        request_header_mode: SEND
                        response_header_mode: SEND
                        request_body_mode: FULL_DUPLEX_STREAMED
                        response_body_mode: FULL_DUPLEX_STREAMED
                        request_trailer_mode: SEND
                        response_trailer_mode: SEND
                      message_timeout: 1000s
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
                      suppress_envoy_headers: true
                use_remote_address: true
                normalize_path: true
                merge_slashes: true
  clusters:
    - name: original_destination_cluster
      type: ORIGINAL_DST
      connect_timeout: 1000s
      lb_policy: CLUSTER_PROVIDED
      circuit_breakers:
        thresholds:
          - {max_connections: 40000, max_pending_requests: 40000, max_requests: 40000}
      original_dst_lb_config:
        use_http_header: true
        http_header_name: x-gateway-destination-endpoint
    - name: ext_proc
      type: STATIC
      connect_timeout: 86400s
      lb_policy: LEAST_REQUEST
      circuit_breakers:
        thresholds:
          - {max_connections: 40000, max_pending_requests: 40000, max_requests: 40000, max_retries: 1024}
      health_checks:
        - timeout: 2s
          interval: 10s
          unhealthy_threshold: 3
          healthy_threshold: 2
          reuse_connection: true
          grpc_health_check:
            service_name: envoy.service.ext_proc.v3.ExternalProcessor
          tls_options:
            alpn_protocols: [h2]
      transport_socket:
        name: envoy.transport_sockets.tls
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext
          common_tls_context:
            validation_context: {}
      typed_extension_protocol_options:
        envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
          "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
          explicit_http_config:
            http2_protocol_options: {}
      load_assignment:
        cluster_name: ext_proc
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address: {address: 127.0.0.1, port_value: 9002}
"""


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


def _default_plugin_config(routing: RoutingSpec) -> dict:
    if routing.plugin_config is not None:
        return routing.plugin_config
    if routing.kind == RoutingKind.PD:
        config = {
            "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {"type": "disagg-headers-handler"},
                {"type": "prefill-filter"},
                {"type": "decode-filter"},
                {"type": "prefix-cache-scorer"},
                {"type": "active-request-scorer"},
                {"type": "queue-scorer"},
                {"type": "always-disagg-pd-decider"},
                {"type": "disagg-profile-handler", "parameters": {"deciders": {"prefill": "always-disagg-pd-decider"}}},
                {"type": "weighted-random-picker", "name": "prefill-picker", "parameters": {"threshold": 0.1, "hashBlockSize": 5}},
                {"type": "weighted-random-picker", "name": "decode-picker", "parameters": {"threshold": 0.1}},
            ],
            "schedulingProfiles": [
                {
                    "name": "prefill",
                    "plugins": [
                        {"pluginRef": "prefill-filter"},
                        {"pluginRef": "prefix-cache-scorer", "weight": 3},
                        {"pluginRef": "active-request-scorer", "weight": 2},
                        {"pluginRef": "queue-scorer", "weight": 2},
                        {"pluginRef": "prefill-picker"},
                    ],
                },
                {
                    "name": "decode",
                    "plugins": [
                        {"pluginRef": "decode-filter"},
                        {"pluginRef": "active-request-scorer", "weight": 2},
                        {"pluginRef": "decode-picker"},
                    ],
                },
            ],
        }
    else:
        config = {
            "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {"type": "active-request-scorer"},
                {"type": "queue-scorer"},
                {"type": "weighted-random-picker", "parameters": {"threshold": 0.1}},
            ],
            "schedulingProfiles": [
                {
                    "name": "default",
                    "plugins": [
                        {"pluginRef": "active-request-scorer", "weight": 2},
                        {"pluginRef": "queue-scorer", "weight": 2},
                        {"pluginRef": "weighted-random-picker"},
                    ],
                }
            ],
        }
    return config


def _filter_api_servers(
    config: dict,
    profile_name: str,
    worker_indices: tuple[int, ...],
) -> None:
    """Restrict one private scheduling config to API-serving TP-group leaders."""
    plugins = config.setdefault("plugins", [])
    filter_name = f"manifesto-{profile_name}-api-server-filter"
    if not any(plugin.get("name") == filter_name for plugin in plugins):
        plugins.append(
            {
                "type": "by-label",
                "name": filter_name,
                "parameters": {
                    "label": _LWS_WORKER_INDEX_LABEL,
                    "validValues": [str(index) for index in worker_indices],
                    "allowsNoLabel": False,
                },
            }
        )

    profiles = [
        profile
        for profile in config.get("schedulingProfiles", [])
        if profile.get("name") == profile_name
    ]
    if not profiles:
        raise ValueError(
            f"cross-node TP routing requires a {profile_name} scheduling profile"
        )
    for profile in profiles:
        profile_plugins = profile.setdefault("plugins", [])
        if not any(
            plugin.get("pluginRef") == filter_name
            for plugin in profile_plugins
        ):
            role_filter_index = next(
                (
                    index
                    for index, plugin in enumerate(profile_plugins)
                    if plugin.get("pluginRef") == f"{profile_name}-filter"
                ),
                -1,
            )
            profile_plugins.insert(
                role_filter_index + 1,
                {"pluginRef": filter_name},
            )


def _plugin_configs(
    routing: RoutingSpec,
    *,
    profile_worker_indices: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, str]:
    if routing.epp is not None and routing.epp.plugin_configs:
        source_configs = routing.epp.plugin_configs
    else:
        source_configs = {"plugins.yaml": _default_plugin_config(routing)}
    configs = deepcopy(source_configs)
    if profile_worker_indices:
        selected_config = _plugins_config_file(routing)
        selected = configs[selected_config]
        for profile_name, worker_indices in profile_worker_indices.items():
            _filter_api_servers(
                selected,
                profile_name,
                worker_indices,
            )
    return {
        name: yaml.safe_dump(config, sort_keys=False)
        for name, config in configs.items()
    }


def _plugins_config_file(routing: RoutingSpec) -> str:
    return routing.epp.plugins_config_file if routing.epp is not None else "plugins.yaml"


def _profile_worker_indices(
    spec: DeploymentSpec,
    target_role: str,
) -> dict[str, tuple[int, ...]]:
    profile_roles = (
        {"prefill": "prefill", "decode": "decode"}
        if spec.routing.kind == RoutingKind.PD
        else {"default": target_role}
    )
    result: dict[str, tuple[int, ...]] = {}
    for profile_name, role_name in profile_roles.items():
        layout = parallel_layout(spec.role(role_name))
        if layout.cross_node_tp:
            result[profile_name] = layout.serving_worker_indices
    return result


def render_routing(spec: DeploymentSpec, instance: Instance, cluster: Cluster) -> list[dict]:
    if spec.routing.kind is None:
        raise ValueError("routing kind must be resolved before rendering")
    if spec.routing.target_role is None:
        raise ValueError("routing target role must be resolved before rendering")
    if spec.routing.kind == RoutingKind.DISABLED:
        return []

    target_role = spec.routing.target_role
    role = spec.role(target_role)
    ports = resolve_role(spec, instance, cluster, role).ports
    infpool_name = instance.name("infpool")
    epp_name = instance.name("infpool-epp")
    epp_role_name = instance.name("infpool-epp-rbac")
    plugin_configs = _plugin_configs(
        spec.routing,
        profile_worker_indices=_profile_worker_indices(spec, target_role),
    )
    plugins_config_file = _plugins_config_file(spec.routing)
    standalone = spec.routing.frontend == RoutingFrontend.STANDALONE

    selector = instance.pod_selector(None if spec.routing.kind == RoutingKind.PD else spec.routing.target_role) | {
        "llm-d.ai/inferenceServing": "true",
        "llm-d.ai/deployment": spec.topology.value,
    }
    pool_selector: dict = {"matchLabels": selector}

    epp_container = {
        "name": "epp",
        "image": (
            spec.routing.epp.image
            if spec.routing.epp is not None and spec.routing.epp.image is not None
            else spec.routing.epp_image or cluster.llm_d.epp
        ),
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
            "data": plugin_configs,
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
                    "data": {"envoy.yaml": _ENVOY_CONFIG},
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
                                "spec": {
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
