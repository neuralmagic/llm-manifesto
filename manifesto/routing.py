"""llm-d routing policy and Envoy configuration shared by deployment backends."""

from __future__ import annotations

from copy import deepcopy

import yaml

from .cluster import Cluster
from .dp_ports import RolePorts
from .spec import RoutingKind, RoutingSpec

_LWS_WORKER_INDEX_LABEL = "leaderworkerset.sigs.k8s.io/worker-index"

ENVOY_CONFIG = """\
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


def _default_plugin_config(routing: RoutingSpec) -> dict:
    if routing.plugin_config is not None:
        return routing.plugin_config
    if routing.kind == RoutingKind.PD:
        config = {
            "apiVersion": "llm-d.ai/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {"type": "vllmhttp-parser"},
                {"type": "prefill-filter"},
                {"type": "decode-filter"},
                {"type": "prefix-cache-scorer"},
                {"type": "active-request-scorer"},
                {"type": "queue-scorer"},
                {"type": "always-disagg-pd-decider"},
                {"type": "disagg-profile-handler", "parameters": {"deciders": {"prefill": "always-disagg-pd-decider"}}},
                {"type": "weighted-random-picker", "name": "prefill-picker"},
                {"type": "weighted-random-picker", "name": "decode-picker"},
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
            "apiVersion": "llm-d.ai/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {"type": "active-request-scorer"},
                {"type": "queue-scorer"},
                {"type": "weighted-random-picker"},
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
    """Restrict one scheduling profile to nodes that expose an API."""
    plugins = config.setdefault("plugins", [])
    filter_name = f"manifesto-{profile_name}-api-server-filter"
    if not any(plugin.get("name") == filter_name for plugin in plugins):
        plugins.append(
            {
                "type": "label-selector-filter",
                "name": filter_name,
                "parameters": {
                    "matchExpressions": [{
                        "key": _LWS_WORKER_INDEX_LABEL,
                        "operator": "In",
                        "values": [str(index) for index in worker_indices],
                    }],
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
            f"API endpoint filtering requires a {profile_name} scheduling profile"
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


def plugin_configs(
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
        selected_config = plugins_config_file(routing)
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


def plugins_config_file(routing: RoutingSpec) -> str:
    return routing.epp.plugins_config_file if routing.epp is not None else "plugins.yaml"



def epp_image(routing: RoutingSpec, cluster: Cluster) -> str:
    if routing.epp and routing.epp.image:
        return routing.epp.image
    return routing.epp_image or cluster.llm_d.epp


def proxy_args(ports: RolePorts) -> list[str]:
    return [
        f"--port={ports.public[0]}",
        f"--model-server-port={ports.backend[0]}",
        f"--data-parallel-size={ports.rank_count}",
        "--secure-proxy=false",
        "--kv-connector=nixlv2",
    ]
