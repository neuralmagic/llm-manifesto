"""Top-level render pipeline that emits one YAML-ready object list per deployment."""

from __future__ import annotations

import io

import yaml

from .base import render_dcgm_metrics_configmap, render_model_server_service, render_openshift_scc_binding, render_service_account
from .idle_shutdown import render_idle_shutdown
from .lws import render_accelerator_claim_template, render_workload
from .routing import render_routing
from ..cluster import Cluster
from ..instance import Instance
from ..parallelism import parallel_layout
from ..resolve import resolve_role
from ..spec import DeploymentSpec


class LiteralString(str):
    pass


def _literal_string_representer(dumper: yaml.SafeDumper, data: LiteralString) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.SafeDumper.add_representer(LiteralString, _literal_string_representer)


def _literal_multiline_strings(value):
    if isinstance(value, str) and "\n" in value:
        return LiteralString(value)
    if isinstance(value, list):
        return [_literal_multiline_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _literal_multiline_strings(item) for key, item in value.items()}
    return value


def render(
    spec: DeploymentSpec,
    *,
    user: str,
    cluster: Cluster,
    routing_only: bool = False,
) -> list[dict]:
    instance = Instance(
        user=user,
        release=spec.release,
        include_user_in_name=cluster.naming.user_prefix,
    )
    if routing_only:
        return [
            *render_idle_shutdown(spec, instance, cluster),
            *render_routing(spec, instance, cluster),
        ]

    objects = []
    if cluster.openshift.scc:
        objects.append(render_service_account(instance))
        objects.append(
            render_openshift_scc_binding(instance, namespace=spec.namespace, scc=cluster.openshift.scc)
        )
    if spec.roles and "dcgm-exporter" in spec.runtime.sidecars:
        objects.append(render_dcgm_metrics_configmap(instance))
    for role in spec.roles:
        claim_template = render_accelerator_claim_template(
            spec, instance, cluster, role
        )
        if claim_template is not None:
            objects.append(claim_template)
        objects.append(render_workload(spec, instance, cluster, role))
        resolved = resolve_role(spec, instance, cluster, role)
        objects.append(
            render_model_server_service(
                instance,
                role.name,
                resolved.ports,
                leader_only=parallel_layout(role).cross_node_tp,
            )
        )
    objects.extend(render_idle_shutdown(spec, instance, cluster))
    objects.extend(render_routing(spec, instance, cluster))
    return objects


def render_to_yaml(objects: list[dict], *, header: list[str] | None = None) -> str:
    stream = io.StringIO()
    for line in header or []:
        stream.write(f"# {line}\n")
    yaml.safe_dump_all(_literal_multiline_strings(objects), stream, sort_keys=False, explicit_start=True)
    return stream.getvalue()
