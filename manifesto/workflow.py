"""Higher-level workflow helpers for the manifesto CLI."""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .cluster import Cluster, load_cluster
from .catalog import ROOT, catalog_entries, config_home, resolve_catalog_path
from .instance import Instance
from .overrides import load_routing_profile
from .render import render, render_to_yaml
from .render.bootstrap import render_bootstrap
from .spec import EppSpec, RoutingKind, load_spec


HF_SECRET_NAME = "hf-secret"
HF_SECRET_KEY = "HF_TOKEN"
KUEUE_QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
LWS_RESOURCE_TYPE = "leaderworkersets.leaderworkerset.x-k8s.io"
KUEUE_REQUIRED_RESOURCE_TYPES = {
    "localqueues.kueue.x-k8s.io",
    "clusterqueues.kueue.x-k8s.io",
    "resourceflavors.kueue.x-k8s.io",
    "workloads.kueue.x-k8s.io",
}
# Stateless teardown allowlist. Keep this in sync with the label-bearing objects
# emitted by manifesto.render. Values are kubectl resource names as reported by
# ``kubectl api-resources -o name``.
MANAGED_RESOURCE_TYPES = {
    ("v1", "ConfigMap"): "configmaps",
    ("v1", "Service"): "services",
    ("v1", "ServiceAccount"): "serviceaccounts",
    ("apps/v1", "Deployment"): "deployments.apps",
    ("gateway.networking.k8s.io/v1", "Gateway"): "gateways.gateway.networking.k8s.io",
    ("gateway.networking.k8s.io/v1", "HTTPRoute"): "httproutes.gateway.networking.k8s.io",
    ("inference.networking.k8s.io/v1", "InferencePool"): "inferencepools.inference.networking.k8s.io",
    ("leaderworkerset.x-k8s.io/v1", "LeaderWorkerSet"): "leaderworkersets.leaderworkerset.x-k8s.io",
    ("networking.istio.io/v1", "DestinationRule"): "destinationrules.networking.istio.io",
    ("rbac.authorization.k8s.io/v1", "Role"): "roles.rbac.authorization.k8s.io",
    ("rbac.authorization.k8s.io/v1", "RoleBinding"): "rolebindings.rbac.authorization.k8s.io",
}
POD_RESOURCE_TYPE = "pods"
MANIFESTO_SELECTOR = "app.kubernetes.io/name=manifesto"

# Everything discovery lists. Asking for a type this cluster does not serve is
# harmless -- list_objects tolerates that one error and reads it as zero objects --
# so there is nothing to gain from a `kubectl api-resources` round trip first:
# the lists run concurrently, so pruning the set would not save any wall clock.
DISCOVERY_RESOURCE_TYPES = (*sorted(set(MANAGED_RESOURCE_TYPES.values())), POD_RESOURCE_TYPE)

# Discovery talks to clusters whose API round trips can be seconds long, so every
# read is bounded, retried on transient faults, and fanned out across resource
# types instead of walking them one at a time inside a single kubectl process.
DEFAULT_KUBECTL_TIMEOUT = 120.0
DEFAULT_KUBECTL_RETRIES = 2
# One worker per managed type, so adding a type never reintroduces a serial round.
MAX_DISCOVERY_WORKERS = len(MANAGED_RESOURCE_TYPES) + 1
COMPLETION_KUBECTL_TIMEOUT = 3.0

TRANSIENT_KUBECTL_ERRORS = (
    "connection refused",
    "connection reset",
    "context deadline exceeded",
    "client.timeout",
    "i/o timeout",
    "tls handshake timeout",
    "unexpected eof",
    "etcdserver: request timed out",
    "the server is currently unable to handle the request",
    "the server was unable to return a response in the time allotted",
    "too many requests",
    "temporary failure in name resolution",
    "no route to host",
)
# Deliberately narrow: a generic 404 can also mean a CRD is mid-upgrade, and
# treating that as "no such objects" would hide live resources from teardown.
MISSING_RESOURCE_ERRORS = ("the server doesn't have a resource type",)

TRACE_OFF_VALUES = frozenset({"", "0", "false", "no", "off"})

_UNSET = object()
_kubectl_limits: dict[str, float | int | None] = {}


class WorkflowError(RuntimeError):
    """Expected error that should be printed without a traceback."""

    def __init__(self, message: str, *, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeConfig:
    user: str
    namespace: str
    cluster_path: str | None
    render_out: Path
    context: str | None = None

    @classmethod
    def from_args(cls, args, *, require_cluster: bool = True) -> "RuntimeConfig":
        load_dotenv()
        user = resolve_user(getattr(args, "user", None))
        context = getattr(args, "context", None)
        namespace = resolve_namespace(getattr(args, "namespace", None), context=context)
        cluster_path = (
            resolve_cluster(getattr(args, "cluster", None), context=context)
            if require_cluster
            else getattr(args, "cluster", None)
        )
        render_out = Path(
            getattr(args, "output", None)
            or os.environ.get("MANIFESTO_RENDER_OUT", f"/tmp/{user}-manifesto.yaml")
        )
        return cls(
            user=user,
            namespace=namespace,
            cluster_path=cluster_path,
            render_out=render_out,
            context=context,
        )

    def kubectl_base(self) -> list[str]:
        command = ["kubectl"]
        if self.context:
            command.extend(["--context", self.context])
        return command

    def kubectl(self) -> list[str]:
        return [*self.kubectl_base(), "-n", self.namespace]


@dataclass(frozen=True)
class LiveResource:
    api_version: str
    kind: str
    name: str
    labels: dict[str, str]
    creation_timestamp: str | None = None
    deletion_timestamp: str | None = None
    ready: bool = False

    @classmethod
    def from_object(cls, obj: dict) -> "LiveResource":
        metadata = obj.get("metadata", {})
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in obj.get("status", {}).get("conditions", [])
        )
        return cls(
            api_version=obj.get("apiVersion", ""),
            kind=obj.get("kind", ""),
            name=metadata.get("name", ""),
            labels=metadata.get("labels", {}),
            creation_timestamp=metadata.get("creationTimestamp"),
            deletion_timestamp=metadata.get("deletionTimestamp"),
            ready=ready,
        )

    @property
    def instance_id(self) -> str | None:
        return self.labels.get("app.kubernetes.io/instance")

    @property
    def resource_type(self) -> str:
        if self.kind == "Pod":
            return POD_RESOURCE_TYPE
        resource_type = MANAGED_RESOURCE_TYPES.get((self.api_version, self.kind))
        if not resource_type:
            raise WorkflowError(f"unsupported managed resource: {self.api_version} {self.kind}")
        return resource_type

    @property
    def kubectl_ref(self) -> str:
        return f"{self.resource_type}/{self.name}"


@dataclass(frozen=True)
class ServerRecord:
    instance_id: str
    resources: tuple[LiveResource, ...]

    @property
    def pods(self) -> tuple[LiveResource, ...]:
        return tuple(resource for resource in self.resources if resource.kind == "Pod")

    @property
    def model(self) -> str:
        return next(
            (resource.labels["llm-d.ai/model"] for resource in self.resources if "llm-d.ai/model" in resource.labels),
            "-",
        )

    @property
    def roles(self) -> str:
        roles = sorted(
            {resource.labels["llm-d.ai/role"] for resource in self.resources if "llm-d.ai/role" in resource.labels}
        )
        return ",".join(roles) or "-"

    @property
    def pod_readiness(self) -> str:
        return f"{sum(pod.ready for pod in self.pods)}/{len(self.pods)}"

    @property
    def state(self) -> str:
        if any(resource.deletion_timestamp for resource in self.resources):
            return "Stopping"
        if not self.pods:
            return "Pending"
        ready = sum(pod.ready for pod in self.pods)
        if ready == len(self.pods):
            return "Ready"
        if ready:
            return "Degraded"
        return "Starting"

    @property
    def age(self) -> str:
        timestamps = [resource.creation_timestamp for resource in self.resources if resource.creation_timestamp]
        if not timestamps:
            return "-"
        try:
            created = min(datetime.fromisoformat(value.replace("Z", "+00:00")) for value in timestamps)
        except ValueError:
            return "-"
        seconds = max(0, int((datetime.now(timezone.utc) - created).total_seconds()))
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m"
        if seconds < 86400:
            return f"{seconds // 3600}h"
        return f"{seconds // 86400}d"

    def as_dict(self) -> dict:
        return {
            "instance": self.instance_id,
            "state": self.state,
            "model": self.model,
            "roles": self.roles.split(",") if self.roles != "-" else [],
            "pods": {"ready": sum(pod.ready for pod in self.pods), "total": len(self.pods)},
            "age": self.age,
            "resources": [
                {"apiVersion": resource.api_version, "kind": resource.kind, "name": resource.name}
                for resource in sorted(self.resources, key=lambda item: (item.kind, item.name))
            ],
        }


def load_dotenv(path: Path | None = None) -> None:
    paths = [path] if path is not None else [config_home() / ".env", ROOT / ".env"]
    for env_path in paths:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_model(value: str) -> str:
    return resolve_catalog_path(value, "models")


def resolve_routing(value: str) -> str:
    return resolve_catalog_path(value, "routing")


def resolve_user(explicit: str | None = None) -> str:
    return explicit or os.environ.get("USER") or "dev"


def resolve_namespace(explicit: str | None = None, *, context: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("MANIFESTO_NAMESPACE"):
        return os.environ["MANIFESTO_NAMESPACE"]
    command = ["kubectl"]
    if context:
        command.extend(["--context", context])
    namespace = capture(
        [*command, "config", "view", "--minify", "-o", "jsonpath={..namespace}"],
        check=False,
    )
    return namespace.strip() or "default"


def resolve_cluster(explicit: str | None = None, *, context: str | None = None) -> str:
    if explicit:
        return resolve_catalog_path(explicit, "clusters")
    if os.environ.get("MANIFESTO_CLUSTER"):
        return resolve_catalog_path(os.environ["MANIFESTO_CLUSTER"], "clusters")
    mapping = os.environ.get("MANIFESTO_CLUSTER_MAP", "")
    selected_context = context or capture(
        ["kubectl", "config", "current-context"], check=False
    ).strip()
    command = ["kubectl"]
    if context:
        command.extend(["--context", context])
    kube_cluster = capture(
        [*command, "config", "view", "--minify", "-o", "jsonpath={.clusters[0].name}"],
        check=False,
    ).strip()
    if mapping:
        for entry in mapping.split(","):
            key, sep, value = entry.partition("=")
            if sep and key.strip() in {selected_context, kube_cluster}:
                return resolve_catalog_path(value.strip(), "clusters")
    for name in (selected_context, kube_cluster):
        if not name:
            continue
        candidate = resolve_catalog_path(name, "clusters")
        if Path(candidate).exists():
            return candidate
    raise WorkflowError(
        "No cluster profile configured. Pass --cluster, set MANIFESTO_CLUSTER, "
        "add the current kube context to MANIFESTO_CLUSTER_MAP, or create "
        f"{config_home() / 'clusters' / '<context>.yaml'}.",
        code=2,
    )


def load_runtime_cluster(config: RuntimeConfig, args):
    if not config.cluster_path:
        raise WorkflowError("No cluster profile configured.", code=2)
    return load_cluster_with_overrides(config.cluster_path, args)


def load_cluster_with_overrides(cluster_path: str, args):
    return load_cluster(cluster_path).with_path_overrides(
        user_root=getattr(args, "user_root", None),
        log_root=getattr(args, "log_root", None),
        cache_root=getattr(args, "cache_root", None),
    )


def apply_runtime_overrides(spec, args, config: RuntimeConfig) -> None:
    spec.namespace = config.namespace
    if getattr(args, "accelerator", None):
        spec.accelerator = args.accelerator
    if getattr(args, "vllm_env", None) is not None:
        spec.runtime.vllm_env = args.vllm_env
    if getattr(args, "idle_timeout_minutes", None) is not None:
        spec.runtime.idle_shutdown.enabled = True
        spec.runtime.idle_shutdown.timeout_minutes = args.idle_timeout_minutes
    if getattr(args, "no_idle_shutdown", False):
        spec.runtime.idle_shutdown.enabled = False
    spec.runtime.pre_launch.extend(getattr(args, "pre_launch", None) or [])
    routing_profile = getattr(args, "routing_profile", None) or os.environ.get(
        "MANIFESTO_ROUTING_PROFILE"
    )
    if routing_profile:
        path, plugin_config = load_routing_profile(routing_profile)
        current = spec.routing.epp
        spec.routing.epp = EppSpec(
            image=current.image if current is not None else spec.routing.epp_image,
            replicas=current.replicas if current is not None else spec.routing.replicas,
            plugins_config_file=path.name,
            plugin_configs={path.name: plugin_config},
        )


def render_manifest(
    args,
    config: RuntimeConfig,
    *,
    routing_only: bool = False,
    cluster: Cluster | None = None,
) -> str:
    cluster = cluster or load_runtime_cluster(config, args)
    spec = load_spec(resolve_model(args.spec), cluster)
    apply_runtime_overrides(spec, args, config)
    return render_to_yaml(
        render(spec, user=config.user, cluster=cluster, routing_only=routing_only),
        header=manifest_header(args, config, routing_only=routing_only),
    )


def manifest_header(args, config: RuntimeConfig, *, routing_only: bool) -> list[str]:
    if not config.cluster_path:
        raise WorkflowError("No cluster profile configured.", code=2)
    command = [
        "manifesto",
        "render",
        "routing" if routing_only else "manifest",
        resolve_model(args.spec),
        "--cluster",
        config.cluster_path,
        "--namespace",
        config.namespace,
        "--user",
        config.user,
    ]
    if getattr(args, "accelerator", None):
        command.extend(["--gpu", args.accelerator])
    routing_profile = getattr(args, "routing_profile", None) or os.environ.get(
        "MANIFESTO_ROUTING_PROFILE"
    )
    if routing_profile:
        command.extend(["--routing-profile", routing_profile])
    for name in ("user_root", "log_root", "cache_root", "vllm_env"):
        value = getattr(args, name, None)
        if value:
            command.extend([f"--{name.replace('_', '-')}", value])
    for hook in getattr(args, "pre_launch", None) or []:
        command.extend(["--pre-launch", hook])
    if getattr(args, "idle_timeout_minutes", None) is not None:
        command.extend(["--idle-timeout", f"{args.idle_timeout_minutes}m"])
    if getattr(args, "no_idle_shutdown", False):
        command.append("--no-idle-shutdown")
    return [
        "Generated by:",
        f"  {shlex.join(command)}",
        "Source: https://github.com/neuralmagic/llm-manifesto",
        "Safe to edit before applying.",
    ]


def render_to_file(args) -> Path:
    config = RuntimeConfig.from_args(args)
    config.render_out.parent.mkdir(parents=True, exist_ok=True)
    config.render_out.write_text(render_manifest(args, config))
    return config.render_out


def deploy(
    args,
    *,
    routing_only: bool = False,
    config: RuntimeConfig | None = None,
    cluster: Cluster | None = None,
) -> int:
    config = config or RuntimeConfig.from_args(args)
    manifest = render_manifest(args, config, routing_only=routing_only, cluster=cluster)
    if not routing_only:
        require_hf_token()
        objects = parse_manifest(manifest)
        preflight_workloads(config, objects)
        transitions = plan_workload_transitions(config, objects)
        rc = sync_hf_secret(config)
        if rc:
            return rc
        rc = execute_workload_transitions(config, transitions)
        if rc:
            return rc
    return run([*config.kubectl(), "apply", "-f", "-"], input_text=manifest)


def parse_manifest(manifest: str) -> list[dict]:
    try:
        return [obj for obj in yaml.safe_load_all(manifest) if obj]
    except yaml.YAMLError as exc:
        raise WorkflowError(f"rendered manifest is invalid YAML: {exc}") from exc


def _model_workloads(objects: list[dict]) -> list[dict]:
    return [
        obj
        for obj in objects
        if obj.get("kind") in {"Deployment", "LeaderWorkerSet"}
        and obj.get("metadata", {})
        .get("labels", {})
        .get("llm-d.ai/inferenceServing")
        == "true"
    ]


def _api_resources(config: RuntimeConfig, group: str) -> set[str]:
    raw = capture(
        [
            *config.kubectl_base(),
            "api-resources",
            f"--api-group={group}",
            "-o",
            "name",
        ]
    )
    return set(raw.splitlines())


def _condition_active(resource: dict) -> bool:
    return any(
        condition.get("type") == "Active" and condition.get("status") == "True"
        for condition in resource.get("status", {}).get("conditions", [])
    )


def _get_json(cmd: list[str], description: str) -> dict:
    try:
        raw = capture(cmd)
        resource = json.loads(raw)
    except (WorkflowError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Kueue preflight failed: cannot read {description}: {exc}") from exc
    if not isinstance(resource, dict) or not resource:
        raise WorkflowError(f"Kueue preflight failed: {description} was not found")
    return resource


def _format_requests(requests: dict) -> str:
    if not requests:
        return "none"
    return ", ".join(f"{name}={value}" for name, value in sorted(requests.items()))


def _surface_lws_requests(workload: dict) -> None:
    name = workload["metadata"]["name"]
    template = workload["spec"]["leaderWorkerTemplate"]
    replicas = workload["spec"].get("replicas", 1)
    size = template.get("size", 1)
    queue = workload["metadata"].get("labels", {}).get(KUEUE_QUEUE_LABEL, "unmanaged")
    print(
        f"Kueue preflight {name}: queue={queue}, replicas={replicas}, "
        f"pods-per-replica={size}",
        file=sys.stderr,
    )
    pod_templates = [("worker", template["workerTemplate"])]
    if "leaderTemplate" in template:
        pod_templates.append(("leader", template["leaderTemplate"]))
    for pod_set, pod_template in pod_templates:
        pod_spec = pod_template.get("spec", {})
        for field in ("resources", "overhead"):
            values = pod_spec.get(field, {})
            requests = values.get("requests", {}) if field == "resources" else values
            if requests:
                print(
                    f"  {pod_set}/pod/{field}: requests={_format_requests(requests)}",
                    file=sys.stderr,
                )
        for container_kind in ("initContainers", "containers"):
            for container in pod_spec.get(container_kind, []):
                print(
                    f"  {pod_set}/{container_kind}/{container.get('name', 'unnamed')}: "
                    f"requests={_format_requests(container.get('resources', {}).get('requests', {}))}",
                    file=sys.stderr,
                )


def _lws_request_resources(workload: dict) -> set[str]:
    resources: set[str] = set()
    template = workload["spec"]["leaderWorkerTemplate"]
    pod_templates = [template["workerTemplate"]]
    if "leaderTemplate" in template:
        pod_templates.append(template["leaderTemplate"])
    for pod_template in pod_templates:
        pod_spec = pod_template.get("spec", {})
        resources.update(pod_spec.get("resources", {}).get("requests", {}))
        resources.update(pod_spec.get("overhead", {}))
        for container_kind in ("initContainers", "containers"):
            for container in pod_spec.get(container_kind, []):
                resources.update(container.get("resources", {}).get("requests", {}))
    return resources


def _cluster_queue_covered_resources(cluster_queue: dict) -> set[str]:
    covered: set[str] = set()
    for group in cluster_queue.get("spec", {}).get("resourceGroups", []):
        covered.update(group.get("coveredResources", []))
    return covered


def preflight_workloads(config: RuntimeConfig, objects: list[dict]) -> None:
    lws_objects = [obj for obj in _model_workloads(objects) if obj["kind"] == "LeaderWorkerSet"]
    if not lws_objects:
        return
    served = _api_resources(config, "leaderworkerset.x-k8s.io")
    if LWS_RESOURCE_TYPE not in served:
        raise WorkflowError(
            "workload preflight failed: LeaderWorkerSet API is not served "
            f"({LWS_RESOURCE_TYPE})"
        )

    for workload in lws_objects:
        _surface_lws_requests(workload)
    queues = {
        obj["metadata"].get("labels", {}).get(KUEUE_QUEUE_LABEL)
        for obj in lws_objects
    } - {None}
    if not queues:
        return

    kueue_resources = _api_resources(config, "kueue.x-k8s.io")
    missing = KUEUE_REQUIRED_RESOURCE_TYPES - kueue_resources
    if missing:
        raise WorkflowError(
            "Kueue preflight failed: required APIs are not served: "
            + ", ".join(sorted(missing))
        )
    for queue in sorted(queues):
        local_queue = _get_json(
            [*config.kubectl(), "get", "localqueue", queue, "-o", "json"],
            f"LocalQueue {config.namespace}/{queue}",
        )
        if not _condition_active(local_queue):
            raise WorkflowError(
                f"Kueue preflight failed: LocalQueue {config.namespace}/{queue} "
                "is not Active=True"
            )
        cluster_queue_name = local_queue.get("spec", {}).get("clusterQueue")
        if not cluster_queue_name:
            raise WorkflowError(
                f"Kueue preflight failed: LocalQueue {config.namespace}/{queue} "
                "does not reference a ClusterQueue"
            )
        cluster_queue = _get_json(
            [
                *config.kubectl_base(),
                "get",
                "clusterqueue",
                cluster_queue_name,
                "-o",
                "json",
            ],
            f"ClusterQueue {cluster_queue_name}",
        )
        if not _condition_active(cluster_queue):
            raise WorkflowError(
                f"Kueue preflight failed: ClusterQueue {cluster_queue_name} "
                "is not Active=True"
            )
        requested = set().union(
            *(
                _lws_request_resources(workload)
                for workload in lws_objects
                if workload["metadata"].get("labels", {}).get(KUEUE_QUEUE_LABEL)
                == queue
            )
        )
        uncovered = requested - _cluster_queue_covered_resources(cluster_queue)
        if uncovered:
            raise WorkflowError(
                f"Kueue preflight failed: ClusterQueue {cluster_queue_name} does "
                "not cover rendered PodSet resources: "
                + ", ".join(sorted(uncovered))
            )


@dataclass(frozen=True)
class WorkloadTransition:
    resource_type: str
    name: str
    reason: str


def _live_named_resource(
    config: RuntimeConfig,
    resource_type: str,
    name: str,
) -> dict | None:
    raw = capture(
        [
            *config.kubectl(),
            "get",
            resource_type,
            name,
            "-o",
            "json",
            "--ignore-not-found",
        ],
        tolerate=MISSING_RESOURCE_ERRORS,
    )
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkflowError(
            f"kubectl returned invalid JSON for {resource_type}/{name}: {exc}"
        ) from exc


def plan_workload_transitions(
    config: RuntimeConfig,
    objects: list[dict],
) -> list[WorkloadTransition]:
    transitions: list[WorkloadTransition] = []
    for desired in _model_workloads(objects):
        name = desired["metadata"]["name"]
        desired_kind = desired["kind"]
        deployment = _live_named_resource(config, "deployments.apps", name)
        lws = _live_named_resource(config, LWS_RESOURCE_TYPE, name)
        if desired_kind == "LeaderWorkerSet":
            if deployment is not None:
                transitions.append(
                    WorkloadTransition("deployments.apps", name, "Deployment to LeaderWorkerSet")
                )
            if lws is not None:
                desired_queue = desired["metadata"].get("labels", {}).get(KUEUE_QUEUE_LABEL)
                live_queue = lws.get("metadata", {}).get("labels", {}).get(KUEUE_QUEUE_LABEL)
                if live_queue != desired_queue:
                    transitions.append(
                        WorkloadTransition(
                            LWS_RESOURCE_TYPE,
                            name,
                            f"queue changed from {live_queue or 'unmanaged'} "
                            f"to {desired_queue or 'unmanaged'}",
                        )
                    )
        elif lws is not None:
            transitions.append(
                WorkloadTransition(LWS_RESOURCE_TYPE, name, "LeaderWorkerSet to Deployment")
            )
    return transitions


def execute_workload_transitions(
    config: RuntimeConfig,
    transitions: list[WorkloadTransition],
) -> int:
    for transition in transitions:
        print(
            f"Recreating {transition.name}: {transition.reason}; serving will be interrupted.",
            file=sys.stderr,
        )
        rc = run(
            [
                *config.kubectl(),
                "delete",
                f"{transition.resource_type}/{transition.name}",
                "--ignore-not-found=true",
                "--wait=true",
            ]
        )
        if rc:
            return rc
    return 0


def render_bootstrap_manifest(args, config: RuntimeConfig | None = None) -> str:
    config = config or RuntimeConfig.from_args(args)
    cluster = load_runtime_cluster(config, args)
    resources = render_bootstrap(cluster, config.namespace)
    if not resources:
        raise WorkflowError(
            f"Cluster profile {config.cluster_path} declares no bootstrap resources.",
            code=2,
        )
    return render_to_yaml(resources)


def bootstrap(args) -> int:
    config = RuntimeConfig.from_args(args)
    return run(
        [*config.kubectl(), "apply", "-f", "-"],
        input_text=render_bootstrap_manifest(args, config),
    )


def require_hf_token() -> str:
    token = os.environ.get(HF_SECRET_KEY, "").strip()
    if not token:
        raise WorkflowError(
            "HF_TOKEN is not configured. Set a fine-grained read token in "
            f"{config_home() / '.env'} or the environment before deploying.",
            code=2,
        )
    return token


def sync_hf_secret(config: RuntimeConfig) -> int:
    token = require_hf_token()
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": HF_SECRET_NAME, "namespace": config.namespace},
        "type": "Opaque",
        "stringData": {HF_SECRET_KEY: token},
    }
    return run(
        [*config.kubectl(), "apply", "-f", "-"],
        input_text=json.dumps(secret),
    )


def _objects_from_list(raw: str) -> list[dict]:
    """Parse a kubectl list into its items.

    A payload that is not a list is an error rather than an empty result: this
    feeds teardown, where "could not read it" must never look like "nothing there".
    """

    payload = json.loads(raw)
    if not isinstance(payload, dict) or "items" not in payload:
        raise WorkflowError("kubectl returned invalid discovery data: expected a list")
    # kubectl rewraps single-type output as a v1 List whose items carry their own
    # kind. Typed lists (PodList and friends) elide it; restore it from the list
    # kind so a hand-run or piped manifest parses the same way.
    list_kind = payload.get("kind", "")
    item_kind = list_kind[:-4] if list_kind.endswith("List") and list_kind != "List" else ""
    item_api_version = payload.get("apiVersion", "") if item_kind else ""
    objects = []
    for item in payload["items"] or []:
        if item_kind and not item.get("kind"):
            item = {
                **item,
                "kind": item_kind,
                "apiVersion": item.get("apiVersion") or item_api_version,
            }
        objects.append(item)
    return objects


def list_objects(
    config: RuntimeConfig,
    resource_types: tuple[str, ...],
    selector: str,
) -> list[dict]:
    """List each resource type concurrently and merge the results.

    ``kubectl get a,b,c`` issues one LIST per type sequentially, so a namespace
    holding a dozen managed types costs a dozen serial round trips. One process
    per type turns that into a single round trip of wall-clock time.
    """

    resource_types = tuple(resource_types)
    if not resource_types:
        return []
    timeout = kubectl_timeout()
    retries = kubectl_retries()

    def fetch(resource_type: str) -> list[dict]:
        raw = capture(
            [
                *config.kubectl(),
                "get",
                resource_type,
                "-l",
                selector,
                "-o",
                "json",
                *_request_timeout_flag(timeout),
            ],
            timeout=timeout,
            retries=retries,
            # A type this cluster does not serve means "no such objects"; any
            # other failure must surface rather than read as an empty namespace.
            tolerate=MISSING_RESOURCE_ERRORS,
        )
        if not raw.strip():
            return []
        try:
            return _objects_from_list(raw)
        except json.JSONDecodeError as exc:
            raise WorkflowError(
                f"kubectl returned invalid discovery data for {resource_type}: {exc}"
            ) from exc

    workers = min(len(resource_types), MAX_DISCOVERY_WORKERS)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="manifesto-get"
    ) as pool:
        return [obj for group in pool.map(fetch, resource_types) for obj in group]


def discover_live_resources(
    config: RuntimeConfig,
    *,
    instance_id: str | None = None,
    resource_types: tuple[str, ...] | None = None,
) -> list[LiveResource]:
    if resource_types is None:
        resource_types = DISCOVERY_RESOURCE_TYPES
    selector = MANIFESTO_SELECTOR
    if instance_id:
        selector += f",app.kubernetes.io/instance={instance_id}"
    objects = list_objects(config, resource_types, selector)
    return [LiveResource.from_object(obj) for obj in objects]


def group_servers(resources: list[LiveResource]) -> list[ServerRecord]:
    grouped: dict[str, list[LiveResource]] = {}
    for resource in resources:
        if resource.instance_id:
            grouped.setdefault(resource.instance_id, []).append(resource)
    return [
        ServerRecord(instance_id=instance_id, resources=tuple(grouped[instance_id]))
        for instance_id in sorted(grouped)
    ]


def servers(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    records = group_servers(discover_live_resources(config, instance_id=args.instance))
    if args.instance:
        records = [record for record in records if record.instance_id == args.instance]

    if args.output == "name":
        for record in records:
            print(record.instance_id)
        return 0
    if args.output == "json":
        print(json.dumps([record.as_dict() for record in records], indent=2))
        return 0

    print(format_server_table(records))
    if args.instance and records:
        print(f"\n{format_server_resources(records[0])}")
    return 0


def format_server_table(records: list[ServerRecord], *, numbered: bool = False) -> str:
    headers = ("#", "INSTANCE", "STATE", "MODEL", "ROLES", "PODS", "AGE") if numbered else (
        "INSTANCE",
        "STATE",
        "MODEL",
        "ROLES",
        "PODS",
        "AGE",
    )
    rows = [
        ((str(index),) if numbered else ())
        + (record.instance_id, record.state, record.model, record.roles, record.pod_readiness, record.age)
        for index, record in enumerate(records, start=1)
    ]
    widths = [
        max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, header in enumerate(headers)
    ]
    lines = ["  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)).rstrip()]
    lines.extend(
        "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)).rstrip() for row in rows
    )
    return "\n".join(lines)


def format_server_resources(record: ServerRecord) -> str:
    lines = ["Resources:"]
    lines.extend(
        f"  {resource.kind:<20} {resource.name}"
        for resource in sorted(record.resources, key=lambda item: (item.kind, item.name))
    )
    return "\n".join(lines)


def stop(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    if args.spec and args.instance:
        raise WorkflowError("Pass either SPEC or --instance, not both.", code=2)

    resources: list[LiveResource] | None = None
    if args.spec:
        spec = load_spec(resolve_model(args.spec))
        instance_id = Instance(user=config.user, release=spec.release).instance_id
    elif args.instance:
        instance_id = args.instance
    else:
        if not sys.stdin.isatty():
            raise WorkflowError(
                "No server target provided. Pass SPEC or --instance ID; interactive selection requires a TTY.",
                code=2,
            )
        print(f"Looking for Manifesto servers in namespace {config.namespace}...", file=sys.stderr)
        records = group_servers(discover_live_resources(config))
        if not records:
            print(f"No running Manifesto servers found in namespace {config.namespace}.")
            return 0
        selected = pick_server(records, config)
        if selected is None:
            print("Teardown canceled.")
            return 130
        instance_id = selected.instance_id
        resources = list(selected.resources)

    if resources is None:
        print(f"Looking up {instance_id} in namespace {config.namespace}...", file=sys.stderr)
        resources = discover_live_resources(config, instance_id=instance_id)
    return delete_instance(config, instance_id, resources, now=args.now)


def pick_server(records: list[ServerRecord], config: RuntimeConfig) -> ServerRecord | None:
    if shutil.which("fzf"):
        lines = [
            "\t".join(
                (record.instance_id, record.state, record.model, record.roles, record.pod_readiness, record.age)
            )
            for record in records
        ]
        # Discovery already returned every resource, so previews read from disk
        # rather than re-querying the cluster on each keystroke.
        with tempfile.TemporaryDirectory(prefix="manifesto-preview-") as preview_dir:
            for record in records:
                (Path(preview_dir) / record.instance_id).write_text(
                    format_server_resources(record)
                )
            proc = subprocess.run(
                [
                    "fzf",
                    "--delimiter=\\t",
                    "--with-nth=1..",
                    "--header=INSTANCE  STATE  MODEL  ROLES  PODS  AGE",
                    f"--preview=cat {shlex.quote(preview_dir)}/{{1}}",
                    "--preview-window=right,55%",
                    "--prompt=Stop server> ",
                ],
                input="\n".join(lines) + "\n",
                text=True,
                stdout=subprocess.PIPE,
            )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        instance_id = proc.stdout.split("\t", 1)[0].strip()
        return next((record for record in records if record.instance_id == instance_id), None)

    print(format_server_table(records, numbered=True))
    while True:
        choice = input(f"Select server to stop [1-{len(records)}] (q to cancel): ").strip()
        if choice.casefold() in {"q", "quit"}:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(records):
            return records[int(choice) - 1]
        print("Invalid selection.")


def delete_instance(
    config: RuntimeConfig,
    instance_id: str,
    resources: list[LiveResource],
    *,
    now: bool,
) -> int:
    if not resources:
        print(f"Manifesto server {instance_id} is already absent from namespace {config.namespace}.")
        return 0

    top_level = sorted(
        (resource.kubectl_ref for resource in resources if resource.kind != "Pod")
    )
    print(
        f"Stopping {instance_id} in namespace {config.namespace} "
        f"({len(top_level)} resources, {sum(resource.kind == 'Pod' for resource in resources)} pods)..."
    )
    cmd = [*config.kubectl(), "delete", *top_level, "--ignore-not-found=true"]
    if now:
        cmd.extend(["--grace-period=0", "--force"])
    rc = run(cmd) if top_level else 0

    if rc:
        report_incomplete_teardown(config, instance_id)
        return rc

    # Controllers recreate pods, so this round only needs the pods themselves.
    print("Checking for orphaned pods...", file=sys.stderr)
    pods = sorted(
        resource.kubectl_ref
        for resource in discover_live_resources(
            config, instance_id=instance_id, resource_types=(POD_RESOURCE_TYPE,)
        )
    )
    if pods:
        pod_cmd = [*config.kubectl(), "delete", *pods, "--ignore-not-found=true"]
        if now:
            pod_cmd.extend(["--grace-period=0", "--force"])
        rc = max(rc, run(pod_cmd))

    print("Verifying teardown...", file=sys.stderr)
    if report_incomplete_teardown(config, instance_id):
        return rc or 1
    if rc:
        return rc
    print(f"Stopped {instance_id}.")
    return 0


def report_incomplete_teardown(config: RuntimeConfig, instance_id: str) -> bool:
    """Print any resources still present for the instance; return whether any were.

    This is the last word on whether a teardown finished, so it re-reads the full
    managed set rather than only the types seen before the delete. The lists run
    concurrently, so breadth here costs one round trip, not one per type.
    """

    leftovers = discover_live_resources(config, instance_id=instance_id)
    if not leftovers:
        return False
    print(f"Teardown incomplete for {instance_id}; resources still present:", file=sys.stderr)
    for resource in sorted(leftovers, key=lambda item: (item.kind, item.name)):
        print(f"  {resource.kind}/{resource.name}", file=sys.stderr)
    return True


def diff_file(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    return run([*config.kubectl(), "diff", "-f", str(config.render_out)])


def apply_file(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    rc = sync_hf_secret(config)
    if rc:
        return rc
    return run([*config.kubectl(), "apply", "-f", str(config.render_out)])


def delete_file(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    cmd = [*config.kubectl(), "delete", "-f", str(config.render_out), "--ignore-not-found=true"]
    if args.now:
        cmd.extend(["--grace-period=0", "--force"])
    return run(cmd)


def ready(
    args,
    *,
    config: RuntimeConfig | None = None,
    cluster: Cluster | None = None,
) -> int:
    config = config or RuntimeConfig.from_args(args, require_cluster=False)
    spec = load_spec(resolve_model(args.spec), cluster)
    instance = Instance(user=config.user, release=spec.release)
    epp = instance.name("infpool-epp")
    routing_enabled = spec.routing.kind != RoutingKind.DISABLED

    gateway = ""
    if routing_enabled:
        cluster = cluster or load_runtime_cluster(config, args)
        gateway_name = instance.name("gateway", max_length=63 - len(cluster.gateway.class_name) - 1)
        gateway = f"{gateway_name}-{cluster.gateway.class_name}"

    print("Waiting for model pods and endpoint picker...")
    waits = [
        [
            *config.kubectl(),
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            ",".join(f"{key}={value}" for key, value in instance.pod_selector(role.name).items()),
            "--timeout=1200s",
        ]
        for role in spec.roles
    ]
    if routing_enabled:
        waits.append([*config.kubectl(), "wait", "--for=condition=Available", f"deploy/{epp}", "--timeout=120s"])
    procs = [subprocess.Popen(cmd) for cmd in waits]
    rc = max(proc.wait() for proc in procs)
    if rc:
        return rc
    if not routing_enabled:
        print("Ready.")
        return 0

    print("Checking gateway...")
    url = f"http://{gateway}:80/v1/models"
    deadline = time.monotonic() + args.gateway_timeout
    while time.monotonic() < deadline:
        out = capture(["curl", "-sf", "--max-time", "5", url], check=False)
        if '"id"' in out:
            print("Ready.")
            return 0
        out = capture(
            [*config.kubectl(), "exec", f"deploy/{epp}", "--", "curl", "-sf", "--max-time", "5", url],
            check=False,
        )
        if '"id"' in out:
            print("Ready.")
            return 0
        time.sleep(2)
    print(f"Gateway did not become ready within {args.gateway_timeout}s.", file=sys.stderr)
    return 1


def kubectl_timeout() -> float | None:
    """Per-attempt wall-clock budget for a cluster read, or None to wait forever."""

    if "timeout" in _kubectl_limits:
        return _kubectl_limits["timeout"]
    raw = os.environ.get("MANIFESTO_KUBECTL_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_KUBECTL_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_KUBECTL_TIMEOUT
    return value if value > 0 else None


def kubectl_retries() -> int:
    if "retries" in _kubectl_limits:
        return int(_kubectl_limits["retries"] or 0)
    raw = os.environ.get("MANIFESTO_KUBECTL_RETRIES", "").strip()
    if not raw:
        return DEFAULT_KUBECTL_RETRIES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_KUBECTL_RETRIES


@contextlib.contextmanager
def kubectl_limits(*, timeout: float | None | object = _UNSET, retries: int | object = _UNSET):
    """Temporarily override the read timeout and retry budget."""

    previous = dict(_kubectl_limits)
    if timeout is not _UNSET:
        _kubectl_limits["timeout"] = timeout
    if retries is not _UNSET:
        _kubectl_limits["retries"] = retries
    try:
        yield
    finally:
        _kubectl_limits.clear()
        _kubectl_limits.update(previous)


def _request_timeout_flag(timeout: float | None) -> list[str]:
    """Bound the HTTP request just inside the process deadline.

    The headroom lets kubectl report its own diagnostic instead of being killed
    mid-request and surfacing only a generic timeout.
    """

    if not timeout:
        return []
    return [f"--request-timeout={max(1, int(timeout * 0.9))}s"]


def _trace(message: str) -> None:
    if os.environ.get("MANIFESTO_TRACE", "").strip().casefold() not in TRACE_OFF_VALUES:
        print(f"[manifesto] {message}", file=sys.stderr, flush=True)


def _matches(stderr: str, markers: tuple[str, ...]) -> bool:
    lowered = stderr.casefold()
    return any(marker in lowered for marker in markers)


def run(cmd: list[str], *, input_text: str | None = None) -> int:
    started = time.monotonic()
    rc = subprocess.run(cmd, input=input_text, text=True).returncode
    _trace(f"{shlex.join(cmd)} -> rc={rc} in {time.monotonic() - started:.1f}s")
    return rc


def capture(
    cmd: list[str],
    *,
    check: bool = True,
    timeout: float | None | object = _UNSET,
    retries: int = 0,
    tolerate: tuple[str, ...] = (),
) -> str:
    """Run a command and return stdout, retrying transient cluster failures.

    ``tolerate`` names error substrings that mean "nothing to report" rather than
    "the read failed" — an unserved resource type, for instance.
    """

    if timeout is _UNSET:
        timeout = kubectl_timeout()
    attempt = 0
    while True:
        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except FileNotFoundError:
            if check:
                raise WorkflowError(f"command not found: {cmd[0]}")
            return ""
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            _trace(f"{shlex.join(cmd)} -> timeout after {elapsed:.1f}s")
            if attempt < retries:
                attempt += 1
                time.sleep(min(2.0 ** attempt, 8.0))
                continue
            if check:
                raise WorkflowError(
                    f"timed out after {elapsed:.0f}s: {shlex.join(cmd)}\n"
                    "Raise MANIFESTO_KUBECTL_TIMEOUT if this cluster is simply slow."
                )
            return ""

        _trace(
            f"{shlex.join(cmd)} -> rc={proc.returncode} in {time.monotonic() - started:.1f}s"
        )
        if proc.returncode == 0:
            return proc.stdout
        if tolerate and _matches(proc.stderr, tolerate):
            # Swallowing a read must never be invisible on a teardown path.
            _trace(f"tolerated: {proc.stderr.strip()}")
            return ""
        if attempt < retries and _matches(proc.stderr, TRANSIENT_KUBECTL_ERRORS):
            attempt += 1
            time.sleep(min(2.0 ** attempt, 8.0))
            continue
        if check:
            raise WorkflowError(
                proc.stderr.strip() or f"command failed ({proc.returncode}): {shlex.join(cmd)}"
            )
        return proc.stdout
