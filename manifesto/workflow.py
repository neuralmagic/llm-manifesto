"""Higher-level workflow helpers for the manifesto CLI."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .cluster import Cluster, load_cluster
from .catalog import ROOT, catalog_entries, config_home, resolve_catalog_path
from .instance import Instance
from .overrides import load_routing_profile
from .render import render, render_to_yaml
from .render.bootstrap import render_bootstrap
from .render.devpod import render_dev_pod
from .spec import EppSpec, RoutingKind, load_spec


HF_SECRET_NAME = "hf-secret"
HF_SECRET_KEY = "HF_TOKEN"
VLLM_DEV_REMOTE = "https://github.com/vllm-project/vllm.git"
VLLM_DEV_BRANCH = "main"
VLLM_BUILD_JOBS = 16
DEV_WORKTREE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

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

    @classmethod
    def from_args(cls, args, *, require_cluster: bool = True) -> "RuntimeConfig":
        load_dotenv()
        user = resolve_user(getattr(args, "user", None))
        namespace = resolve_namespace(getattr(args, "namespace", None))
        cluster_path = resolve_cluster(getattr(args, "cluster", None)) if require_cluster else getattr(args, "cluster", None)
        render_out = Path(
            getattr(args, "output", None)
            or os.environ.get("MANIFESTO_RENDER_OUT", f"/tmp/{user}-manifesto.yaml")
        )
        return cls(user=user, namespace=namespace, cluster_path=cluster_path, render_out=render_out)

    def kubectl(self) -> list[str]:
        return ["kubectl", "-n", self.namespace]


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
    def kubectl_ref(self) -> str:
        if self.kind == "Pod":
            resource_type = POD_RESOURCE_TYPE
        else:
            resource_type = MANAGED_RESOURCE_TYPES.get((self.api_version, self.kind))
        if not resource_type:
            raise WorkflowError(f"unsupported managed resource: {self.api_version} {self.kind}")
        return f"{resource_type}/{self.name}"


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


def resolve_namespace(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("MANIFESTO_NAMESPACE"):
        return os.environ["MANIFESTO_NAMESPACE"]
    namespace = capture(["kubectl", "config", "view", "--minify", "-o", "jsonpath={..namespace}"], check=False)
    return namespace.strip() or "default"


def resolve_cluster(explicit: str | None = None) -> str:
    if explicit:
        return resolve_catalog_path(explicit, "clusters")
    if os.environ.get("MANIFESTO_CLUSTER"):
        return resolve_catalog_path(os.environ["MANIFESTO_CLUSTER"], "clusters")
    mapping = os.environ.get("MANIFESTO_CLUSTER_MAP", "")
    context = capture(["kubectl", "config", "current-context"], check=False).strip()
    kube_cluster = capture(
        ["kubectl", "config", "view", "--minify", "-o", "jsonpath={.clusters[0].name}"],
        check=False,
    ).strip()
    if mapping:
        for entry in mapping.split(","):
            key, sep, value = entry.partition("=")
            if sep and key.strip() in {context, kube_cluster}:
                return resolve_catalog_path(value.strip(), "clusters")
    for name in (context, kube_cluster):
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
    cluster = load_cluster_with_overrides(config.cluster_path, args)
    worktree_name = getattr(args, "dev_worktree", None)
    if worktree_name:
        if getattr(args, "dev_source", None) or getattr(args, "dev_venv", None):
            raise WorkflowError(
                "--dev-worktree cannot be combined with --dev-source or --dev-venv",
                code=2,
            )
        worktree = cluster.dev_worktrees(
            user=Instance(user=config.user, release="dev").user_slug,
            release="",
        ) + f"/{validate_dev_worktree_name(worktree_name)}"
        cluster = cluster.with_path_overrides(
            dev_source=worktree,
            dev_venv=f"{worktree}/.venv",
        )
    return cluster


def load_cluster_with_overrides(cluster_path: str, args):
    return load_cluster(cluster_path).with_path_overrides(
        user_root=getattr(args, "user_root", None),
        log_root=getattr(args, "log_root", None),
        cache_root=getattr(args, "cache_root", None),
        dev_venv=getattr(args, "dev_venv", None),
        dev_source=getattr(args, "dev_source", None),
    )


def apply_runtime_overrides(spec, args, config: RuntimeConfig) -> None:
    spec.namespace = config.namespace
    if getattr(args, "accelerator", None):
        spec.accelerator = args.accelerator
    if getattr(args, "dev", False) or getattr(args, "dev_worktree", None):
        spec.runtime.dev = True
    if getattr(args, "dev_venv", None):
        spec.runtime.dev_venv = args.dev_venv
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
    if getattr(args, "dev", False):
        command.append("--dev")
    if getattr(args, "dev_worktree", None):
        command.extend(["--dev-worktree", args.dev_worktree])
    if getattr(args, "accelerator", None):
        command.extend(["--gpu", args.accelerator])
    routing_profile = getattr(args, "routing_profile", None) or os.environ.get(
        "MANIFESTO_ROUTING_PROFILE"
    )
    if routing_profile:
        command.extend(["--routing-profile", routing_profile])
    for name in ("user_root", "log_root", "cache_root", "dev_venv", "dev_source"):
        value = getattr(args, name, None)
        if value:
            command.extend([f"--{name.replace('_', '-')}", value])
    for hook in getattr(args, "pre_launch", None) or []:
        command.extend(["--pre-launch", hook])
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
        rc = sync_hf_secret(config)
        if rc:
            return rc
    return run([*config.kubectl(), "apply", "-f", "-"], input_text=manifest)


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


def sync_hf_secret(config: RuntimeConfig) -> int:
    token = os.environ.get(HF_SECRET_KEY, "").strip()
    if not token:
        raise WorkflowError(
            "HF_TOKEN is not configured. Set a fine-grained read token in "
            f"{config_home() / '.env'} or the environment before deploying.",
            code=2,
        )
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


def _dev_identity(config: RuntimeConfig) -> tuple[Instance, str]:
    instance = Instance(user=config.user, release="dev")
    return instance, instance.user_scoped_name("vllm-dev")


def _dev_paths(
    config: RuntimeConfig,
    args,
    *,
    cluster: Cluster | None = None,
) -> tuple[Cluster, str, str, str]:
    cluster = cluster or load_runtime_cluster(config, args)
    instance, _ = _dev_identity(config)
    user = instance.user_slug
    return (
        cluster,
        cluster.user_root(user=user, release=""),
        cluster.dev_source(user=user, release=""),
        cluster.dev_venv(user=user, release=""),
    )


def validate_dev_worktree_name(name: str) -> str:
    if not DEV_WORKTREE_NAME_RE.fullmatch(name) or name in {".", ".."}:
        raise WorkflowError(
            "worktree name must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-' (128 characters maximum)",
            code=2,
        )
    return name


def _dev_worktree_storage(
    config: RuntimeConfig,
    args,
) -> tuple[str, str, str]:
    cluster = load_runtime_cluster(config, args)
    instance, _ = _dev_identity(config)
    user = instance.user_slug
    root = cluster.dev_worktrees(user=user, release="")
    cache = cluster.dev_envs_cache(user=user, release="")
    source = cluster.dev_source(user=user, release="")
    return source, root, cache


def _dev_worktree_paths(
    config: RuntimeConfig,
    args,
) -> tuple[str, str, str, str]:
    source, root, cache = _dev_worktree_storage(config, args)
    name = validate_dev_worktree_name(args.name)
    return source, root, cache, f"{root}/{name}"


def _dev_worktree_env(root: str, cache: str) -> list[str]:
    return [f"VE_ENVS_ROOT={root}", f"VE_CACHE_DIR={cache}"]


def _openshift_namespace_uid(config: RuntimeConfig) -> int:
    uid_range = capture(
        [
            *config.kubectl(),
            "get",
            "namespace",
            config.namespace,
            "-o",
            "jsonpath={.metadata.annotations.openshift\\.io/sa\\.scc\\.uid-range}",
        ]
    ).strip()
    value = uid_range.partition("/")[0]
    if not value.isdigit():
        raise WorkflowError(
            f"namespace {config.namespace} has no valid OpenShift SCC UID range",
            code=2,
        )
    return int(value)


def dev_start(
    args,
    *,
    config: RuntimeConfig | None = None,
    cluster: Cluster | None = None,
) -> int:
    """Create the persistent dev pod and initialize its checkout and venv."""
    config = config or RuntimeConfig.from_args(args)
    cluster, _, _, _ = _dev_paths(config, args, cluster=cluster)
    accelerator = cluster.accelerators.get(args.accelerator)
    _, pod_name = _dev_identity(config)
    run_as_user = None
    if cluster.platform == "openshift" and not (
        cluster.pod_defaults.container_security_context or {}
    ).get("runAsUser"):
        run_as_user = _openshift_namespace_uid(config)

    if not getattr(args, "skip_hf_secret_sync", False):
        rc = sync_hf_secret(config)
        if rc:
            return rc
    rc = run(
        [*config.kubectl(), "apply", "-f", "-"],
        input_text=render_to_yaml(
            [
                render_dev_pod(
                    cluster,
                    config.user,
                    accelerator,
                    image=getattr(args, "image", None),
                    cpu=getattr(args, "dev_cpu", None),
                    memory=getattr(args, "dev_memory", None),
                    gpus=getattr(args, "dev_gpus", None),
                    run_as_user=run_as_user,
                )
            ]
        ),
    )
    if rc:
        return rc
    rc = run(
        [
            *config.kubectl(),
            "wait",
            "--for=condition=Ready",
            f"pod/{pod_name}",
            "--timeout=300s",
        ]
    )
    if rc:
        return rc
    return dev_init(args, config=config, cluster=cluster)


def dev_init(
    args,
    *,
    config: RuntimeConfig | None = None,
    cluster: Cluster | None = None,
) -> int:
    """Idempotently initialize the per-user vLLM checkout and virtualenv."""
    config = config or RuntimeConfig.from_args(args)
    _, user_root, source, venv = _dev_paths(config, args, cluster=cluster)
    _, pod_name = _dev_identity(config)
    remote = getattr(args, "remote", VLLM_DEV_REMOTE)
    branch = getattr(args, "branch", VLLM_DEV_BRANCH)
    script = r'''set -euo pipefail
mkdir -p "$MANIFESTO_DEV_USER_ROOT"

if [ -d "$MANIFESTO_DEV_SOURCE/.git" ]; then
  echo "Using existing vLLM checkout at $MANIFESTO_DEV_SOURCE"
elif [ -e "$MANIFESTO_DEV_SOURCE" ] && [ ! -d "$MANIFESTO_DEV_SOURCE" ]; then
  echo "Error: $MANIFESTO_DEV_SOURCE exists and is not a directory" >&2
  exit 1
elif [ -n "$(find "$MANIFESTO_DEV_SOURCE" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "Error: $MANIFESTO_DEV_SOURCE exists, is not empty, and is not a git checkout" >&2
  exit 1
else
  echo "Cloning $MANIFESTO_VLLM_REMOTE ($MANIFESTO_VLLM_BRANCH) to $MANIFESTO_DEV_SOURCE"
  git clone --depth 1 --branch "$MANIFESTO_VLLM_BRANCH" --single-branch \
    "$MANIFESTO_VLLM_REMOTE" "$MANIFESTO_DEV_SOURCE"
fi

if [ -f "$MANIFESTO_DEV_VENV/bin/activate" ]; then
  echo "Using existing vLLM venv at $MANIFESTO_DEV_VENV"
elif [ -e "$MANIFESTO_DEV_VENV" ]; then
  echo "Error: $MANIFESTO_DEV_VENV exists but is not a complete virtualenv" >&2
  exit 1
elif [ -f /opt/vllm/bin/activate ]; then
  echo "Seeding vLLM venv from /opt/vllm at $MANIFESTO_DEV_VENV"
  cp -a /opt/vllm "$MANIFESTO_DEV_VENV"
else
  echo "Creating vLLM venv at $MANIFESTO_DEV_VENV"
  uv venv --python /usr/bin/python3.12 "$MANIFESTO_DEV_VENV"
fi
'''
    return run(
        [
            *config.kubectl(),
            "exec",
            "-i",
            pod_name,
            "--",
            "env",
            f"MANIFESTO_DEV_VENV={venv}",
            f"MANIFESTO_DEV_SOURCE={source}",
            f"MANIFESTO_DEV_USER_ROOT={user_root}",
            f"MANIFESTO_VLLM_REMOTE={remote}",
            f"MANIFESTO_VLLM_BRANCH={branch}",
            "bash",
            "-s",
        ],
        input_text=script,
    )


def dev_shell(args) -> int:
    config = RuntimeConfig.from_args(args, require_cluster=False)
    _, pod_name = _dev_identity(config)
    return run([*config.kubectl(), "exec", "-it", pod_name, "--", "/bin/zsh"])


def dev_worktree_create(args) -> int:
    config = RuntimeConfig.from_args(args)
    source, root, cache, _ = _dev_worktree_paths(config, args)
    _, pod_name = _dev_identity(config)
    return run(
        [
            *config.kubectl(),
            "exec",
            "-i",
            pod_name,
            "--",
            "env",
            *_dev_worktree_env(root, cache),
            "bash",
            "-c",
            (
                'mkdir -p "$4" "$5" "$(dirname "$3")"; '
                'if [ ! -d "$3/.git" ]; then '
                'if [ -e "$3" ] && [ -n "$(find "$3" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; '
                'then echo "Error: $3 is not an empty directory or git checkout" >&2; exit 1; fi; '
                'git clone --depth 1 --branch main --single-branch "$6" "$3"; fi; '
                'command -v ve >/dev/null 2>&1 || { echo "Error: ve is not installed in the dev image" >&2; exit 1; }; '
                'if git -C "$3" rev-parse --verify --quiet "$1^{commit}" >/dev/null; then '
                'resolved_ref="$1"; else git -C "$3" fetch origin "$1" && resolved_ref=FETCH_HEAD; fi; '
                'exec ve new "$resolved_ref" --name "$2" --repo "$3"'
            ),
            "bash",
            args.ref,
            args.name,
            source,
            root,
            cache,
            args.remote,
        ]
    )


def dev_worktree_list(args) -> int:
    config = RuntimeConfig.from_args(args)
    _, root, cache = _dev_worktree_storage(config, args)
    _, pod_name = _dev_identity(config)
    return run(
        [
            *config.kubectl(),
            "exec",
            pod_name,
            "--",
            "env",
            *_dev_worktree_env(root, cache),
            "ve",
            "list",
        ]
    )


def dev_worktree_sync(args) -> int:
    config = RuntimeConfig.from_args(args)
    _, root, cache, worktree = _dev_worktree_paths(config, args)
    _, pod_name = _dev_identity(config)
    return run(
        [
            *config.kubectl(),
            "exec",
            "-i",
            pod_name,
            "--",
            "env",
            *_dev_worktree_env(root, cache),
            "bash",
            "-c",
            'cd "$1" && exec ve sync',
            "bash",
            worktree,
        ]
    )


def dev_worktree_shell(args) -> int:
    config = RuntimeConfig.from_args(args)
    _, root, cache, worktree = _dev_worktree_paths(config, args)
    _, pod_name = _dev_identity(config)
    return run(
        [
            *config.kubectl(),
            "exec",
            "-it",
            pod_name,
            "--",
            "env",
            *_dev_worktree_env(root, cache),
            "bash",
            "-c",
            'cd "$1" && source .venv/bin/activate && exec /bin/zsh',
            "bash",
            worktree,
        ]
    )


def dev_worktree_remove(args) -> int:
    if not args.force:
        raise WorkflowError(
            "removing a worktree can discard uncommitted changes; rerun with --force",
            code=2,
        )
    config = RuntimeConfig.from_args(args)
    _, root, cache, _ = _dev_worktree_paths(config, args)
    _, pod_name = _dev_identity(config)
    return run(
        [
            *config.kubectl(),
            "exec",
            "-i",
            pod_name,
            "--",
            "env",
            *_dev_worktree_env(root, cache),
            "ve",
            "rm",
            args.name,
        ]
    )


def dev_build(args) -> int:
    config = RuntimeConfig.from_args(args)
    rc = dev_init(args, config=config)
    if rc:
        return rc
    _, user_root, source, venv = _dev_paths(config, args)
    _, pod_name = _dev_identity(config)
    script = r'''set -euo pipefail
VENV_PYTHON="$MANIFESTO_DEV_VENV/bin/python"
if [ "$(readlink "$VENV_PYTHON")" != "/usr/bin/python3.12" ]; then
  echo "Repointing venv python: $(readlink "$VENV_PYTHON") -> /usr/bin/python3.12"
  ln -sf /usr/bin/python3.12 "$VENV_PYTHON"
fi

source "$MANIFESTO_DEV_VENV/bin/activate"
# Toolchain images may constrain the PyTorch version used to build their baked
# vLLM checkout. A mutable dev checkout must resolve against its own revision.
unset UV_CONSTRAINT
cd "$MANIFESTO_DEV_SOURCE"
PREVIOUS_HEAD=$(git rev-parse HEAD 2>/dev/null || true)
git remote set-url origin "$MANIFESTO_VLLM_REMOTE"
git fetch origin
git checkout "$MANIFESTO_VLLM_BRANCH"
git reset --hard "origin/$MANIFESTO_VLLM_BRANCH"
CURRENT_HEAD=$(git rev-parse HEAD)

# An editable install leaves generated package metadata in the shared checkout.
# Move it aside before resolving a different revision so stale dependency pins
# (notably the exact PyTorch version) cannot be combined with the new tree.
if [ -d vllm.egg-info ]; then
  mv vllm.egg-info "vllm.egg-info.old.$(date +%s)"
  echo "Flushed stale vllm.egg-info"
fi

if [ "$PREVIOUS_HEAD" != "$CURRENT_HEAD" ] && [ -d .deps ]; then
  mv .deps ".deps.old.$(date +%s)"
  echo "Flushed stale native dependency sources"
fi

if [ -d "$MANIFESTO_DEV_USER_ROOT/jit-cache" ]; then
  mv "$MANIFESTO_DEV_USER_ROOT/jit-cache" \
    "$MANIFESTO_DEV_USER_ROOT/jit-cache.old.$(date +%s)"
  echo "Flushed jit-cache"
fi

nohup bash -c '
  uv pip uninstall vllm || true
  uv pip install -r requirements/build/cuda.txt
  MAX_JOBS="$MANIFESTO_VLLM_BUILD_JOBS" \
    uv pip install --no-build-isolation -e .
' \
  > "$MANIFESTO_DEV_USER_ROOT/build.log" 2>&1 &

echo "Build started ($MANIFESTO_VLLM_REMOTE $MANIFESTO_VLLM_BRANCH, jobs=$MANIFESTO_VLLM_BUILD_JOBS)"
echo "Follow with: manifesto dev build-log"
'''
    return run(
        [
            *config.kubectl(),
            "exec",
            "-i",
            pod_name,
            "--",
            "env",
            f"MANIFESTO_DEV_VENV={venv}",
            f"MANIFESTO_DEV_SOURCE={source}",
            f"MANIFESTO_DEV_USER_ROOT={user_root}",
            f"MANIFESTO_VLLM_REMOTE={args.remote}",
            f"MANIFESTO_VLLM_BRANCH={args.branch}",
            f"MANIFESTO_VLLM_BUILD_JOBS={args.jobs}",
            "bash",
            "-s",
        ],
        input_text=script,
    )


def dev_build_log(args) -> int:
    config = RuntimeConfig.from_args(args)
    _, user_root, _, _ = _dev_paths(config, args)
    _, pod_name = _dev_identity(config)
    return run([*config.kubectl(), "exec", pod_name, "--", "tail", "-f", f"{user_root}/build.log"])


def dev_stop(args, *, config: RuntimeConfig | None = None) -> int:
    config = config or RuntimeConfig.from_args(args, require_cluster=False)
    _, pod_name = _dev_identity(config)
    return run([*config.kubectl(), "delete", f"pod/{pod_name}", "--ignore-not-found=true"])


def discover_live_resources(config: RuntimeConfig, *, instance_id: str | None = None) -> list[LiveResource]:
    available = set(
        capture(
            ["kubectl", "api-resources", "--namespaced=true", "--verbs=list,delete", "-o", "name"]
        ).splitlines()
    )
    resource_types = sorted(set(MANAGED_RESOURCE_TYPES.values()) & available)
    if POD_RESOURCE_TYPE in available:
        resource_types.append(POD_RESOURCE_TYPE)
    if not resource_types:
        raise WorkflowError("No Manifesto-managed Kubernetes resource types are available in this cluster.")

    selector = MANIFESTO_SELECTOR
    if instance_id:
        selector += f",app.kubernetes.io/instance={instance_id}"
    raw = capture([*config.kubectl(), "get", ",".join(resource_types), "-l", selector, "-o", "json"])
    try:
        objects = json.loads(raw).get("items", [])
    except (AttributeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"kubectl returned invalid discovery data: {exc}") from exc
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

    print_server_table(records)
    if args.instance and records:
        print("\nResources:")
        for resource in sorted(records[0].resources, key=lambda item: (item.kind, item.name)):
            print(f"  {resource.kind:<20} {resource.name}")
    return 0


def print_server_table(records: list[ServerRecord], *, numbered: bool = False) -> None:
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
    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)).rstrip())
    for row in rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)).rstrip())


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

    resources = resources if resources is not None else discover_live_resources(config, instance_id=instance_id)
    return delete_instance(config, instance_id, resources, now=args.now)


def pick_server(records: list[ServerRecord], config: RuntimeConfig) -> ServerRecord | None:
    if shutil.which("fzf"):
        lines = [
            "\t".join(
                (record.instance_id, record.state, record.model, record.roles, record.pod_readiness, record.age)
            )
            for record in records
        ]
        preview = shlex.join(
            [
                sys.executable,
                "-m",
                "manifesto.cli",
                "servers",
                "--namespace",
                config.namespace,
                "--instance",
                "{1}",
            ]
        )
        proc = subprocess.run(
            [
                "fzf",
                "--delimiter=\\t",
                "--with-nth=1..",
                "--header=INSTANCE  STATE  MODEL  ROLES  PODS  AGE",
                f"--preview={preview}",
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

    print_server_table(records, numbered=True)
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

    remaining = discover_live_resources(config, instance_id=instance_id)
    if rc:
        print(f"Teardown incomplete for {instance_id}; resources still present:", file=sys.stderr)
        for resource in sorted(remaining, key=lambda item: (item.kind, item.name)):
            print(f"  {resource.kind}/{resource.name}", file=sys.stderr)
        return rc

    pods = sorted(resource.kubectl_ref for resource in remaining if resource.kind == "Pod")
    if pods:
        pod_cmd = [*config.kubectl(), "delete", *pods, "--ignore-not-found=true"]
        if now:
            pod_cmd.extend(["--grace-period=0", "--force"])
        rc = max(rc, run(pod_cmd))

    leftovers = discover_live_resources(config, instance_id=instance_id)
    if leftovers:
        print(f"Teardown incomplete for {instance_id}; resources still present:", file=sys.stderr)
        for resource in sorted(leftovers, key=lambda item: (item.kind, item.name)):
            print(f"  {resource.kind}/{resource.name}", file=sys.stderr)
        return rc or 1
    if rc:
        return rc
    print(f"Stopped {instance_id}.")
    return 0


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


def run(cmd: list[str], *, input_text: str | None = None) -> int:
    return subprocess.run(cmd, input=input_text, text=True).returncode


def capture(cmd: list[str], *, check: bool = True) -> str:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        if check:
            raise WorkflowError(f"command not found: {cmd[0]}")
        return ""
    if check and proc.returncode != 0:
        raise WorkflowError(proc.stderr.strip() or f"command failed ({proc.returncode}): {shlex.join(cmd)}")
    return proc.stdout
