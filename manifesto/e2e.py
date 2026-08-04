"""Fresh-namespace, in-cluster integration-test orchestration."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import uuid
from typing import Any

from . import workflow
from .cluster import Cluster
from .images import DEFAULT_IMAGES
from .instance import Instance
from .render import render, render_to_yaml
from .resolve import resolve_role
from .spec import DeploymentSpec, RoutingKind, load_spec


E2E_IMAGE = DEFAULT_IMAGES.get("test.e2e")
NAMESPACE_DELETE_TIMEOUT = 120

_PROBE_SCRIPT = r'''import json
import os
import time
import urllib.error
import urllib.request

base_url = os.environ["MANIFESTO_E2E_URL"].rstrip("/")
deadline = time.monotonic() + int(os.environ["MANIFESTO_E2E_TIMEOUT"])

def request_json(request, request_timeout):
    while True:
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                return json.load(response)
        except urllib.error.URLError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)

models = request_json(f"{base_url}/models", 30)
model = models["data"][0]["id"]

request = urllib.request.Request(
    f"{base_url}/completions",
    data=json.dumps({
        "model": model,
        "prompt": "Hello",
        "max_tokens": 8,
        "temperature": 0,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
completion = request_json(request, 120)

choices = completion.get("choices", [])
if not choices or not isinstance(choices[0].get("text"), str):
    raise RuntimeError(f"invalid completion response: {completion}")
print(json.dumps({"status": "passed", "model": model, "completion": choices[0]["text"]}))
'''


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def ephemeral_cluster(cluster: Cluster) -> Cluster:
    """Return a disposable profile, rejecting hidden PVC dependencies."""

    pvc_volumes = [
        volume.get("name", "<unnamed>")
        for volume in cluster.pod_defaults.extra_volumes
        if "persistentVolumeClaim" in volume
    ]
    if pvc_volumes:
        names = ", ".join(pvc_volumes)
        raise workflow.WorkflowError(
            f"PVC-free e2e cannot use pod_defaults.extra_volumes PVCs: {names}",
            code=2,
        )

    ephemeral = cluster.model_copy(deep=True)
    ephemeral.storage.shared_claim = None
    ephemeral.storage.shared_volume = {"emptyDir": {}}
    ephemeral.storage.local_nvme_path = None
    ephemeral.cache.hf_host_path = None
    ephemeral.cache.jit_host_path = None
    ephemeral.cache.hf_home = f"{ephemeral.storage.shared_mount_path}/hf_cache"
    ephemeral.logging.pvc = None
    ephemeral.logging.root = f"{ephemeral.storage.shared_mount_path}/{{user}}/logs"
    return ephemeral


def _preflight(
    args: argparse.Namespace,
    config: workflow.RuntimeConfig,
    cluster: Cluster,
) -> DeploymentSpec:
    spec = load_spec(workflow.resolve_model(args.spec), cluster)
    workflow.apply_runtime_overrides(spec, args, config)
    objects = render(spec, user=config.user, cluster=cluster)
    if not spec.runtime.vllm_env and (
        any(obj.get("kind") == "PersistentVolumeClaim" for obj in objects)
        or _contains_key(objects, "persistentVolumeClaim")
    ):
        raise workflow.WorkflowError(
            "PVC-free e2e preflight found a persistentVolumeClaim reference",
            code=2,
        )
    return spec


def render_probe_job(
    *,
    name: str,
    instance: Instance,
    url: str,
    image: str,
    timeout: int,
    cluster: Cluster,
) -> dict:
    pod_security_context: dict[str, Any] = {
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    if cluster.platform != "openshift":
        pod_security_context["runAsUser"] = 65532

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "securityContext": pod_security_context,
        "containers": [
            {
                "name": "e2e",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["python", "-c", _PROBE_SCRIPT],
                "env": [
                    {"name": "MANIFESTO_E2E_URL", "value": url},
                    {"name": "MANIFESTO_E2E_TIMEOUT", "value": str(timeout)},
                ],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "resources": {
                    "requests": {"cpu": "50m", "memory": "64Mi"},
                    "limits": {"cpu": "500m", "memory": "256Mi"},
                },
            }
        ],
    }
    if cluster.pod_defaults.affinity:
        pod_spec["affinity"] = copy.deepcopy(cluster.pod_defaults.affinity)
    if cluster.pod_defaults.tolerations:
        pod_spec["tolerations"] = copy.deepcopy(cluster.pod_defaults.tolerations)
    if cluster.pod_defaults.dns_policy:
        pod_spec["dnsPolicy"] = cluster.pod_defaults.dns_policy
    if cluster.pod_defaults.dns_config:
        pod_spec["dnsConfig"] = copy.deepcopy(cluster.pod_defaults.dns_config)

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "labels": instance.labels("e2e-test")},
        "spec": {
            "activeDeadlineSeconds": timeout,
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 300,
            "template": {
                "metadata": {"labels": instance.labels("e2e-test")},
                "spec": pod_spec,
            },
        },
    }


def _probe_url(
    config: workflow.RuntimeConfig,
    cluster: Cluster,
    spec: DeploymentSpec,
) -> str:
    instance = Instance(user=config.user, release=spec.release)
    if spec.routing.kind == RoutingKind.DISABLED:
        role = next((item for item in spec.roles if item.name == "decode"), spec.roles[0])
        port = resolve_role(spec, instance, cluster, role).ports.public[0]
        service = instance.name(f"{role.name}-svc")
    else:
        service = instance.name(
            "gateway", max_length=63 - len(cluster.gateway.class_name) - 1
        ) + f"-{cluster.gateway.class_name}"
        port = 80
    return f"http://{service}.{config.namespace}.svc.cluster.local:{port}/v1"


def _failed_create_event(config: workflow.RuntimeConfig, job_name: str) -> str | None:
    raw = workflow.capture(
        [
            *config.kubectl(),
            "get",
            "events",
            "--field-selector",
            f"involvedObject.kind=Job,involvedObject.name={job_name}",
            "-o",
            "json",
        ],
        check=False,
    )
    try:
        events = json.loads(raw).get("items", [])
    except json.JSONDecodeError:
        return None
    failures = [
        event.get("message", "Job pod creation failed")
        for event in events
        if event.get("type") == "Warning" and event.get("reason") == "FailedCreate"
    ]
    return failures[-1] if failures else None


def _job_state(config: workflow.RuntimeConfig, job_name: str) -> tuple[str, str | None]:
    raw = workflow.capture(
        [*config.kubectl(), "get", f"job/{job_name}", "-o", "json"],
        check=False,
    )
    try:
        status = json.loads(raw).get("status", {})
    except json.JSONDecodeError:
        status = {}
    if status.get("succeeded", 0) > 0:
        return "succeeded", None
    conditions = {
        condition.get("type")
        for condition in status.get("conditions", [])
        if condition.get("status") == "True"
    }
    if status.get("failed", 0) > 0 or conditions & {"Failed", "FailureTarget"}:
        return "failed", "Job reported failure"
    if message := _failed_create_event(config, job_name):
        return "failed-create", message
    return "running", None


def run_probe(
    args: argparse.Namespace,
    config: workflow.RuntimeConfig,
    cluster: Cluster,
    spec: DeploymentSpec,
) -> int:
    instance = Instance(user=config.user, release=spec.release)
    url = _probe_url(config, cluster, spec)
    job_name = instance.name("e2e")
    job = render_probe_job(
        name=job_name,
        instance=instance,
        url=url,
        image=args.image,
        timeout=args.timeout,
        cluster=cluster,
    )
    print(f"Testing {url} from job/{job_name}...")
    rc = workflow.run(
        [*config.kubectl(), "create", "-f", "-"],
        input_text=render_to_yaml([job]),
    )
    if rc:
        return rc

    state = "running"
    detail = None
    try:
        deadline = time.monotonic() + args.timeout
        while state == "running" and time.monotonic() < deadline:
            state, detail = _job_state(config, job_name)
            if state == "running":
                time.sleep(2)

        if state != "failed-create":
            workflow.run([*config.kubectl(), "logs", f"job/{job_name}"])
        if state != "succeeded":
            reason = detail or f"timed out after {args.timeout}s"
            print(f"In-cluster test failed: {reason}", file=sys.stderr)
            return 1
        return 0
    finally:
        if not args.keep_namespace:
            workflow.run(
                [*config.kubectl(), "delete", f"job/{job_name}", "--ignore-not-found=true"]
            )


def _run_lifecycle(
    args: argparse.Namespace,
    config: workflow.RuntimeConfig,
    cluster: Cluster,
    spec: DeploymentSpec,
) -> int:
    if spec.runtime.vllm_env:
        print(f"Deploying with existing vllm-envs worktree {spec.runtime.vllm_env}...")
    else:
        print("Deploying with the image vLLM and disposable emptyDir storage...")
    rc = workflow.deploy(args, config=config, cluster=cluster)
    if rc:
        return rc
    rc = workflow.ready(args, config=config, cluster=cluster)
    if rc:
        return rc
    return run_probe(args, config, cluster, spec)


def e2e(args: argparse.Namespace) -> int:
    """Exercise an image or external vllm-envs environment in a fresh namespace."""

    if args.timeout < 1:
        raise workflow.WorkflowError("--timeout must be at least 1 second", code=2)

    workflow.load_dotenv()
    user = workflow.resolve_user(args.user)
    namespace = args.namespace or Instance(user=user, release="e2e").name(
        f"ns-{uuid.uuid4().hex[:8]}"
    )
    run_args = argparse.Namespace(**vars(args))
    run_args.namespace = namespace
    config = workflow.RuntimeConfig.from_args(run_args)
    configured_cluster = workflow.load_runtime_cluster(config, run_args)
    configured_spec = load_spec(workflow.resolve_model(run_args.spec), configured_cluster)
    use_external_env = (
        run_args.vllm_env is not None or configured_spec.runtime.vllm_env is not None
    )
    cluster = configured_cluster if use_external_env else ephemeral_cluster(configured_cluster)
    spec = _preflight(run_args, config, cluster)

    existing = workflow.capture(
        ["kubectl", "get", f"namespace/{namespace}", "-o", "name"],
        check=False,
    )
    if existing.strip():
        raise workflow.WorkflowError(
            f"Namespace {namespace!r} already exists; e2e requires a fresh namespace.",
            code=2,
        )

    print(f"Creating disposable namespace {namespace}...")
    rc = workflow.run(["kubectl", "create", "namespace", namespace])
    if rc:
        return rc

    result = 0
    try:
        result = _run_lifecycle(run_args, config, cluster, spec)
    finally:
        if run_args.keep_namespace:
            print(f"Keeping namespace {namespace} for inspection.")
        else:
            print(f"Deleting namespace {namespace}...")
            cleanup_rc = workflow.run(
                [
                    "kubectl",
                    "delete",
                    f"namespace/{namespace}",
                    "--wait=true",
                    f"--timeout={NAMESPACE_DELETE_TIMEOUT}s",
                ]
            )
            if result == 0:
                result = cleanup_rc
    return result
