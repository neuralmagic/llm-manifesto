"""Render self-contained Slurm batch scripts and invoke Slurm client tools.

One array element is one serving replica, with one srun task per node. All
nodes in a replica share the allocation's first host as their rendezvous.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from decimal import Decimal, ROUND_CEILING

from .cluster import Cluster
from .instance import Instance
from .launch import build_launch_script
from .parallelism import parallel_layout
from .resolve import POD_CACHE_DIRS, resolve_role
from .spec import DeploymentSpec, RoutingKind, TopologyKind


def _ceil(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def cpu_count(value: str) -> int:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?m?", value):
        raise ValueError(f"unsupported Slurm CPU quantity: {value!r}")
    count = Decimal(value.removesuffix("m")) / (1000 if value.endswith("m") else 1)
    if count <= 0:
        raise ValueError("Slurm CPU requests must be positive")
    return _ceil(count)


def memory_mib(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]i|[kMGTPE])?", value)
    if not match:
        raise ValueError(f"unsupported Slurm memory quantity: {value!r}")
    suffix = match[2] or ""
    power = "KMGTPE".index(suffix[0].upper()) + 1 if suffix else 0
    size = Decimal(match[1]) * ((1024 if suffix.endswith("i") else 1000) ** power)
    if size <= 0:
        raise ValueError("Slurm memory requests must be positive")
    return _ceil(size / (1024 ** 2))


def _script_variable(name: str, script: str) -> list[str]:
    delimiter = f"{name}_EOF"
    while delimiter in script.splitlines():
        delimiter += "_"
    # read reaches EOF (status 1) because shell scripts contain no NUL. Avoid
    # command substitution: older Bash versions misparse nested heredocs there.
    return [f"IFS= read -r -d '' {name} <<'{delimiter}' || true", script, delimiter]


def _pyxis_image(image: str) -> str:
    image = image.removeprefix("docker://")
    if image.startswith(("/", "./", "../")) or "#" in image:
        return image
    registry, separator, path = image.partition("/")
    if separator and ("." in registry or ":" in registry or registry == "localhost"):
        return f"{registry}#{path}"
    return image


def validate_slurm(spec: DeploymentSpec, cluster: Cluster) -> None:
    if cluster.platform != "slurm" or cluster.slurm is None:
        raise ValueError("Slurm rendering requires a platform: slurm cluster profile")
    if spec.topology != TopologyKind.AGGREGATED or spec.routing.kind != RoutingKind.DISABLED:
        raise ValueError("Slurm currently supports aggregated serving with routing.kind: disabled")
    if len(spec.roles) != 1:
        raise ValueError("Slurm requires exactly one serving role; use lws.replicas for replicas")
    runtime = spec.runtime
    if "sidecars" in runtime.model_fields_set and runtime.sidecars:
        raise ValueError("Slurm does not support runtime.sidecars; set sidecars: []")
    if "idle_shutdown" in runtime.model_fields_set and runtime.idle_shutdown.enabled:
        raise ValueError("Slurm does not support idle shutdown; use slurm.time and disable idle_shutdown")
    role = spec.roles[0]
    if role.workload != "auto":
        raise ValueError("Slurm requires workload: auto; Kubernetes workload kinds are unsupported")
    if role.routing_proxy or role.kv_transfer_config:
        raise ValueError("Slurm does not support routing proxies or KV-transfer connectors")
    if role.resources.ephemeral_storage or role.shm_size:
        raise ValueError("Slurm cannot allocate ephemeral_storage or shm_size; configure these on the host")
    spec.apply_cluster_defaults(cluster)
    cpu_count(role.resources.cpu)
    memory_mib(role.resources.memory)


def render_slurm(
    spec: DeploymentSpec,
    *,
    user: str,
    cluster: Cluster,
    header: list[str] | None = None,
) -> str:
    validate_slurm(spec, cluster)
    settings = cluster.slurm
    assert settings is not None
    role = spec.roles[0]
    layout = parallel_layout(role)
    instance = Instance(user, spec.release, include_user_in_name=cluster.naming.user_prefix)
    resolved = resolve_role(spec, instance, cluster, role)
    allocation = spec.accelerator_config(cluster).allocation.slurm
    assert allocation is not None
    name = f"manifesto-{instance.instance_id}"
    directives = {
        "job-name": name,
        "nodes": str(role.lws.size),
        "ntasks": str(role.lws.size),
        "ntasks-per-node": "1",
        "cpus-per-task": str(cpu_count(role.resources.cpu)),
        "mem": f"{memory_mib(role.resources.memory)}M",
        "gres": f"{allocation.gres}:{role.gpus_per_pod}",
        "time": settings.time,
        "output": f"{name}-%A_%a.out",
        "export": "ALL",
    }
    for key in ("partition", "account", "qos", "constraint"):
        if value := getattr(settings, key):
            directives[key] = value
    if role.lws.replicas > 1:
        directives["array"] = f"0-{role.lws.replicas - 1}"
    lines = ["#!/bin/bash", *(f"#SBATCH --{key}={value}" for key, value in directives.items())]
    if settings.exclusive:
        lines.append("#SBATCH --exclusive")
    lines += [
        "# Generated by Manifesto. Inspect or edit before submitting with sbatch.",
        "# Slurm wall time controls lifetime; Kubernetes sidecars and idle shutdown are not installed.",
        *(f"# {line}" for line in header or []),
        "set -euo pipefail",
        *settings.setup,
        'MANIFESTO_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST")',
        'export LWS_LEADER_ADDRESS="${MANIFESTO_HOSTS%%$\'\\n\'*}"',
        'test -n "$LWS_LEADER_ADDRESS"',
        f'echo "Manifesto endpoint: http://${{LWS_LEADER_ADDRESS}}:{resolved.ports.backend[0]}"',
    ]
    # Literal heredocs keep the artifact readable and self-contained. Slurm
    # transfers only the batch script, not adjacent files.
    launch = ["set -euo pipefail"]
    for key, value in resolved.env.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        launch.append(f"export {key}={shlex.quote(value)}")
    if resolved.persistent_cache:
        for key in POD_CACHE_DIRS:
            launch.append(f'export {key}="${{{key}}}/${{MANIFESTO_POD_UID}}"')
    launch.append(build_launch_script(
        spec, role, resolved.ports,
        log_dir=resolved.log_dir,
        trace_dir=resolved.trace_dir,
        vllm_env=resolved.vllm_env,
        persistent_cache=resolved.persistent_cache,
        vllm_args=resolved.vllm_args,
        distributed_dp=layout.distributed_dp,
        respect_visible_devices=True,
        vllm_raw_args=resolved.vllm_raw_args,
    ))
    task = [
        "set -euo pipefail",
        'export LWS_WORKER_INDEX="$SLURM_PROCID"',
        'export HOSTNAME="${SLURMD_NODENAME:-$(hostname)}"',
        'export MANIFESTO_POD_UID="${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"',
    ]
    command = []
    if settings.runtime in {"apptainer", "singularity"}:
        prefix = "APPTAINERENV" if settings.runtime == "apptainer" else "SINGULARITYENV"
        for key in ("LWS_WORKER_INDEX", "LWS_LEADER_ADDRESS", "HOSTNAME", "MANIFESTO_POD_UID", "CUDA_VISIBLE_DEVICES", "HF_TOKEN"):
            task.append(f'if [ "${{{key}+set}}" = set ]; then export {prefix}_{key}="${{{key}}}"; fi')
        image = spec.model.image
        if not ("://" in image or image.startswith(("/", "./", "../")) or image.endswith(".sif")):
            image = f"docker://{image}"
        container = [settings.runtime, "exec", "--nv", "--no-eval"]
        for bind in settings.binds:
            container += ["--bind", f"{bind.source}:{bind.target}" + (":ro" if bind.read_only else ":rw")]
        command = [*container, image]
    task.extend(_script_variable("MANIFESTO_LAUNCH", "\n".join(launch)))
    task.append("exec " + shlex.join([*command, "bash", "-c"]) + ' "$MANIFESTO_LAUNCH"')
    lines.extend(_script_variable("MANIFESTO_TASK", "\n".join(task)))
    srun = [
        "srun", f"--ntasks={role.lws.size}", "--ntasks-per-node=1",
        f"--cpus-per-task={cpu_count(role.resources.cpu)}",
        "--cpu-bind=none", "--kill-on-bad-exit=1", "--wait=30", "--export=ALL",
    ]
    if settings.runtime == "pyxis":
        image = _pyxis_image(spec.model.image)
        lines.append('export HF_TOKEN="${HF_TOKEN:-}"')
        srun += [f"--container-image={image}", "--no-container-entrypoint",
                 "--container-env=LWS_LEADER_ADDRESS,HF_TOKEN"]
        if settings.binds:
            mounts = ",".join(f"{bind.source}:{bind.target}" + (":ro" if bind.read_only else ":rw") for bind in settings.binds)
            srun.append(f"--container-mounts={mounts}")
    lines.append("exec " + shlex.join([*srun, "bash", "-c"]) + ' "$MANIFESTO_TASK"')
    return "\n".join(lines) + "\n"


def _capture(command: list[str], *, script: str | None = None, cluster: Cluster | None = None) -> str:
    from .workflow import WorkflowError

    if cluster is not None:
        if cluster.platform != "slurm" or cluster.slurm is None:
            raise ValueError("Slurm commands require a platform: slurm profile")
        if cluster.slurm.ssh_host:
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                       "--", cluster.slurm.ssh_host, shlex.join(command)]
    try:
        result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise WorkflowError(f"{command[0]} timed out after 60 seconds; check Slurm before retrying") from exc
    if result.returncode:
        raise WorkflowError(result.stderr.strip() or f"{command[0]} failed", code=result.returncode)
    return result.stdout.strip() or result.stderr.strip()


def submit(script: str, *, cluster: Cluster | None = None, test_only: bool = False) -> int:
    """Submit once; retries could create duplicate allocations."""
    job = _capture(["sbatch", "--test-only" if test_only else "--parsable"], script=script, cluster=cluster)
    print(job)
    return 0


def servers(args) -> int:
    result = _capture([
        "squeue", "--noheader", "--array", f"--user={args.user}" if args.user else "--me",
        "--format=%i|%j|%T|%N|%R",
    ], cluster=_client_cluster(args))
    records = []
    for line in result.splitlines():
        fields = line.strip().split("|", 4)
        if len(fields) == 5 and fields[1].startswith("manifesto-"):
            records.append(dict(zip(("job_id", "name", "state", "nodes", "reason"), fields)))
    if args.output == "json":
        print(json.dumps(records, indent=2))
    elif args.output == "name":
        for record in records:
            print(record["job_id"])
    else:
        print("JOB ID  NAME  STATE  NODES  REASON")
        for record in records:
            print("  ".join(record.values()))
    return 0


def stop(args) -> int:
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", args.job_id):
        raise ValueError("Slurm job ID must be numeric, optionally followed by _ARRAY_INDEX")
    _capture(["scancel", args.job_id], cluster=_client_cluster(args))
    print(f"Stopped Slurm job {args.job_id}.")
    return 0


def _client_cluster(args) -> Cluster | None:
    import os
    from .workflow import load_dotenv, load_cluster_with_overrides, resolve_cluster

    load_dotenv()
    name = getattr(args, "cluster", None) or os.environ.get("MANIFESTO_CLUSTER")
    return load_cluster_with_overrides(resolve_cluster(name), args) if name else None
