"""Render self-contained Slurm batch scripts and invoke Slurm client tools.

Direct replicas use job arrays; routed replicas share an allocation. Both
use one srun task per node and the first host in each replica as rendezvous.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from decimal import Decimal, ROUND_CEILING

from .cluster import Cluster
from .features import connector_backends
from .instance import Instance
from .launch import build_launch_script
from .resolve import CACHE_DIRS, ResolvedRole, resolve_role
from .slurm_config import SlurmConfig
from .spec import DeploymentSpec, RoleSpec, RoutingFrontend, RoutingKind, TopologyKind


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


def batch_directives(
    settings: SlurmConfig, *, name: str, nodes: int, cpus: int, memory: int, gres: str,
) -> list[str]:
    """Describe one allocation, whether a direct job or a routed component."""
    directives = {
        "job-name": name, "nodes": nodes, "ntasks": nodes, "ntasks-per-node": 1,
        "cpus-per-task": cpus, "mem": f"{memory}M", "gres": gres, "time": settings.time,
    }
    for key in ("partition", "account", "qos", "constraint"):
        if value := getattr(settings, key):
            directives[key] = value
    lines = [f"#SBATCH --{key}={value}" for key, value in directives.items()]
    if settings.exclusive:
        lines.append("#SBATCH --exclusive")
    return lines


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
    routed = spec.routing.kind != RoutingKind.DISABLED
    if not routed and spec.routing.epp is not None:
        raise ValueError("routing profiles require llm-d routing")
    if routed:
        if not cluster.slurm.exclusive:
            raise ValueError("Slurm llm-d requires slurm.exclusive: true because its services use fixed host ports")
        if (spec.routing.kind == RoutingKind.PD) != (spec.topology == TopologyKind.PD):
            raise ValueError("Slurm P/D routing requires topology: pd and routing.kind: pd")
        if spec.routing.frontend != RoutingFrontend.STANDALONE:
            raise ValueError("Slurm llm-d requires the standalone routing frontend")
        replicas = spec.routing.epp.replicas if spec.routing.epp else spec.routing.replicas
        if replicas != 1:
            raise ValueError("Slurm supports one llm-d router per deployment")
        expected = {"prefill", "decode"} if spec.topology == TopologyKind.PD else {spec.routing.target_role}
        if {role.name for role in spec.roles} != expected:
            raise ValueError(f"Slurm routing requires exactly these roles: {sorted(expected)}")
    elif len(spec.roles) != 1:
        raise ValueError("Slurm requires exactly one direct serving role; use lws.replicas for replicas")
    runtime = spec.runtime
    if "sidecars" in runtime.model_fields_set and runtime.sidecars:
        raise ValueError("Slurm does not support runtime.sidecars; set sidecars: []")
    if "idle_shutdown" in runtime.model_fields_set and runtime.idle_shutdown.enabled:
        raise ValueError("Slurm does not support idle shutdown; use slurm.time and disable idle_shutdown")
    spec.apply_cluster_defaults(cluster)
    for role in spec.roles:
        if role.workload != "auto":
            raise ValueError("Slurm requires workload: auto; Kubernetes workload kinds are unsupported")
        if not routed and (role.routing_proxy or role.kv_transfer_config):
            raise ValueError("Slurm routing proxies and KV-transfer connectors require llm-d routing")
        if spec.topology == TopologyKind.PD and connector_backends(role.kv_transfer_config) != {"NixlConnector"}:
            raise ValueError("Slurm P/D requires NixlConnector on both serving roles")
        if role.resources.ephemeral_storage or role.shm_size:
            raise ValueError("Slurm cannot allocate ephemeral_storage or shm_size; configure these on the host")
        cpu_count(role.resources.cpu)
        memory_mib(role.resources.memory)


def launch_task(spec: DeploymentSpec, role: RoleSpec, resolved: ResolvedRole, *, routed: bool = False) -> str:
    """Prepare one scheduled node, then reuse the platform-independent vLLM launch."""
    task = [
        "set -euo pipefail",
        'export HOSTNAME="${SLURMD_NODENAME:-$(hostname)}"',
        'export MANIFESTO_POD_UID="${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"',
    ]
    if routed:
        task += [
            'MANIFESTO_ROLE_HOSTS=(); while IFS= read -r host; do MANIFESTO_ROLE_HOSTS+=("$host"); done <<< "$MANIFESTO_HOSTS"',
            'MANIFESTO_ROLE_IPS=(); while IFS= read -r address; do MANIFESTO_ROLE_IPS+=("$address"); done <<< "$MANIFESTO_IPS"',
            f'export LWS_WORKER_INDEX=$(( SLURM_PROCID % {role.lws.size} ))',
            'export LWS_LEADER_ADDRESS="${MANIFESTO_ROLE_HOSTS[SLURM_PROCID-LWS_WORKER_INDEX]}"',
            f'MANIFESTO_ROLE={shlex.quote(role.name)}',
            f'export MANIFESTO_POD_UID="${{SLURM_JOB_ID}}-${{MANIFESTO_ROLE}}-$(( SLURM_PROCID / {role.lws.size} ))"',
        ]
    else:
        task.append('export LWS_WORKER_INDEX="$SLURM_PROCID"')
    for key, value in resolved.env.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        task.append(f"export {key}={shlex.quote(value)}")
    for contribution in resolved.features.field_ref_env:
        if contribution.field_path != "status.podIP":
            raise ValueError(f"unsupported Slurm environment source: {contribution.field_path}")
        task.append(f'export {contribution.name}="${{MANIFESTO_ROLE_IPS[SLURM_PROCID]}}"')
    if resolved.persistent_cache:
        for key in CACHE_DIRS:
            task.append(f'export {key}="${{{key}}}/${{MANIFESTO_POD_UID}}"')
    task.append(build_launch_script(spec, role, resolved, respect_visible_devices=True))
    return "\n".join(task)


def step_command(
    settings: SlurmConfig, image: str, command: str, options: list[str], *,
    gpu: bool = False, router_config: bool = False, host_variable: str | None = None,
) -> str:
    """Run a command without requiring a shell in service container images."""
    env = ("LWS_LEADER_ADDRESS", "MANIFESTO_HOSTS", "MANIFESTO_IPS", "CUDA_VISIBLE_DEVICES", "HF_TOKEN")
    mounts = [f"{bind.source}:{bind.target}:" + ("ro" if bind.read_only else "rw") for bind in settings.binds]
    srun = ["srun", *options, "--cpu-bind=none", "--kill-on-bad-exit=1", "--wait=30", "--export=ALL"]
    prefix = shlex.join(srun)
    if host_variable:
        prefix += f' --nodelist="${{{host_variable}}}"'
    if settings.runtime == "pyxis":
        prefix += " " + shlex.join([
            f"--container-image={_pyxis_image(image)}", "--no-container-entrypoint",
            f"--container-env={','.join(env)}",
        ])
        if router_config:
            mount_prefix = ",".join(mounts) + ("," if mounts else "")
            prefix += " --container-mounts=" + shlex.quote(mount_prefix) + '\"$MANIFESTO_ROUTER_DIR:$MANIFESTO_ROUTER_DIR:ro\"'
        elif mounts:
            prefix += " " + shlex.quote(f"--container-mounts={','.join(mounts)}")
    elif settings.runtime in {"apptainer", "singularity"}:
        env_prefix = "APPTAINERENV" if settings.runtime == "apptainer" else "SINGULARITYENV"
        # Slurm assigns the GPU mask per task, so forward it on the compute
        # node, not from the batch process. The wrapper stays outside the image.
        forward = "\n".join(f'if [ "${{{key}+set}}" = set ]; then export {env_prefix}_{key}="${{{key}}}"; fi' for key in env)
        prefix += " " + shlex.join(["bash", "-c", forward + '\nexec "$@"', "manifesto", settings.runtime, "exec", "--no-eval"])
        if gpu:
            prefix += " --nv"
        for mount in mounts:
            prefix += " --bind " + shlex.quote(mount)
        if router_config:
            prefix += ' --bind "$MANIFESTO_ROUTER_DIR:$MANIFESTO_ROUTER_DIR:ro"'
        if not ("://" in image or image.startswith(("/", "./", "../")) or image.endswith(".sif")):
            image = f"docker://{image}"
        prefix += " " + shlex.quote(image)
    return prefix + " " + command


def render_slurm(
    spec: DeploymentSpec,
    *,
    user: str,
    cluster: Cluster,
    header: list[str] | None = None,
) -> str:
    validate_slurm(spec, cluster)
    if spec.routing.kind != RoutingKind.DISABLED:
        from .slurm_routing import render_routed
        return render_routed(spec, user=user, cluster=cluster, header=header)
    settings = cluster.slurm
    assert settings is not None
    role = spec.roles[0]
    instance = Instance(user, spec.release, include_user_in_name=cluster.naming.user_prefix)
    resolved = resolve_role(spec, instance, cluster, role)
    allocation = spec.accelerator_config(cluster).allocation.slurm
    assert allocation is not None
    name = f"manifesto-{instance.instance_id}"
    task_options = {
        "ntasks": str(role.lws.size),
        "ntasks-per-node": "1",
        "cpus-per-task": str(cpu_count(role.resources.cpu)),
    }
    lines = ["#!/bin/bash", *batch_directives(
        settings, name=name, nodes=role.lws.size, cpus=cpu_count(role.resources.cpu),
        memory=memory_mib(role.resources.memory), gres=f"{allocation.gres}:{role.gpus_per_pod}",
    ), f"#SBATCH --output={name}-%A_%a.out", "#SBATCH --export=ALL"]
    if role.lws.replicas > 1:
        lines.append(f"#SBATCH --array=0-{role.lws.replicas - 1}")
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
    lines.append('export HF_TOKEN="${HF_TOKEN:-}"')
    lines.extend(_script_variable("MANIFESTO_TASK", launch_task(spec, role, resolved)))
    command = step_command(settings, spec.model.image, 'bash -c "$MANIFESTO_TASK"',
                           [f"--{key}={value}" for key, value in task_options.items()], gpu=True)
    lines.append("exec " + command)
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
    if not re.fullmatch(r"[0-9]+(?:[_+][0-9]+)?", args.job_id):
        raise ValueError("Slurm job ID must be numeric, optionally followed by _ARRAY_INDEX or +COMPONENT")
    _capture(["scancel", args.job_id], cluster=_client_cluster(args))
    print(f"Stopped Slurm job {args.job_id}.")
    return 0


def _client_cluster(args) -> Cluster | None:
    import os
    from .workflow import load_dotenv, load_cluster_with_overrides, resolve_cluster

    load_dotenv()
    name = getattr(args, "cluster", None) or os.environ.get("MANIFESTO_CLUSTER")
    return load_cluster_with_overrides(resolve_cluster(name), args) if name else None
