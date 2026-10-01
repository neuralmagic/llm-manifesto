"""Run the llm-d file-discovery stack in one Slurm allocation.

Each serving role is a heterogeneous job component. Its replicas occupy
consecutive groups of nodes. The router shares the first component's first
node; each decode proxy shares its vLLM node. No shared config filesystem or
Kubernetes API is needed.
"""

from __future__ import annotations

import json
import shlex

import yaml

from .cluster import Cluster
from .instance import Instance
from .resolve import ResolvedRole, resolve_role
from .routing import ENVOY_CONFIG, epp_image, plugin_configs, plugins_config_file, proxy_args
from .slurm import _script_variable, cpu_count, launch_task, memory_mib, step_command
from .spec import DeploymentSpec, RoutingSpec


def file_discovery_config(routing: RoutingSpec, endpoint_path: str) -> dict[str, str]:
    configs = plugin_configs(routing)
    if any(name in {".", ".."} for name in configs):
        raise ValueError("routing config names must be filenames")
    selected = plugins_config_file(routing)
    config = yaml.safe_load(configs[selected])
    plugins = config.setdefault("plugins", [])
    name = "manifesto-file-discovery"
    if any(plugin.get("name", plugin["type"]) == name for plugin in plugins):
        raise ValueError(f"routing plugin name {name!r} is reserved for Slurm discovery")
    layer = config.setdefault("dataLayer", {})
    if layer.get("discovery"):
        raise ValueError("cluster backend owns endpoint discovery; remove routing dataLayer.discovery")
    plugins.append({"type": "file-discovery", "name": name,
                    "parameters": {"path": endpoint_path, "watchFile": False}})
    layer["discovery"] = {"endpoints": {"pluginRef": name}}
    configs[selected] = yaml.safe_dump(config, sort_keys=False)
    return configs


def _prepare_config(spec: DeploymentSpec, resolved: dict[str, ResolvedRole], port: int) -> str:
    """A stdlib-only program run on the batch node after hosts are allocated."""
    roles = [
        {"name": role.name, "size": role.lws.size, "replicas": role.lws.replicas,
         "api_nodes": resolved[role.name].api_nodes,
         "ports": (resolved[role.name].ports.public if role.routing_proxy else resolved[role.name].ports.backend)}
        for role in spec.roles
    ]
    configs = file_discovery_config(spec.routing, "MANIFESTO_ENDPOINTS")
    envoy = yaml.safe_load(ENVOY_CONFIG)
    for listener in envoy["static_resources"]["listeners"]:
        if listener["name"] == "vllm":
            listener["address"]["socket_address"]["port_value"] = port
    # Keep the same TLS connection to EPP used by the Kubernetes renderer.
    envoy_config = yaml.safe_dump(envoy, sort_keys=False)
    return f'''import json, os, socket
from pathlib import Path
root = Path(os.environ["MANIFESTO_ROUTER_DIR"])
(root / "epp").mkdir()
configs = json.loads({json.dumps(configs)!r})
for name, content in configs.items():
    (root / "epp" / name).write_text(content.replace("MANIFESTO_ENDPOINTS", str(root / "endpoints.json")))
(root / "envoy.yaml").write_text({envoy_config!r})
roles = json.loads({json.dumps(roles)!r})
endpoints = []
for group, role in enumerate(roles):
    hosts = os.environ[f"MANIFESTO_HOSTS_{{group}}"].splitlines()
    if len(hosts) != role["size"] * role["replicas"]:
        raise RuntimeError(f"wrong node count for {{role['name']}}: {{hosts}}")
    ips = [socket.gethostbyname(host) for host in hosts]
    (root / f"ips-{{group}}").write_text("\\n".join(ips))
    api_hosts = []
    for node, (host, address) in enumerate(zip(hosts, ips)):
        replica, worker = divmod(node, role["size"])
        if worker not in role["api_nodes"]:
            continue
        api_hosts.append(host)
        for port in role["ports"]:
            endpoints.append({{
                "name": f"{{role['name']}}-{{replica}}-{{worker}}-{{port}}",
                "namespace": {spec.namespace!r}, "address": address, "port": str(port),
                "labels": {{"llm-d.ai/role": role["name"], "model": {spec.model.id!r},
                           "leaderworkerset.sigs.k8s.io/worker-index": str(worker)}},
            }})
    (root / f"api-hosts-{{group}}").write_text(",".join(api_hosts))
(root / "endpoints.json").write_text(json.dumps({{"endpoints": endpoints}}, indent=2))
'''


def render_routed(
    spec: DeploymentSpec, *, user: str, cluster: Cluster, header: list[str] | None = None,
) -> str:
    settings = cluster.slurm
    assert settings is not None
    instance = Instance(user, spec.release, include_user_in_name=cluster.naming.user_prefix)
    resolved = {role.name: resolve_role(spec, instance, cluster, role) for role in spec.roles}
    gpu = spec.accelerator_config(cluster).allocation.slurm
    assert gpu is not None
    name = f"manifesto-{instance.instance_id}"
    heterogeneous = len(spec.roles) > 1
    router_mem = memory_mib(settings.router_memory)
    proxy_mem = memory_mib(settings.proxy_memory)
    if router_mem < 2:
        raise ValueError("slurm.router_memory must reserve at least 1 MiB each for Envoy and EPP")
    router_ports = {settings.port, 9002, 9003, 9090, 19000, 19001}
    if len(router_ports) != 6:
        raise ValueError("slurm.port conflicts with an llm-d internal port")
    first = resolved[spec.roles[0].name]
    if router_ports.intersection(first.ports.public + first.ports.backend + [5555, 8100]):
        raise ValueError("llm-d router ports conflict with serving ports on the first node")
    lines = ["#!/bin/bash"]
    for group, role in enumerate(spec.roles):
        if group:
            lines.append("#SBATCH hetjob")
        cpus = cpu_count(role.resources.cpu)
        memory = memory_mib(role.resources.memory)
        if role.routing_proxy:
            cpus += settings.proxy_cpus
            memory += proxy_mem
        if group == 0:
            cpus += settings.router_cpus
            memory += router_mem
        directives = {
            "job-name": name, "nodes": role.lws.size * role.lws.replicas,
            "ntasks": role.lws.size * role.lws.replicas, "ntasks-per-node": 1,
            "cpus-per-task": cpus, "mem": f"{memory}M",
            "gres": f"{gpu.gres}:{role.gpus_per_pod}", "time": settings.time,
        }
        if group == 0:
            directives.update({"output": f"{name}-%j.out", "export": "ALL"})
        for key in ("partition", "account", "qos", "constraint"):
            if value := getattr(settings, key):
                directives[key] = value
        lines.extend(f"#SBATCH --{key}={value}" for key, value in directives.items())
        if settings.exclusive:
            lines.append("#SBATCH --exclusive")
    lines += [
        "# Generated by Manifesto. llm-d uses file discovery; all services share this job's lifetime.",
        *(f"# {line}" for line in header or []),
        "set -euo pipefail",
        *settings.setup,
        'export HF_TOKEN="${HF_TOKEN:-}"',
        'MANIFESTO_ROUTER_DIR=$(mktemp -d "${TMPDIR:-/tmp}/manifesto-${SLURM_JOB_ID}.XXXXXX")',
        "export MANIFESTO_ROUTER_DIR",
        "MANIFESTO_PIDS=()",
        "cleanup() {",
        "  status=$?",
        "  trap - EXIT",
        "  set +e",
        '  if [ "${#MANIFESTO_PIDS[@]}" -gt 0 ]; then',
        '    kill -TERM "${MANIFESTO_PIDS[@]}" 2>/dev/null',
        "    for attempt in {1..30}; do",
        "      running=0",
        '      for pid in "${MANIFESTO_PIDS[@]}"; do',
        '        if kill -0 "$pid" 2>/dev/null; then running=1; fi',
        "      done",
        '      if [ "$running" -eq 0 ]; then break; fi',
        "      sleep 1",
        "    done",
        '    kill -KILL "${MANIFESTO_PIDS[@]}" 2>/dev/null',
        '    wait "${MANIFESTO_PIDS[@]}" 2>/dev/null',
        "  fi",
        '  rm -rf -- "$MANIFESTO_ROUTER_DIR"',
        '  exit "$status"',
        "}",
        "trap cleanup EXIT",
        "trap 'exit 143' TERM",
        "trap 'exit 130' INT",
    ]
    for group, role in enumerate(spec.roles):
        nodelist = f"SLURM_JOB_NODELIST_HET_GROUP_{group}" if heterogeneous else "SLURM_JOB_NODELIST"
        lines.append(f'export MANIFESTO_HOSTS_{group}=$(scontrol show hostnames "${{{nodelist}}}")')
    lines += _script_variable("MANIFESTO_PREPARE", _prepare_config(spec, resolved, settings.port))
    lines += ['python3 -c "$MANIFESTO_PREPARE"',
              'MANIFESTO_ROUTER_HOST="${MANIFESTO_HOSTS_0%%$\'\\n\'*}"',
              f'echo "Manifesto endpoint: http://${{MANIFESTO_ROUTER_HOST}}:{settings.port}"']

    def start(command: str) -> None:
        lines.extend([command + " &", 'MANIFESTO_PIDS+=("$!")'])

    for group, role in enumerate(spec.roles):
        plan = resolved[role.name]
        options = [f"--het-group={group}"] if heterogeneous else []
        options += ["--overlap", "--ntasks-per-node=1"]
        count = role.lws.size * role.lws.replicas
        lines += [f'export MANIFESTO_HOSTS="$MANIFESTO_HOSTS_{group}"',
                  f'export MANIFESTO_IPS=$(cat "$MANIFESTO_ROUTER_DIR/ips-{group}")']
        lines += _script_variable("MANIFESTO_TASK", launch_task(spec, role, plan, routed=True))
        start(step_command(settings, spec.model.image, 'bash -c "$MANIFESTO_TASK"', [
            *options, f"--nodes={count}", f"--ntasks={count}",
            f"--cpus-per-task={cpu_count(role.resources.cpu)}", f"--mem={memory_mib(role.resources.memory)}M",
            f"--gres={gpu.gres}:{role.gpus_per_pod}",
        ], gpu=True))
        if role.routing_proxy:
            count = len(plan.api_nodes) * role.lws.replicas
            binary = "pd-sidecar" if settings.runtime == "native" else "/app/pd-sidecar"
            lines.append(f'MANIFESTO_PROXY_HOSTS=$(cat "$MANIFESTO_ROUTER_DIR/api-hosts-{group}")')
            command = step_command(settings, cluster.llm_d.routing_sidecar, shlex.join([binary, *proxy_args(plan.ports)]), [
                *options, f"--nodes={count}", f"--ntasks={count}",
                f"--cpus-per-task={settings.proxy_cpus}", f"--mem={proxy_mem}M", "--gres=none",
            ], host_variable="MANIFESTO_PROXY_HOSTS")
            start(command)
    options = ["--het-group=0"] if heterogeneous else []
    options += ["--overlap", "--nodes=1", "--ntasks=1", "--ntasks-per-node=1",
                f"--cpus-per-task={settings.router_cpus // 2}", f"--mem={router_mem // 2}M", "--gres=none"]
    filename = plugins_config_file(spec.routing)
    for image, command in [
        (epp_image(spec.routing, cluster), shlex.join([
            "epp" if settings.runtime == "native" else "/app/epp",
            f"--pool-name={instance.name('infpool')}", f"--pool-namespace={spec.namespace}",
            "--grpc-port=9002", "--grpc-health-port=9003", "--metrics-port=9090",
        ]) + f' "--config-file=$MANIFESTO_ROUTER_DIR/epp/{filename}"'),
        (cluster.llm_d.envoy, shlex.join([
            "envoy" if settings.runtime == "native" else "/usr/local/bin/envoy",
            "--log-level", "warn", "--concurrency", str(settings.router_cpus // 2),
            "--drain-strategy", "immediate", "--drain-time-s", "1",
        ]) + ' -c "$MANIFESTO_ROUTER_DIR/envoy.yaml"'),
    ]:
        command = step_command(settings, image, command, options, router_config=True, host_variable="MANIFESTO_ROUTER_HOST")
        start(command)
    lines += [
        "# Any service exit ends the deployment; Slurm cancellation also stops every step.",
        "while true; do",
        '  for pid in "${MANIFESTO_PIDS[@]}"; do',
        '    if ! kill -0 "$pid" 2>/dev/null; then',
        '      status=0; wait "$pid" || status=$?',
        '      if [ "$status" -eq 0 ]; then status=1; fi',
        '      exit "$status"',
        "    fi",
        "  done",
        "  sleep 1",
        "done",
    ]
    return "\n".join(lines) + "\n"
