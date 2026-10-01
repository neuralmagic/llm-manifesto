"""Exercise complete routed allocations, including failure and cancellation."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import yaml

from manifesto.cluster import load_cluster
from manifesto.instance import Instance
from manifesto.render import render, render_kubernetes
from manifesto.resolve import resolve_role
from manifesto.routing import plugin_configs
from manifesto.slurm_routing import _prepare_config, file_discovery_config
from manifesto.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/qwen/qwen3-0.6b-pd.yaml"


@pytest.fixture
def deployment():
    cluster = load_cluster(ROOT / "clusters/example-slurm.yaml")
    cluster.paths.cache_root = None
    cluster.paths.log_root = None
    return cluster, load_spec(MODEL, cluster)


def test_pd_model_is_shared_by_backends(deployment):
    cluster, spec = deployment
    script = render(spec, cluster=cluster, user="test")
    assert "#SBATCH hetjob" in script
    assert "#SBATCH --cpus-per-task=16" in script  # prefill + router
    assert "#SBATCH --cpus-per-task=10" in script  # decode + proxy
    assert "#SBATCH --array" not in script
    kube = load_cluster(ROOT / "clusters/example-stateless-b200.yaml")
    assert render_kubernetes(load_spec(MODEL, kube), cluster=kube, user="test")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_file_discovery_preserves_policy(deployment):
    _, spec = deployment
    original = plugin_configs(spec.routing)
    selected = yaml.safe_load(file_discovery_config(spec.routing, "/tmp/endpoints.json")["plugins.yaml"])
    policy = yaml.safe_load(original["plugins.yaml"])
    assert selected["schedulingProfiles"] == policy["schedulingProfiles"]
    assert selected["plugins"][:-1] == policy["plugins"]
    assert selected["dataLayer"]["discovery"] == {"endpoints": {"pluginRef": "manifesto-file-discovery"}}
    assert plugin_configs(spec.routing) == original


@pytest.mark.parametrize(("tp", "dp", "nodes", "api_nodes", "ports"), [
    (1, False, 1, [0], [8000]),
    (16, False, 2, [0], [8000]),
    (2, 4, 2, [0, 1], [8000, 8001]),
    (8, 2, 4, [0, 2], [8000]),
])
def test_inventory_contains_only_api_endpoints(tmp_path, deployment, tp, dp, nodes, api_nodes, ports):
    cluster, spec = deployment
    for role in spec.roles:
        role.lws.size, role.lws.replicas = nodes, 2
        role.parallelism.tp, role.parallelism.dp = tp, dp
    instance = Instance("test", spec.release)
    resolved = {r.name: resolve_role(spec, instance, cluster, r) for r in spec.roles}
    env = dict(os.environ, MANIFESTO_ROUTER_DIR=str(tmp_path))
    for group in range(2):
        env[f"MANIFESTO_HOSTS_{group}"] = "\n".join(f"127.0.{group}.{i+1}" for i in range(nodes * 2))
    subprocess.run([sys.executable, "-c", _prepare_config(spec, resolved, 8081)], env=env, check=True)
    endpoints = json.loads((tmp_path / "endpoints.json").read_text())["endpoints"]
    expected = {(r.name, f"127.0.{g}.{replica * nodes + worker + 1}", str(port))
                for g, r in enumerate(spec.roles) for replica in range(2)
                for worker in api_nodes for port in ports}
    actual = {(e["labels"]["llm-d.ai/role"], e["address"], e["port"]) for e in endpoints}
    assert actual == expected
    assert len({e["name"] for e in endpoints}) == len(endpoints)
    assert all(e["address"] != "" for e in endpoints)


# Fake Slurm launches real shell tasks concurrently. Container images are
# simulated, but their entrypoints and environment precedence are enforced.
FAKE_TOOLS = r'''
import json, os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(os.environ["CAPTURE"])
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "scontrol":
    print(args[-1].replace(",", "\n"))
elif name == "srun":
    options = {}
    while args and args[0].startswith("--"):
        key, _, value = args.pop(0).partition("=")
        options[key] = value
    group = options.get("--het-group", "0")
    hosts = options.get("--nodelist", os.environ.get("SLURM_JOB_NODELIST_HET_GROUP_" + group, os.environ["SLURM_JOB_NODELIST"])).split(",")
    assert len(hosts) == int(options["--ntasks"]), options
    children = []
    def stop(*_):
        for child in children:
            child.terminate()
    signal.signal(signal.SIGTERM, stop)
    for rank, host in enumerate(hosts):
        env = dict(os.environ, SLURM_PROCID=str(rank), SLURMD_NODENAME=host, CUDA_VISIBLE_DEVICES=f"GPU-{host}")
        if "--container-image" in options:
            for key in ("MANIFESTO_HOSTS", "MANIFESTO_IPS", "CUDA_VISIBLE_DEVICES"):
                if key not in options["--container-env"].split(","):
                    env[key] = "wrong-image-default"
        command = [str(root / pathlib.Path(args[0]).name), *args[1:]] if args[0].startswith("/app/") or args[0].startswith("/usr/local/bin/envoy") else args
        children.append(subprocess.Popen(command, env=env))
    codes = [p.wait() for p in children]
    sys.exit(next((c for c in codes if c), 0))
elif name in ("apptainer", "singularity"):
    assert args.pop(0) == "exec"
    while args[0].startswith("--"):
        key = args.pop(0)
        if key == "--bind":
            args.pop(0)
    args.pop(0)  # image
    for key, value in list(os.environ.items()):
        if key.startswith(("APPTAINERENV_", "SINGULARITYENV_")):
            os.environ[key.split("_", 1)[1]] = value
    if args[0].startswith(("/app/", "/usr/local/bin/envoy")):
        args[0] = str(root / pathlib.Path(args[0]).name)
    os.execvp(args[0], args)
else:
    record = {"name": name, "args": args, "env": dict(os.environ)}
    if name == "epp":
        config = next(a.split("=", 1)[1] for a in args if a.startswith("--config-file="))
        record["endpoints"] = json.loads((pathlib.Path(config).parent.parent / "endpoints.json").read_text())
    (root / f"{name}-{os.getpid()}.json").write_text(json.dumps(record))
    def stop(*_):
        (root / f"stopped-{os.getpid()}").touch()
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    while not (root / "fail").exists() or name != "epp":
        time.sleep(0.02)
    sys.exit(17)
'''


@pytest.mark.parametrize("runtime", ["native", "apptainer", "singularity", "pyxis"])
@pytest.mark.parametrize("failure", [False, True])
def test_routed_job_launches_and_stops_all_components(tmp_path, deployment, runtime, failure):
    cluster, spec = deployment
    cluster.slurm.runtime = runtime
    for role in spec.roles:
        role.lws.size = 2
        role.lws.replicas = 2
        role.parallelism.tp = 16
    tool = tmp_path / "tool"
    tool.write_text(f"#!{sys.executable}\n" + FAKE_TOOLS)
    tool.chmod(0o755)
    for name in ("scontrol", "srun", "vllm", "epp", "envoy", "pd-sidecar", "apptainer", "singularity"):
        (tmp_path / name).symlink_to(tool)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", CAPTURE=str(tmp_path),
               TMPDIR=str(tmp_path), SLURM_JOB_ID="42", SLURM_JOB_NODELIST="127.0.0.1,127.0.0.2,127.0.0.3,127.0.0.4",
               SLURM_JOB_NODELIST_HET_GROUP_0="127.0.0.1,127.0.0.2,127.0.0.3,127.0.0.4",
               SLURM_JOB_NODELIST_HET_GROUP_1="127.0.1.1,127.0.1.2,127.0.1.3,127.0.1.4")
    script = tmp_path / "job.sh"
    script.write_text(render(spec, user="test", cluster=cluster))
    process = subprocess.Popen(["bash", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while len(list(tmp_path.glob("*.json"))) < 12 and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.02)
        records = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
        if failure:
            (tmp_path / "fail").touch()
        else:
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert len(records) == 12, (stdout, stderr)
        assert process.returncode == (17 if failure else 143), stderr
        assert len(list(tmp_path.glob("stopped-*"))) == (11 if failure else 12)
        assert not list(tmp_path.glob("manifesto-42.*"))
        workers = [r for r in records if r["name"] == "vllm"]
        for worker in workers:
            env = worker["env"]
            index = int(env["HOSTNAME"].rsplit(".", 1)[1]) - 1
            assert env["LWS_WORKER_INDEX"] == str(index % 2)
            assert env["LWS_LEADER_ADDRESS"] == env["HOSTNAME"].rsplit(".", 1)[0] + "." + str(index // 2 * 2 + 1)
            assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == env["HOSTNAME"]
            assert env["CUDA_VISIBLE_DEVICES"] == f"GPU-{env['HOSTNAME']}"
            assert ("--headless" in worker["args"]) == bool(index % 2)
        proxies = [r for r in records if r["name"] == "pd-sidecar"]
        assert {p["env"]["SLURMD_NODENAME"] for p in proxies} == {"127.0.1.1", "127.0.1.3"}
        endpoints = next(r["endpoints"]["endpoints"] for r in records if r["name"] == "epp")
        assert {e["address"] for e in endpoints} == {"127.0.0.1", "127.0.0.3", "127.0.1.1", "127.0.1.3"}
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


def test_load_aware_replicas_share_one_allocation(deployment):
    cluster, _ = deployment
    spec = load_spec(ROOT / "models/qwen/qwen3-0.6b.yaml", cluster)
    spec.routing.kind = "load_aware"
    spec.roles[0].lws.replicas = 3
    script = render(spec, user="test", cluster=cluster)
    assert "#SBATCH --nodes=3" in script
    assert "#SBATCH --array" not in script
    assert "#SBATCH hetjob" not in script
    assert "--het-group" not in script
    assert '"${SLURM_JOB_NODELIST}"' in script
    assert "/app/epp" in script
    assert "/app/pd-sidecar" not in script


def test_pd_routing_requires_pd_topology(deployment):
    cluster, _ = deployment
    spec = load_spec(ROOT / "models/qwen/qwen3-0.6b.yaml", cluster)
    spec.routing.kind = "pd"
    with pytest.raises(ValueError, match="P/D routing requires topology: pd"):
        render(spec, user="test", cluster=cluster)


@pytest.mark.parametrize("change, message", [
    (lambda s, c: setattr(s.roles[0], "kv_transfer_config", None), "NixlConnector"),
    (lambda s, c: setattr(s.routing, "replicas", 2), "one llm-d router"),
    (lambda s, c: setattr(c.slurm, "port", 8000), "conflict"),
    (lambda s, c: setattr(c.slurm, "port", 9002), "conflict"),
    (lambda s, c: setattr(c.slurm, "router_memory", "1Mi"), "at least 1 MiB each"),
])
def test_unsupported_routed_config_is_rejected(deployment, change, message):
    cluster, spec = deployment
    change(spec, cluster)
    with pytest.raises(ValueError, match=message):
        render(spec, user="test", cluster=cluster)


def test_discovery_is_owned_by_backend(deployment):
    _, spec = deployment
    spec.routing.plugin_config = yaml.safe_load(plugin_configs(spec.routing)["plugins.yaml"])
    spec.routing.plugin_config["dataLayer"] = {"discovery": {"endpoints": {"pluginRef": "kubernetes"}}}
    with pytest.raises(ValueError, match="backend owns endpoint discovery"):
        file_discovery_config(spec.routing, "/tmp/endpoints.json")


def test_heterogeneous_component_can_be_stopped(monkeypatch):
    import argparse
    from manifesto import slurm
    calls = []
    monkeypatch.setattr(slurm, "_capture", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(slurm, "_client_cluster", lambda args: None)
    slurm.stop(argparse.Namespace(job_id="42+1"))
    assert calls == [["scancel", "42+1"]]
