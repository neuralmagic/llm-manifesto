"""Slurm schema, offline workflows, and execution of generated scripts."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from manifesto import slurm, workflow
from manifesto.cli import main
from manifesto.cluster import Cluster, load_cluster
from manifesto.instance import Instance
from manifesto.parallelism import parallel_layout
from manifesto.resolve import resolve_role
from manifesto.slurm_config import SlurmConfig
from manifesto.spec import DeploymentSpec, load_spec


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "clusters/example-slurm.yaml"
MODEL = ROOT / "models/qwen/slurm.yaml"


@pytest.fixture
def cluster():
    value = load_cluster(CLUSTER)
    value.paths.cache_root = None
    value.paths.log_root = None
    return value


@pytest.fixture
def spec(cluster):
    return load_spec(MODEL, cluster)


@pytest.mark.parametrize(("quantity", "expected"), [("8", 8), ("1500m", 2), ("0.25", 1)])
def test_cpu_requests_round_up(quantity, expected):
    assert slurm.cpu_count(quantity) == expected


@pytest.mark.parametrize(("quantity", "expected"), [("32Gi", 32768), ("1.5Gi", 1536), ("1G", 954), ("1048577", 2)])
def test_memory_requests_round_up_in_mib(quantity, expected):
    assert slurm.memory_mib(quantity) == expected


@pytest.mark.parametrize("value", ["0", "-1", "1;id", "nan", ""])
def test_invalid_resource_quantities_fail(value):
    with pytest.raises(ValueError):
        slurm.cpu_count(value)
    with pytest.raises(ValueError):
        slurm.memory_mib(value)


def test_slurm_allocation_must_match_platform():
    data = yaml.safe_load(CLUSTER.read_text())
    data["accelerators"]["profiles"]["b200"]["allocation"] = {
        "extended_resource": {"resource_name": "nvidia.com/gpu"}
    }
    data["accelerators"]["profiles"]["b200"]["presence_label"] = "gpu.present"
    with pytest.raises(ValueError, match="allocation.slurm"):
        Cluster.model_validate(data)
    data = yaml.safe_load(CLUSTER.read_text())
    data["platform"] = "kubernetes"
    with pytest.raises(ValueError, match="set together"):
        Cluster.model_validate(data)


@pytest.mark.parametrize("field", ["partition", "account", "qos", "constraint", "time"])
def test_directives_cannot_inject_lines_or_extra_options(field):
    for value in ("gpu\n#SBATCH --nodes=100", "gpu --nodes=100", "$(id)", ""):
        with pytest.raises(ValueError):
            SlurmConfig.model_validate({field: value})


def test_kubernetes_cluster_settings_are_rejected():
    data = yaml.safe_load(CLUSTER.read_text())
    data["kueue"] = {"local_queue": "gpu"}
    with pytest.raises(ValueError, match="Kubernetes-only"):
        Cluster.model_validate(data)


@pytest.mark.parametrize("update", [
    {"routing": {"kind": "load_aware"}},
    {"runtime": {"sidecars": ["dcgm-exporter"]}},
    {"runtime": {"idle_shutdown": {"enabled": True}}},
    {"roles": [{"name": "decode", "workload": "deployment"}]},
    {"roles": [{"name": "decode", "resources": {"ephemeral_storage": "1Gi"}}]},
    {"roles": [{"name": "decode", "kv_transfer_config": {"kv_connector": "NixlConnector"}}]},
    {"roles": []},
])
def test_unsupported_serving_features_fail(cluster, update):
    data = yaml.safe_load(MODEL.read_text())
    data["model"] = {"id": "test/model", "image": "test/image:v1"}
    data.update(update)
    with pytest.raises(ValueError):
        slurm.render_slurm(DeploymentSpec.model_validate(data), user="tester", cluster=cluster)


def test_default_kubernetes_controllers_do_not_block_direct_serving(cluster):
    data = yaml.safe_load(MODEL.read_text())
    data.pop("runtime")
    data["model"] = {"id": "test/model", "image": "test/image:v1"}
    script = slurm.render_slurm(DeploymentSpec.model_validate(data), user="tester", cluster=cluster)
    assert "#SBATCH --gres=gpu:1" in script


def test_arrays_and_multinode_resources(cluster, spec):
    role = spec.roles[0]
    role.lws.size = 2
    role.lws.replicas = 3
    role.parallelism.tp = 16
    role.resources.cpu = "1500m"
    script = slurm.render_slurm(spec, user="tester", cluster=cluster)
    for directive in ("--nodes=2", "--ntasks=2", "--gres=gpu:8", "--array=0-2", "--cpus-per-task=2", "--mem=32768M"):
        assert f"#SBATCH {directive}" in script
    assert "--node-rank $LWS_WORKER_INDEX" in script
    assert "--master-addr" in script
    assert "--kill-on-bad-exit=1" in script
    assert "--headless" in script


@pytest.mark.parametrize(("tp", "pp", "dp", "nodes"), [(1, 1, 8, 2), (2, 2, False, 2), (8, 2, 2, 4)])
def test_parallel_layouts_reuse_resolved_launch(cluster, spec, tp, pp, dp, nodes):
    role = spec.roles[0]
    role.parallelism.tp, role.parallelism.pp, role.parallelism.dp = tp, pp, dp
    role.lws.size = nodes
    script = slurm.render_slurm(spec, user="tester", cluster=cluster)
    assert f"#SBATCH --gres=gpu:{role.gpus_per_pod}" in script
    assert f"--tensor-parallel-size {tp}" in script
    if pp > 1:
        assert f"--pipeline-parallel-size {pp}" in script
    if dp:
        assert f"DP_SIZE={dp}" in script
    assert parallel_layout(role).dp_world_size == (dp or 1)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_slurm_cache_does_not_use_kubernetes_emptydir(spec):
    cluster = load_cluster(CLUSTER)
    resolved = resolve_role(spec, Instance("tester", spec.release), cluster, spec.roles[0])
    assert resolved.env["VLLM_CACHE_ROOT"].startswith("/shared/tester/")
    script = slurm.render_slurm(spec, user="tester", cluster=cluster)
    assert 'export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}/${MANIFESTO_POD_UID}"' in script
    assert "/var/cache/manifesto-pod" not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_vllm_environment_must_be_visible_in_container(cluster, spec):
    spec.runtime.vllm_env = "/unmounted/env"
    with pytest.raises(ValueError, match="slurm.binds"):
        slurm.render_slurm(spec, user="tester", cluster=cluster)
    spec.runtime.vllm_env = "/shared/env"
    assert "source" in slurm.render_slurm(spec, user="tester", cluster=cluster)


def _executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


@pytest.mark.parametrize("runtime", ["native", "apptainer", "singularity", "pyxis"])
def test_generated_script_executes_with_correct_ranks_and_literal_env(tmp_path, cluster, spec, runtime):
    """Execute both ranks with fake scheduler/container clients and a fake vLLM."""
    cluster.slurm.runtime = runtime
    spec.roles[0].lws.size = 2
    spec.roles[0].parallelism.tp = 16
    spec.roles[0].env["LITERAL"] = "spaces 'quotes' $(touch SHOULD_NOT_EXIST) $HOME"
    spec.model.revision = "pinned-revision"
    spec.roles[0].computed = {"vllm": {"max_num_seqs": "tp * 2"}}
    _executable(tmp_path / "scontrol", 'print("node01\\nnode02")\n')
    _executable(tmp_path / "srun", '''import os, subprocess, sys
command = sys.argv[sys.argv.index("bash"):]
for rank in range(2):
    env = dict(os.environ, SLURM_PROCID=str(rank), SLURMD_NODENAME=f"node0{rank+1}")
    subprocess.run(command, env=env, check=True)
''')
    _executable(tmp_path / runtime, '''import os, subprocess, sys
for key, value in list(os.environ.items()):
    for prefix in ("APPTAINERENV_", "SINGULARITYENV_"):
        if key.startswith(prefix):
            os.environ[key[len(prefix):]] = value
subprocess.run(sys.argv[sys.argv.index("bash"):], check=True)
''')
    _executable(tmp_path / "vllm", '''import json, os, sys
with open(os.environ["CAPTURE"], "a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:], "env": dict(os.environ)}) + "\\n")
''')
    capture = tmp_path / "capture.jsonl"
    script = slurm.render_slurm(spec, user="tester", cluster=cluster)
    env = dict(PATH=f"{tmp_path}:{os.environ['PATH']}", CAPTURE=str(capture),
               SLURM_JOB_NODELIST="node[01-02]", SLURM_JOB_ID="42", SLURM_ARRAY_TASK_ID="2",
               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", HF_TOKEN="test-token-not-in-artifact")
    assert env["HF_TOKEN"] not in script
    result = subprocess.run(["bash"], input=script, text=True, env=env, cwd=tmp_path, capture_output=True)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(records) == 2
    for rank, record in enumerate(records):
        args, actual = record["args"], record["env"]
        assert args[args.index("--node-rank") + 1] == str(rank)
        assert args[args.index("--master-addr") + 1] == "node01"
        assert args[args.index("--revision") + 1] == "pinned-revision"
        assert args[args.index("--max-num-seqs") + 1] == "32"
        assert ("--headless" in args) == (rank == 1)
        assert actual["LITERAL"] == spec.roles[0].env["LITERAL"]
        assert actual["MANIFESTO_POD_UID"] == "42-2"
        assert actual["HF_TOKEN"] == env["HF_TOKEN"]
        assert actual["CUDA_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"]
        assert "--device-ids" not in args
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


@pytest.fixture
def offline(monkeypatch):
    for key in ("MANIFESTO_CLUSTER", "MANIFESTO_NAMESPACE", "MANIFESTO_ROUTING_PROFILE", "MANIFESTO_RENDER_OUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(workflow, "load_dotenv", lambda: None)
    monkeypatch.setattr(workflow, "capture", lambda *a, **k: pytest.fail("unexpected kubectl read"))
    monkeypatch.setattr(workflow, "run", lambda *a, **k: pytest.fail("unexpected kubectl write"))


def test_render_and_validate_cli_are_offline(offline, capsys, tmp_path):
    args = [str(MODEL), "--cluster", str(CLUSTER), "--user", "tester"]
    assert main(["render", "manifest", *args]) == 0
    script = capsys.readouterr().out
    assert script.startswith("#!/bin/bash\n#SBATCH")
    assert "# Generated by:" in script
    path = tmp_path / "job.sbatch"
    assert main(["render", "slurm", *args, "-o", str(path)]) == 0
    assert path.read_text() == script
    assert main(["config", "validate", *args]) == 0
    assert "1 Slurm batch script" in capsys.readouterr().out
    assert main(["explain", *args]) == 0
    assert "workload: slurm" in capsys.readouterr().out


def test_slurm_env_default_selects_script_output(offline, monkeypatch):
    monkeypatch.setenv("MANIFESTO_CLUSTER", str(CLUSTER))
    config = workflow.RuntimeConfig.from_args(argparse.Namespace())
    assert config.render_out == Path("/tmp/manifesto.sbatch")
    assert config.platform == "slurm"


@pytest.mark.parametrize("command", [["deploy"], ["slurm", "submit"]])
def test_submit_uses_stdin_and_does_not_require_hf_secret(offline, monkeypatch, capsys, command):
    calls = []
    monkeypatch.setattr(workflow, "require_hf_token", lambda: pytest.fail("Kubernetes secret"))
    def capture(cmd, *, script=None, **kwargs):
        calls.append((cmd, script))
        return "1234"
    monkeypatch.setattr(slurm, "_capture", capture)
    assert main([*command, str(MODEL), "--cluster", str(CLUSTER)]) == 0
    assert calls[0][0] == ["sbatch", "--parsable"]
    assert calls[0][1].startswith("#!/bin/bash")
    assert capsys.readouterr().out.strip() == "1234"


def test_failed_submission_surfaces_error_without_retry(monkeypatch):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "invalid partition")
    monkeypatch.setattr(slurm.subprocess, "run", run)
    with pytest.raises(workflow.WorkflowError, match="invalid partition"):
        slurm.submit("#!/bin/bash\n")
    assert len(calls) == 1


def test_servers_and_stop_use_slurm_clients(monkeypatch, capsys):
    calls = []
    def capture(cmd, **kwargs):
        calls.append(cmd)
        return "42_0|manifesto-test|RUNNING|node01|None\n43|unrelated|PENDING||Resources"
    monkeypatch.setattr(slurm, "_capture", capture)
    assert main(["slurm", "servers", "--user", "tester", "--output", "json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert [record["job_id"] for record in records] == ["42_0"]
    assert "--user=tester" in calls[0]
    assert main(["slurm", "stop", "42_0"]) == 0
    assert calls[-1] == ["scancel", "42_0"]
    assert main(["slurm", "stop", "42;id"]) == 2
    assert len(calls) == 2


@pytest.mark.parametrize(("image", "expected"), [
    ("vllm/vllm-openai:latest", "vllm/vllm-openai:latest"),
    ("docker://nvcr.io/nvidia/vllm:26.09", "nvcr.io#nvidia/vllm:26.09"),
    ("registry.example:5000/team/model:v1", "registry.example:5000#team/model:v1"),
    ("/shared/model.sqsh", "/shared/model.sqsh"),
])
def test_pyxis_images_and_mounts(cluster, spec, image, expected):
    cluster.slurm.runtime = "pyxis"
    spec.model.image = image
    script = slurm.render_slurm(spec, user="tester", cluster=cluster)
    assert f"--container-image={expected}" in script
    assert "--container-mounts=/shared:/shared:rw" in script
    assert "--no-container-entrypoint" in script


def test_remote_clients_quote_arguments_and_send_script_only_on_stdin(monkeypatch, cluster):
    cluster.slurm.ssh_host = "user@login.example"
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "123", "")
    monkeypatch.setattr(slurm.subprocess, "run", run)
    slurm.submit("#!/bin/bash\necho test\n", cluster=cluster, test_only=True)
    command, kwargs = calls[0]
    assert command == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                       "--", "user@login.example", "sbatch --test-only"]
    assert kwargs["input"] == "#!/bin/bash\necho test\n"
    assert kwargs["timeout"] == 60
    slurm._capture(["squeue", "--format=%i|%j"], cluster=cluster)
    assert calls[1][0][-1] == "squeue '--format=%i|%j'"


def test_submit_timeout_is_not_retried(monkeypatch):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, 60)
    monkeypatch.setattr(slurm.subprocess, "run", run)
    with pytest.raises(workflow.WorkflowError, match="check Slurm before retrying"):
        slurm.submit("#!/bin/bash\n")
    assert len(calls) == 1


def test_dp_slices_assigned_gpu_ids(cluster, spec):
    spec.roles[0].parallelism.dp = 2
    spec.roles[0].parallelism.tp = 2
    script = slurm.render_slurm(spec, user="tester", cluster=cluster)
    assert "--device-ids" not in script
    assert 'CUDA_VISIBLE_DEVICES="$GPUS"' in script
    # Execute the generated selection expression on non-contiguous and UUID IDs.
    expression = next(line for line in script.splitlines() if "GPUS=$(IFS=" in line)
    result = subprocess.run(["bash", "-c", '\n'.join([
        'IFS=, read -r -a MANIFESTO_VISIBLE_GPUS <<< "2,5,GPU-abcd,GPU-efgh"',
        "GPU_START=2", expression, 'echo "$GPUS"',
    ])], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "GPU-abcd,GPU-efgh"


def test_kubernetes_commands_fail_before_remote_mutation(offline, capsys):
    assert main(["render", "routing", str(MODEL), "--cluster", str(CLUSTER)]) == 2
    assert "routing" in capsys.readouterr().err
    assert main(["render", "bootstrap", "--cluster", str(CLUSTER)]) == 2
    assert "bootstrap" in capsys.readouterr().err
    assert main(["render", "slurm", str(MODEL), "--cluster", str(CLUSTER), "--namespace", "test"]) == 2
    assert "--namespace" in capsys.readouterr().err


def test_kubernetes_workload_settings_reject_slurm_allocation():
    from manifesto.workload import WorkloadAccelerator

    with pytest.raises(ValueError, match="cannot use Slurm"):
        WorkloadAccelerator.model_validate({"allocation": {"slurm": {"gres": "gpu"}}})
