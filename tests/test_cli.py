"""CLI regression tests for derived paths, overrides, and routing-only rendering."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from manifesto.cli import _build_parser, main
from manifesto.cluster import load_cluster
from manifesto.overrides import load_routing_profile
from manifesto.render import render
from manifesto.spec import load_spec
import manifesto.workflow as workflow
from manifesto.workflow import (
    MANAGED_RESOURCE_TYPES,
    catalog_entries,
    config_home,
    resolve_cluster,
    resolve_model,
    resolve_namespace,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "deepseek-v4" / "1P-EP8-1D-EP8.yaml"
STANDALONE_MODEL = ROOT / "models" / "qwen" / "aggregated.yaml"
CLUSTER = ROOT / "clusters" / "example-gb200.yaml"


def _bootstrap_cluster(tmp_path):
    profile = yaml.safe_load(CLUSTER.read_text())
    profile["storage"]["shared_claim"] = {
        "storage_class_name": "nfs",
        "access_modes": ["ReadWriteMany"],
        "size": "30Gi",
    }
    profile["storage"]["shared_volume"]["persistentVolumeClaim"]["claimName"] = "example-shared-cache"
    cluster_path = tmp_path / "bootstrap-cluster.yaml"
    cluster_path.write_text(yaml.safe_dump(profile))
    return cluster_path


def _live_object(kind, name, instance, *, api_version="v1", labels=None, ready=False):
    conditions = [{"type": "Ready", "status": "True"}] if ready else []
    return {
        "apiVersion": api_version,
        "kind": kind,
        "metadata": {
            "name": name,
            "creationTimestamp": "2026-07-21T12:00:00Z",
            "labels": {
                "app.kubernetes.io/name": "manifesto",
                "app.kubernetes.io/instance": instance,
                **(labels or {}),
            },
        },
        "status": {"conditions": conditions},
    }


def _mock_discovery(monkeypatch, responses):
    """Serve one canned discovery round per ``list_objects`` call."""

    rounds = iter(responses)

    def fake_list_objects(config, resource_types, selector, **_):
        assert resource_types, "discovery must request at least one resource type"
        return list(next(rounds))

    monkeypatch.setattr(workflow, "list_objects", fake_list_objects)
    monkeypatch.setattr(
        workflow, "capture", lambda cmd, **_: pytest.fail(f"unexpected cluster read: {cmd}")
    )


def test_user_config_catalog_resolves_models_and_clusters(monkeypatch, tmp_path):
    user_config = tmp_path / "manifesto-config"
    model = user_config / "models" / "model_provider" / "model.yaml"
    cluster = user_config / "clusters" / "local.yaml"
    model.parent.mkdir(parents=True)
    cluster.parent.mkdir(parents=True)
    model.write_text("release: local\n")
    cluster.write_text("name: local\n")
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    assert config_home() == user_config
    assert resolve_model("model_provider/model") == str(model)
    assert resolve_cluster("local") == str(cluster)


def test_model_catalog_name_with_dot_resolves_without_yaml_suffix(monkeypatch, tmp_path):
    user_config = tmp_path / "manifesto-config"
    model = user_config / "models" / "qwen" / "qwen3-0.6b.yaml"
    model.parent.mkdir(parents=True)
    model.write_text("release: qwen3-0-6b\n")
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    assert resolve_model("qwen/qwen3-0.6b") == str(model)


def test_yml_catalog_entry_resolves_without_suffix(monkeypatch, tmp_path):
    user_config = tmp_path / "manifesto-config"
    model = user_config / "models" / "local.yml"
    model.parent.mkdir(parents=True)
    model.write_text("release: local\n")
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    assert resolve_model("local") == str(model)


def test_cluster_named_for_current_context_is_selected(monkeypatch, tmp_path):
    user_config = tmp_path / "manifesto-config"
    cluster = user_config / "clusters" / "local-context.yaml"
    cluster.parent.mkdir(parents=True)
    cluster.write_text("name: local\n")
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))
    monkeypatch.delenv("MANIFESTO_CLUSTER", raising=False)
    monkeypatch.delenv("MANIFESTO_CLUSTER_MAP", raising=False)
    monkeypatch.setattr(
        workflow,
        "capture",
        lambda cmd, **_: "local-context\n" if cmd[-1] == "current-context" else "kube-cluster\n",
    )

    assert resolve_cluster() == str(cluster)


def test_explicit_context_selects_its_namespace_and_cluster(monkeypatch, tmp_path):
    user_config = tmp_path / "manifesto-config"
    cluster = user_config / "clusters" / "remote-context.yaml"
    cluster.parent.mkdir(parents=True)
    cluster.write_text("name: remote-context\n")
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))
    monkeypatch.delenv("MANIFESTO_CLUSTER", raising=False)
    monkeypatch.delenv("MANIFESTO_CLUSTER_MAP", raising=False)
    monkeypatch.delenv("MANIFESTO_NAMESPACE", raising=False)

    def fake_capture(cmd, **_):
        assert cmd[:3] == ["kubectl", "--context", "remote-context"]
        if "jsonpath={..namespace}" in cmd:
            return "workload-ns\n"
        return "openshift\n"

    monkeypatch.setattr(workflow, "capture", fake_capture)

    assert resolve_namespace(context="remote-context") == "workload-ns"
    assert resolve_cluster(context="remote-context") == str(cluster)


def test_render_accepts_user_catalog_names(monkeypatch, tmp_path, capsys):
    user_config = tmp_path / "manifesto-config"
    model = user_config / "models" / "local-model.yaml"
    cluster = user_config / "clusters" / "local-cluster.yaml"
    model.parent.mkdir(parents=True)
    cluster.parent.mkdir(parents=True)
    model.write_text(STANDALONE_MODEL.read_text())
    cluster.write_text(CLUSTER.read_text())
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    rc = main(
        [
            "render",
            "manifest",
            "local-model",
            "--cluster",
            "local-cluster",
            "--namespace",
            "test",
            "--user",
            "tester",
        ]
    )

    assert rc == 0
    assert "kind: Deployment" in capsys.readouterr().out


def test_explain_reports_features_without_polluting_rendered_yaml(capsys):
    rc = main(
        [
            "explain",
            str(MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "test",
            "--user",
            "tester",
        ]
    )

    assert rc == 0
    explanation = yaml.safe_load(capsys.readouterr().out)
    decode = next(role for role in explanation["roles"] if role["name"] == "decode")
    assert decode["workload"] == "LeaderWorkerSet"
    assert set(decode["features"]) >= {
        "data-parallel",
        "expert-parallel",
        "llm-d",
        "prefill-decode",
    }
    assert "connector:NixlConnector" in decode["backends"]
    assert decode["environment"]["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "backend:NixlConnector"

    rc = main(
        [
            "render",
            "manifest",
            str(MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "test",
            "--user",
            "tester",
        ]
    )

    assert rc == 0
    rendered = capsys.readouterr().out
    assert "manifesto.features" not in rendered
    assert "backend:NixlConnector" not in rendered


def test_config_catalog_lists_effective_entries_and_shadowing(monkeypatch, tmp_path, capsys):
    user_config = tmp_path / "manifesto-config"
    user_model = user_config / "models" / "qwen" / "aggregated.yaml"
    user_model.parent.mkdir(parents=True)
    user_model.write_text(STANDALONE_MODEL.read_text())
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    entries = {entry.name: entry for entry in catalog_entries("models")}
    assert entries["qwen/aggregated"].path == user_model
    assert entries["qwen/aggregated"].source == "user"
    assert entries["qwen/aggregated"].shadows == ROOT / "models" / "qwen" / "aggregated.yaml"

    assert main(["config", "list", "models", "--output", "json"]) == 0
    output = json.loads(capsys.readouterr().out)
    qwen = next(item for item in output if item["name"] == "qwen/aggregated")
    assert qwen == {
        "name": "qwen/aggregated",
        "source": "user",
        "path": str(user_model),
        "shadows": str(ROOT / "models" / "qwen" / "aggregated.yaml"),
    }


def test_routing_catalog_profile_is_separate_and_user_override_wins(monkeypatch, tmp_path, capsys):
    user_config = tmp_path / "manifesto-config"
    user_profile = user_config / "routing" / "wide-ep-lws-config.yaml"
    user_profile.parent.mkdir(parents=True)
    user_profile.write_text(
        "apiVersion: llm-d.ai/v1alpha1\n"
        "kind: EndpointPickerConfig\n"
        "plugins:\n"
        "  - type: prefix-cache-scorer\n"
        "    name: user-kv-scorer\n"
    )
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    path, config = load_routing_profile("wide-ep-lws-config")
    assert path == user_profile
    assert config["plugins"] == [{"type": "prefix-cache-scorer", "name": "user-kv-scorer"}]

    entries = {entry.name: entry for entry in catalog_entries("routing")}
    assert entries["wide-ep-lws-config"].path == user_profile
    assert entries["wide-ep-lws-config"].source == "user"
    assert entries["wide-ep-lws-config"].shadows == ROOT / "routing" / "wide-ep-lws-config.yaml"
    assert main(["config", "resolve", "routing", "wide-ep-lws-config"]) == 0
    assert capsys.readouterr().out.strip() == str(user_profile.resolve())


def test_render_selects_routing_profile_outside_model(capsys):
    assert main(
        [
            "render",
            "routing",
            str(MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "test",
            "--user",
            "tester",
            "--routing-profile",
            "wide-ep-lws-config",
        ]
    ) == 0
    rendered = capsys.readouterr().out
    assert "--routing-profile wide-ep-lws-config" in rendered
    assert "wide-ep-lws-config.yaml: |" in rendered
    assert "--config-file=/etc/epp/wide-ep-lws-config.yaml" in rendered


def test_config_resolve_reports_unknown_name(capsys):
    assert main(["config", "resolve", "models", "does-not-exist"]) == 2
    assert capsys.readouterr().err == (
        "Unknown model config 'does-not-exist'. "
        "Run 'manifesto config list models' to see available names.\n"
    )


def test_config_validate_exercises_model_and_cluster(capsys):
    assert main(
        [
            "config",
            "validate",
            "qwen/aggregated",
            "--cluster",
            "example-gb200",
            "--user",
            "tester",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert f"Valid cluster: {CLUSTER}" in output
    assert f"Valid model:   {STANDALONE_MODEL}" in output
    assert "Kubernetes objects" in output


def test_config_edit_copies_bundled_model_and_extends_parent(monkeypatch, tmp_path, capsys):
    user_config = tmp_path / "manifesto-config"
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))
    monkeypatch.setenv("EDITOR", "test-editor --wait")
    calls = []
    monkeypatch.setattr(
        "manifesto.cli.subprocess.run",
        lambda cmd: calls.append(cmd) or SimpleNamespace(returncode=0),
    )

    assert main(["config", "edit", "models", "deepseek-v4/1P-EP8-1D-EP8"]) == 0

    destination = user_config / "models" / "deepseek-v4" / "1P-EP8-1D-EP8.yaml"
    assert calls == [["test-editor", "--wait", str(destination)]]
    assert destination.read_text() == MODEL.read_text()
    assert (destination.parent / "wide-ep-base.yaml").is_file()
    assert load_spec(destination).release == "wide-ep-1p-ep8-1d-ep8"
    assert "Created user config from bundled entry" in capsys.readouterr().out


def test_config_edit_creates_new_model_and_validates_after_editor(monkeypatch, tmp_path):
    user_config = tmp_path / "manifesto-config"
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    def fake_editor(cmd):
        Path(cmd[-1]).write_text(STANDALONE_MODEL.read_text())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("manifesto.cli.subprocess.run", fake_editor)

    assert main(["config", "edit", "models", "experiments/new-model"]) == 0
    assert (user_config / "models" / "experiments" / "new-model.yaml").is_file()


def test_config_export_and_import_flatten_model_inheritance(monkeypatch, tmp_path, capsys):
    assert main(["config", "export", "models", "deepseek-v4/1P-EP8-1D-EP8"]) == 0
    exported = capsys.readouterr().out
    data = yaml.safe_load(exported)
    assert "extends" not in data
    assert isinstance(data["roles"], list)

    portable = tmp_path / "portable.yaml"
    portable.write_text(exported)
    user_config = tmp_path / "manifesto-config"
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    assert main(
        ["config", "import", "models", str(portable), "--name", "experiments/imported"]
    ) == 0
    destination = user_config / "models" / "experiments" / "imported.yaml"
    assert capsys.readouterr().out.strip() == str(destination)
    assert "extends" not in yaml.safe_load(destination.read_text())
    assert load_spec(destination).release == "wide-ep-1p-ep8-1d-ep8"


def test_config_import_refuses_to_overwrite(monkeypatch, tmp_path, capsys):
    user_config = tmp_path / "manifesto-config"
    destination = user_config / "models" / "existing.yaml"
    destination.parent.mkdir(parents=True)
    destination.write_text(STANDALONE_MODEL.read_text())
    source = tmp_path / "source.yaml"
    source.write_text(STANDALONE_MODEL.read_text())
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    assert main(["config", "import", "models", str(source), "--name", "existing"]) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_config_export_and_import_cluster(monkeypatch, tmp_path, capsys):
    assert main(["config", "export", "clusters", "example-gb200"]) == 0
    exported = capsys.readouterr().out
    source = tmp_path / "cluster.yaml"
    source.write_text(exported)
    user_config = tmp_path / "manifesto-config"
    monkeypatch.setenv("MANIFESTO_CONFIG_HOME", str(user_config))

    assert main(["config", "import", "clusters", str(source), "--name", "workload"]) == 0
    destination = user_config / "clusters" / "workload.yaml"
    assert load_cluster(destination).name == "example-gb200"


def test_top_level_help_has_compact_command_surface():
    help_text = _build_parser().format_help()

    assert "{render,explain,file,deploy,servers,stop,ready,test,completion,config}" in help_text
    for removed in (
        "cache-path",
        "dev-path",
        "edit-file",
        "ensure-hf-secret",
        "instance-id",
        "log-path",
        "render-dev-pod",
        "render-routing",
    ):
        assert removed not in help_text


@pytest.mark.parametrize(
    "command",
    [
        "apply",
        "bootstrap",
        "cache-path",
        "delete",
        "dev",
        "deploy-routing",
        "diff",
        "edit-file",
        "ensure-hf-secret",
        "instance-id",
        "log-path",
        "name",
        "render-bootstrap",
        "render-dev-pod",
        "render-file",
        "render-routing",
    ],
)
def test_removed_top_level_commands_are_rejected(command):
    with pytest.raises(SystemExit, match="2"):
        main([command])


def test_render_cli_vllm_env_override(monkeypatch, capsys):
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    rc = main(
        [
            "render",
            "manifest",
            str(MODEL),
            "--cluster",
            str(CLUSTER),
            "--user",
            "tester",
            "--vllm-env",
            "/mnt/shared/custom-vllm",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "name: MANIFESTO_VLLM_ENV\n            value: /mnt/shared/custom-vllm" in out
    assert 'source "${MANIFESTO_VLLM_ENV}/.venv/bin/activate"' in out


@pytest.mark.parametrize("legacy_args", [["--dev"], ["--dev-venv", "/custom/venv"]])
def test_removed_dev_render_flags_are_rejected(legacy_args):
    with pytest.raises(SystemExit, match="2"):
        main(["render", "manifest", str(MODEL), *legacy_args])


def test_render_cli_prefixes_manifest_with_generation_command(monkeypatch, capsys):
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    rc = main(
        [
            "render",
            "manifest",
            str(MODEL),
            "--cluster",
            str(CLUSTER),
            "--user",
            "tester",
            "--vllm-env",
            "/mnt/shared/tester/vllm-envs/feature",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("# Generated by:\n")
    assert f"#   manifesto render manifest {MODEL} --cluster {CLUSTER} --namespace workload-ns --user tester --vllm-env /mnt/shared/tester/vllm-envs/feature\n" in out
    assert "# Source: https://github.com/neuralmagic/llm-manifesto\n" in out
    assert "# Safe to edit before applying.\n" in out
    assert "comments are ignored by kubectl" not in out
    assert "kind: LeaderWorkerSet" in out


def test_render_cli_namespace_override(capsys):
    rc = main(
        [
            "render",
            "routing",
            str(MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "workload-ns",
            "--user",
            "tester",
        ]
    )

    assert rc == 0
    assert "--pool-namespace=workload-ns" in capsys.readouterr().out


def test_render_cli_accelerator_override_changes_cache_architecture(capsys):
    rc = main(
        [
            "render",
            "manifest",
            str(STANDALONE_MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "workload-ns",
            "--user",
            "tester",
            "--gpu",
            "b200",
        ]
    )

    assert rc == 0
    rendered = capsys.readouterr().out
    assert "--gpu b200" in rendered
    assert "/jit-cache/b200/cu13/" in rendered


def test_accelerator_cli_alias_remains_supported(capsys):
    rc = main(
        [
            "render",
            "manifest",
            str(STANDALONE_MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "workload-ns",
            "--user",
            "tester",
            "--accelerator",
            "b200",
        ]
    )

    assert rc == 0
    assert "/jit-cache/b200/cu13/" in capsys.readouterr().out


def test_invalid_cluster_profile_prints_concise_validation_error(tmp_path, capsys):
    profile = yaml.safe_load(CLUSTER.read_text())
    del profile["accelerators"]
    cluster_path = tmp_path / "invalid-cluster.yaml"
    cluster_path.write_text(yaml.safe_dump(profile))

    rc = main(
        [
            "render",
            "manifest",
            str(STANDALONE_MODEL),
            "--cluster",
            str(cluster_path),
            "--user",
            "tester",
        ]
    )

    assert rc == 2
    assert capsys.readouterr().err == (
        "Invalid configuration:\n"
        "  accelerators: Field required\n"
    )


def test_unknown_gpu_prints_concise_error(capsys):
    rc = main(
        [
            "render",
            "manifest",
            str(STANDALONE_MODEL),
            "--cluster",
            str(CLUSTER),
            "--user",
            "tester",
            "--gpu",
            "quantum-gpu",
        ]
    )

    assert rc == 2
    assert capsys.readouterr().err == (
        "Invalid configuration: unknown accelerator 'quantum-gpu' for this cluster; "
        "choose one of: b200, gb200\n"
    )


def test_render_cli_pre_launch_hook(monkeypatch, capsys):
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    rc = main(
        [
            "render",
            "manifest",
            str(MODEL),
            "--cluster",
            str(CLUSTER),
            "--user",
            "tester",
            "--pre-launch",
            "echo cli-hook",
        ]
    )

    assert rc == 0
    assert "echo cli-hook" in capsys.readouterr().out


def test_render_cli_overrides_idle_shutdown(capsys):
    rc = main(
        [
            "render",
            "manifest",
            str(STANDALONE_MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "workload-ns",
            "--user",
            "tester",
            "--idle-timeout",
            "2m",
        ]
    )

    assert rc == 0
    rendered = capsys.readouterr().out
    assert "--idle-timeout 2m" in rendered
    documents = [doc for doc in yaml.safe_load_all(rendered) if doc]
    controller = next(
        doc
        for doc in documents
        if doc["kind"] == "Deployment"
        and doc["metadata"]["name"].endswith("idle-shutdown")
    )
    env = {
        item["name"]: item
        for item in controller["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["TIMEOUT_SECONDS"]["value"] == "120"


def test_render_cli_can_disable_idle_shutdown(capsys):
    rc = main(
        [
            "render",
            "manifest",
            str(STANDALONE_MODEL),
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "workload-ns",
            "--user",
            "tester",
            "--no-idle-shutdown",
        ]
    )

    assert rc == 0
    rendered = capsys.readouterr().out
    assert "--no-idle-shutdown" in rendered
    assert "app.kubernetes.io/component: idle-shutdown" not in rendered


def test_render_file_uses_env_defaults(tmp_path, monkeypatch, capsys):
    output = tmp_path / "manifest.yaml"
    monkeypatch.setenv("MANIFESTO_CLUSTER", str(CLUSTER))
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.setenv("USER", "tester")

    rc = main([
        "render", "file", str(MODEL), "--output", str(output),
        "--vllm-env", "/mnt/shared/tester/vllm-envs/feature",
    ])

    assert rc == 0
    assert capsys.readouterr().out.strip() == str(output)
    rendered = output.read_text()
    assert f"#   manifesto render manifest {MODEL} --cluster {CLUSTER} --namespace workload-ns --user tester --vllm-env /mnt/shared/tester/vllm-envs/feature\n" in rendered
    assert "namespace: workload-ns" in rendered


def test_deploy_pipes_rendered_manifest_to_kubectl(monkeypatch):
    calls = []

    def fake_run(cmd, *, input_text=None):
        calls.append((cmd, input_text))
        return 0

    monkeypatch.setenv("MANIFESTO_CLUSTER", str(CLUSTER))
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setenv("HF_TOKEN", "hf_read_test")
    monkeypatch.setattr(workflow, "run", fake_run)

    rc = main(["deploy", str(MODEL), "--vllm-env", "/mnt/shared/tester/vllm-envs/feature"])

    assert rc == 0
    assert calls[0][0] == ["kubectl", "-n", "workload-ns", "apply", "-f", "-"]
    secret = json.loads(calls[0][1])
    assert secret["metadata"] == {"name": "hf-secret", "namespace": "workload-ns"}
    assert secret["stringData"] == {"HF_TOKEN": "hf_read_test"}
    assert calls[1][0] == ["kubectl", "-n", "workload-ns", "apply", "-f", "-"]
    assert "kind: LeaderWorkerSet" in calls[1][1]
    assert "hf_read_test" not in calls[1][1]
    assert "--vllm-env /mnt/shared/tester/vllm-envs/feature" in calls[1][1]


def test_deploy_honors_context_and_idle_timeout(monkeypatch):
    calls = []

    def fake_run(cmd, *, input_text=None):
        calls.append((cmd, input_text))
        return 0

    monkeypatch.setenv("MANIFESTO_CLUSTER", str(CLUSTER))
    monkeypatch.setenv("HF_TOKEN", "hf_read_test")
    monkeypatch.setattr(workflow, "run", fake_run)

    rc = main(
        [
            "deploy",
            str(STANDALONE_MODEL),
            "--context",
            "remote-context",
            "--namespace",
            "workload-ns",
            "--user",
            "tester",
            "--idle-timeout",
            "2m",
        ]
    )

    assert rc == 0
    assert [call[0] for call in calls] == [
        ["kubectl", "--context", "remote-context", "-n", "workload-ns", "apply", "-f", "-"],
        ["kubectl", "--context", "remote-context", "-n", "workload-ns", "apply", "-f", "-"],
    ]
    assert "--idle-timeout 2m" in calls[1][1]
    assert "value: '120'" in calls[1][1]


def test_deploy_requires_hf_token_before_applying(monkeypatch, capsys):
    monkeypatch.setenv("MANIFESTO_CLUSTER", str(CLUSTER))
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(workflow, "load_dotenv", lambda: None)
    monkeypatch.setattr(workflow, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))

    rc = main(["deploy", "manifest", str(MODEL), "--user", "tester"])

    assert rc == 2
    assert "HF_TOKEN is not configured" in capsys.readouterr().err


def test_deploy_manifest_remains_a_compatibility_alias(monkeypatch, capsys):
    monkeypatch.setenv("MANIFESTO_CLUSTER", str(CLUSTER))
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(workflow, "load_dotenv", lambda: None)

    assert main(["deploy", "manifest", str(MODEL), "--user", "tester"]) == 2
    assert "HF_TOKEN is not configured" in capsys.readouterr().err


def test_deploy_routing_applies_without_syncing_hf_secret(monkeypatch):
    calls = []
    monkeypatch.setenv("MANIFESTO_CLUSTER", str(CLUSTER))
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.setattr(
        workflow,
        "run",
        lambda cmd, *, input_text=None: calls.append((cmd, input_text)) or 0,
    )

    rc = main(["deploy", "routing", str(MODEL), "--user", "tester"])

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][0] == ["kubectl", "-n", "workload-ns", "apply", "-f", "-"]
    assert "kind: HTTPRoute" in calls[0][1]
    assert "kind: LeaderWorkerSet" not in calls[0][1]


def test_bootstrap_applies_profile_namespace_prerequisites(monkeypatch, tmp_path):
    calls = []
    cluster_path = _bootstrap_cluster(tmp_path)
    monkeypatch.setattr(
        workflow,
        "run",
        lambda cmd, *, input_text=None: calls.append((cmd, input_text)) or 0,
    )

    rc = main(
        [
            "deploy",
            "bootstrap",
            "--cluster",
            str(cluster_path),
            "--namespace",
            "workload-ns",
        ]
    )

    assert rc == 0
    assert calls[0][0] == ["kubectl", "-n", "workload-ns", "apply", "-f", "-"]
    claim = yaml.safe_load(calls[0][1])
    assert claim == {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": "example-shared-cache",
            "namespace": "workload-ns",
            "labels": {"app.kubernetes.io/name": "manifesto"},
        },
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "resources": {"requests": {"storage": "30Gi"}},
            "storageClassName": "nfs",
        },
    }


def test_render_bootstrap_prints_same_manifest_without_applying(monkeypatch, tmp_path, capsys):
    cluster_path = _bootstrap_cluster(tmp_path)
    monkeypatch.setattr(
        workflow,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not apply")),
    )

    rc = main(
        [
            "render",
            "bootstrap",
            "--cluster",
            str(cluster_path),
            "--namespace",
            "workload-ns",
        ]
    )

    assert rc == 0
    claim = yaml.safe_load(capsys.readouterr().out)
    assert claim["kind"] == "PersistentVolumeClaim"
    assert claim["metadata"] == {
        "name": "example-shared-cache",
        "namespace": "workload-ns",
        "labels": {"app.kubernetes.io/name": "manifesto"},
    }


def test_bootstrap_rejects_profile_without_prerequisites(monkeypatch, capsys):
    monkeypatch.setattr(
        workflow,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    rc = main(
        [
            "deploy",
            "bootstrap",
            "--cluster",
            str(CLUSTER),
            "--namespace",
            "workload-ns",
        ]
    )

    assert rc == 2
    assert "declares no bootstrap resources" in capsys.readouterr().err


def test_ready_waits_for_spec_roles_only(monkeypatch, tmp_path):
    spec_file = tmp_path / "aggregated.yaml"
    spec_file.write_text(
        "release: smoke\n"
        "topology: aggregated\n"
        "model: {id: model, image: image}\n"
        "routing: {kind: disabled}\n"
        "roles:\n"
        "  - name: decode\n"
        "    lws: {size: 1}\n"
        "    parallelism: {tp: 1, dp: false, ep: false}\n"
    )
    popen_cmds = []

    class FakeProc:
        def wait(self):
            return 0

    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.setattr(workflow.subprocess, "Popen", lambda cmd: popen_cmds.append(cmd) or FakeProc())

    rc = main(["ready", str(spec_file), "--user", "tester"])

    assert rc == 0
    joined = [" ".join(cmd) for cmd in popen_cmds]
    assert any("llm-d.ai/role=decode" in cmd for cmd in joined)
    assert not any("llm-d.ai/role=prefill" in cmd for cmd in joined)
    # Routing is disabled, so no endpoint picker wait is issued.
    assert not any("deploy/" in cmd for cmd in joined)


def test_ready_uses_gateway_class_from_cluster(monkeypatch):
    popen_cmds = []
    capture_cmds = []

    class FakeProc:
        def wait(self):
            return 0

    cluster = load_cluster(CLUSTER)
    cluster.gateway.class_name = "platform-gateway"
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.setattr(workflow, "load_cluster_with_overrides", lambda *_: cluster)
    monkeypatch.setattr(workflow.subprocess, "Popen", lambda cmd: popen_cmds.append(cmd) or FakeProc())
    monkeypatch.setattr(
        workflow,
        "capture",
        lambda cmd, **_: capture_cmds.append(cmd) or '{"data":[{"id":"model"}]}',
    )

    rc = main(["ready", str(MODEL), "--cluster", "cluster.yaml", "--user", "tester"])

    assert rc == 0
    assert popen_cmds
    gateway_url = capture_cmds[0][-1]
    assert gateway_url.endswith("-platform-gateway:80/v1/models")
    assert len(gateway_url.removeprefix("http://").removesuffix(":80/v1/models")) <= 63


def test_apply_file_does_not_require_cluster(monkeypatch, tmp_path):
    output = tmp_path / "manifest.yaml"
    calls = []

    def fake_run(cmd, *, input_text=None):
        calls.append((cmd, input_text))
        return 0

    monkeypatch.delenv("MANIFESTO_CLUSTER", raising=False)
    monkeypatch.delenv("MANIFESTO_CLUSTER_MAP", raising=False)
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setenv("HF_TOKEN", "hf_read_test")
    monkeypatch.setattr(workflow, "run", fake_run)

    rc = main(["file", "apply", "--output", str(output)])

    assert rc == 0
    assert json.loads(calls[0][1])["stringData"] == {"HF_TOKEN": "hf_read_test"}
    assert calls[1] == (["kubectl", "-n", "workload-ns", "apply", "-f", str(output)], None)


def test_file_diff_and_delete_target_saved_manifest(monkeypatch, tmp_path):
    output = tmp_path / "manifest.yaml"
    calls = []
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.setattr(
        workflow,
        "run",
        lambda cmd, *, input_text=None: calls.append(cmd) or 0,
    )

    assert main(["file", "diff", "--output", str(output)]) == 0
    assert main(["file", "delete", "--output", str(output), "--now"]) == 0
    assert calls == [
        ["kubectl", "-n", "workload-ns", "diff", "-f", str(output)],
        [
            "kubectl",
            "-n",
            "workload-ns",
            "delete",
            "-f",
            str(output),
            "--ignore-not-found=true",
            "--grace-period=0",
            "--force",
        ],
    ]


def test_servers_lists_all_instances_in_namespace(monkeypatch, capsys):
    objects = [
        _live_object(
            "Deployment",
            "alice-model",
            "alice-qwen",
            api_version="apps/v1",
            labels={"llm-d.ai/model": "qwen", "llm-d.ai/role": "decode"},
        ),
        _live_object(
            "Pod",
            "alice-model-1",
            "alice-qwen",
            labels={"llm-d.ai/model": "qwen", "llm-d.ai/role": "decode"},
            ready=True,
        ),
        _live_object(
            "Deployment",
            "bob-model",
            "bob-deepseek",
            api_version="apps/v1",
            labels={"llm-d.ai/model": "deepseek", "llm-d.ai/role": "prefill"},
        ),
    ]
    _mock_discovery(monkeypatch, [objects])

    rc = main(["servers", "--namespace", "workload-ns"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "alice-qwen" in out and "Ready" in out and "1/1" in out
    assert "bob-deepseek" in out and "Pending" in out


def test_servers_json_exposes_exact_resources(monkeypatch, capsys):
    objects = [
        _live_object("Service", "alice-route", "alice-qwen"),
        _live_object("Pod", "alice-model-1", "alice-qwen", ready=True),
    ]
    _mock_discovery(monkeypatch, [objects])

    rc = main(["servers", "--namespace", "workload-ns", "--instance", "alice-qwen", "--output", "json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["instance"] == "alice-qwen"
    assert payload[0]["pods"] == {"ready": 1, "total": 1}
    assert payload[0]["resources"] == [
        {"apiVersion": "v1", "kind": "Pod", "name": "alice-model-1"},
        {"apiVersion": "v1", "kind": "Service", "name": "alice-route"},
    ]


def test_stop_spec_discovers_live_state_without_cluster_profile(monkeypatch, capsys):
    objects = [
        _live_object(
            "Deployment",
            "tester-wide-ep-1p-ep8-1d-ep8-decode",
            "tester-wide-ep-1p-ep8-1d-ep8",
            api_version="apps/v1",
        ),
        _live_object("Pod", "decode-0", "tester-wide-ep-1p-ep8-1d-ep8", ready=True),
    ]
    _mock_discovery(monkeypatch, [objects, [], []])
    calls = []
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")
    monkeypatch.setenv("USER", "tester")
    monkeypatch.delenv("MANIFESTO_CLUSTER", raising=False)
    monkeypatch.setattr(workflow, "run", lambda cmd, **_: calls.append(cmd) or 0)

    rc = main(["stop", str(MODEL)])

    assert rc == 0
    assert calls == [[
        "kubectl",
        "-n",
        "workload-ns",
        "delete",
        "deployments.apps/tester-wide-ep-1p-ep8-1d-ep8-decode",
        "--ignore-not-found=true",
    ]]
    assert "Stopped tester-wide-ep-1p-ep8-1d-ep8." in capsys.readouterr().out


def test_stop_instance_is_idempotent(monkeypatch, capsys):
    _mock_discovery(monkeypatch, [[]])

    rc = main(["stop", "--instance", "alice-qwen", "--namespace", "workload-ns"])

    assert rc == 0
    assert "already absent" in capsys.readouterr().out


def test_bare_stop_requires_tty(monkeypatch, capsys):
    monkeypatch.setattr(workflow.sys.stdin, "isatty", lambda: False)

    rc = main(["stop", "--namespace", "workload-ns"])

    assert rc == 2
    assert "interactive selection requires a TTY" in capsys.readouterr().err


def test_bare_stop_uses_numbered_picker_without_fzf(monkeypatch, capsys):
    objects = [
        _live_object("Deployment", "alice-model", "alice-qwen", api_version="apps/v1"),
        _live_object("Pod", "alice-model-1", "alice-qwen", ready=True),
    ]
    _mock_discovery(monkeypatch, [objects, [], []])
    calls = []
    monkeypatch.setattr(workflow.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(workflow.shutil, "which", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: "1")
    monkeypatch.setattr(workflow, "run", lambda cmd, **_: calls.append(cmd) or 0)

    rc = main(["stop", "--namespace", "workload-ns"])

    assert rc == 0
    assert calls[0][3:6] == ["delete", "deployments.apps/alice-model", "--ignore-not-found=true"]
    assert "Stopped alice-qwen." in capsys.readouterr().out


def test_picker_previews_prefetched_resources_without_more_cluster_reads(monkeypatch):
    record = workflow.ServerRecord(
        "alice-qwen",
        (workflow.LiveResource("apps/v1", "Deployment", "alice-model", {}),),
    )
    captured = {}

    class Result:
        returncode = 0
        stdout = "alice-qwen\tReady\tqwen\tdecode\t1/1\t2h\n"

    def fake_run(cmd, **kwargs):
        preview = next(arg for arg in cmd if arg.startswith("--preview="))
        directory = preview.removeprefix("--preview=cat ").removesuffix("/{1}").strip("'")
        captured.update(cmd=cmd, kwargs=kwargs, preview=(Path(directory) / "alice-qwen").read_text())
        return Result()

    monkeypatch.setattr(workflow.shutil, "which", lambda _: "/usr/bin/fzf")
    monkeypatch.setattr(workflow, "capture", lambda cmd, **_: pytest.fail(f"unexpected read: {cmd}"))
    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    config = workflow.RuntimeConfig("tester", "workload-ns", None, Path("/tmp/rendered.yaml"))

    assert workflow.pick_server([record], config) == record
    assert "Deployment           alice-model" in captured["preview"]
    assert captured["kwargs"]["input"].startswith("alice-qwen\t")


def test_stop_now_preserves_force_delete_flags(monkeypatch):
    objects = [_live_object("Deployment", "alice-model", "alice-qwen", api_version="apps/v1")]
    _mock_discovery(monkeypatch, [objects, [], []])
    calls = []
    monkeypatch.setattr(workflow, "run", lambda cmd, **_: calls.append(cmd) or 0)

    rc = main(["stop", "--namespace", "workload-ns", "--instance", "alice-qwen", "--now"])

    assert rc == 0
    assert calls[0][-2:] == ["--grace-period=0", "--force"]


def test_completion_scripts_use_parser_driven_candidates(capsys):
    assert main(["completion", "bash"]) == 0
    bash = capsys.readouterr().out
    assert "complete -o filenames -F _manifesto_complete manifesto" in bash
    assert "manifesto __complete" in bash

    assert main(["completion", "zsh"]) == 0
    zsh = capsys.readouterr().out
    assert "compdef _manifesto manifesto" in zsh
    assert "manifesto __complete" in zsh
    assert '"${words[@]:1}"' in zsh

    assert main(["completion", "fish"]) == 0
    fish = capsys.readouterr().out
    assert "complete -c manifesto" in fish
    assert "manifesto __complete" in fish


def test_completion_candidates_follow_parser_and_catalogs(capsys):
    assert main(["__complete", "render", "manifest", "--v"]) == 0
    assert capsys.readouterr().out.splitlines() == ["--vllm-env"]

    assert main(["__complete", "render", "manifest", "--c"]) == 0
    assert capsys.readouterr().out.splitlines() == ["--cache-root", "--cluster", "--context"]

    assert main(["__complete", "completion", ""]) == 0
    assert capsys.readouterr().out.splitlines()[-3:] == ["bash", "fish", "zsh"]

    assert main(["__complete", "config", "resolve", "models", "deepseek-v4/"]) == 0
    assert "deepseek-v4/1P-EP8-1D-EP8" in capsys.readouterr().out.splitlines()

    assert main(["__complete", "deploy", "deepseek-v4/"]) == 0
    assert "deepseek-v4/1P-EP8-1D-EP8" in capsys.readouterr().out.splitlines()

    assert main(["__complete", "deploy", "r"]) == 0
    assert "routing" in capsys.readouterr().out.splitlines()

    assert main(["__complete", "deploy", "qwen/aggregated", "--routing-profile", "w"]) == 0
    assert capsys.readouterr().out.splitlines() == ["wide-ep-lws-config"]

    assert main(
        [
            "__complete",
            "deploy",
            "qwen/aggregated",
            "--cluster=example-gb200",
            "--gpu",
            "b",
        ]
    ) == 0
    assert capsys.readouterr().out.splitlines() == ["b200"]


def test_completion_candidates_include_local_config_files(monkeypatch, tmp_path, capsys):
    (tmp_path / "local-cluster.yaml").touch()
    (tmp_path / "local-cluster.txt").touch()
    monkeypatch.chdir(tmp_path)

    assert main(["__complete", "render", "bootstrap", "--cluster", "local-"]) == 0
    assert capsys.readouterr().out.splitlines() == ["local-cluster.yaml"]


def test_completion_candidates_include_live_instances(monkeypatch, capsys):
    objects = [_live_object("Deployment", "alice-model", "alice-qwen", api_version="apps/v1")]
    _mock_discovery(monkeypatch, [objects])

    assert main(
        ["__complete", "stop", "--namespace", "workload-ns", "--instance", "alice"]
    ) == 0
    assert capsys.readouterr().out.splitlines() == ["alice-qwen"]


def test_completion_live_instances_honors_inline_namespace(monkeypatch, capsys):
    objects = [_live_object("Deployment", "alice-model", "alice-qwen", api_version="apps/v1")]
    commands = []

    def fake_capture(cmd, **_):
        commands.append(cmd)
        return json.dumps({"items": objects if "deployments.apps" in cmd else []})

    monkeypatch.setattr(workflow, "capture", fake_capture)

    assert main(["__complete", "servers", "--namespace=inline-ns", "--instance", "alice"]) == 0
    assert capsys.readouterr().out.splitlines() == ["alice-qwen"]
    assert any(command[:3] == ["kubectl", "-n", "inline-ns"] for command in commands)


def test_completion_gives_up_quickly_on_a_slow_cluster(monkeypatch, capsys):
    limits = []

    def fake_capture(cmd, **kwargs):
        if "config" in cmd:
            return "workload-ns"
        limits.append((kwargs.get("timeout"), kwargs.get("retries", 0)))
        raise workflow.WorkflowError("timed out")

    monkeypatch.setattr(workflow, "capture", fake_capture)

    assert main(["__complete", "servers", "--instance", ""]) == 0
    assert capsys.readouterr().out == ""
    # Both the discovery call and the per-type lists must inherit the short budget.
    assert len(limits) > 1
    assert all(timeout == workflow.COMPLETION_KUBECTL_TIMEOUT for timeout, _ in limits)
    assert all(retries == 0 for _, retries in limits)


def test_discovery_lists_every_managed_type_without_a_preflight(monkeypatch):
    requested = []
    # Only satisfiable if every list is in flight at once; a sequential
    # implementation deadlocks here and trips the timeout.
    overlapping = threading.Barrier(len(workflow.DISCOVERY_RESOURCE_TYPES), timeout=10)

    def fake_capture(cmd, **_):
        assert "api-resources" not in cmd, "discovery must not cost a preflight round trip"
        resource_type = cmd[cmd.index("get") + 1]
        requested.append(resource_type)
        overlapping.wait()
        if resource_type != "pods":
            return json.dumps({"apiVersion": "v1", "kind": "List", "items": []})
        # kubectl rewraps single-type output as a v1 List with per-item kinds.
        return json.dumps(
            {
                "apiVersion": "v1",
                "kind": "List",
                "items": [
                    {
                        "apiVersion": "v1",
                        "kind": "Pod",
                        "metadata": {
                            "name": "alice-model-0",
                            "labels": {"app.kubernetes.io/instance": "alice-qwen"},
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(workflow, "capture", fake_capture)
    config = workflow.RuntimeConfig("tester", "workload-ns", None, Path("/tmp/rendered.yaml"))

    resources = workflow.discover_live_resources(config)

    assert sorted(requested) == sorted(workflow.DISCOVERY_RESOURCE_TYPES)
    assert set(MANAGED_RESOURCE_TYPES.values()) <= set(requested)
    assert [(item.api_version, item.kind, item.name) for item in resources] == [
        ("v1", "Pod", "alice-model-0")
    ]


def test_typed_list_items_recover_their_kind():
    objects = workflow._objects_from_list(
        json.dumps({"apiVersion": "v1", "kind": "PodList", "items": [{"metadata": {"name": "p"}}]})
    )

    assert objects == [{"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"}}]


@pytest.mark.parametrize("payload", ['{"kind": "Status", "status": "Failure"}', "[]", "null"])
def test_a_non_list_payload_is_an_error_not_an_empty_result(payload):
    with pytest.raises(workflow.WorkflowError, match="expected a list"):
        workflow._objects_from_list(payload)


def test_servers_reports_an_empty_namespace_without_crashing(monkeypatch, capsys):
    _mock_discovery(monkeypatch, [[]])

    assert main(["servers", "--namespace", "workload-ns"]) == 0
    assert capsys.readouterr().out.splitlines() == ["INSTANCE  STATE  MODEL  ROLES  PODS  AGE"]


def test_kubectl_limits_rejects_unknown_knobs_and_restores_state(monkeypatch):
    monkeypatch.delenv("MANIFESTO_KUBECTL_TIMEOUT", raising=False)
    monkeypatch.delenv("MANIFESTO_KUBECTL_RETRIES", raising=False)

    with pytest.raises(TypeError):
        with workflow.kubectl_limits(tmeout=1):
            pass

    with workflow.kubectl_limits(timeout=3.0, retries=0):
        assert workflow.kubectl_timeout() == 3.0
        assert workflow.kubectl_retries() == 0
        with workflow.kubectl_limits(timeout=9.0):
            assert workflow.kubectl_timeout() == 9.0
            assert workflow.kubectl_retries() == 0
        assert workflow.kubectl_timeout() == 3.0
    assert workflow.kubectl_timeout() == workflow.DEFAULT_KUBECTL_TIMEOUT
    assert workflow.kubectl_retries() == workflow.DEFAULT_KUBECTL_RETRIES


def test_a_failed_list_is_not_mistaken_for_an_empty_namespace(monkeypatch, capsys):

    def fake_capture(cmd, **_):
        if "deployments.apps" in cmd:
            raise workflow.WorkflowError('Error from server (Forbidden): deployments.apps is forbidden')
        return json.dumps({"apiVersion": "v1", "kind": "PodList", "items": []})

    monkeypatch.setattr(workflow, "capture", fake_capture)
    monkeypatch.setenv("MANIFESTO_NAMESPACE", "workload-ns")

    assert main(["stop", "--instance", "alice-qwen"]) == 1
    assert "Forbidden" in capsys.readouterr().err


def test_unserved_resource_types_do_not_fail_discovery(monkeypatch):

    def fake_capture(cmd, **kwargs):
        if "leaderworkersets.leaderworkerset.x-k8s.io" in cmd:
            assert "the server doesn't have a resource type" in kwargs["tolerate"]
            return ""
        return json.dumps({"apiVersion": "v1", "kind": "PodList", "items": []})

    monkeypatch.setattr(workflow, "capture", fake_capture)
    config = workflow.RuntimeConfig("tester", "workload-ns", None, Path("/tmp/rendered.yaml"))

    assert workflow.list_objects(
        config,
        ("leaderworkersets.leaderworkerset.x-k8s.io", "pods"),
        workflow.MANIFESTO_SELECTOR,
    ) == []


def test_teardown_verifies_against_the_full_managed_set(monkeypatch, capsys):
    """The orphan-pod round is pods-only, but the final word must not be narrowed.

    Narrowing the last check to the types seen before the delete would let a type
    missed by an incomplete first discovery go unreported as a teardown success.
    """

    objects = [
        _live_object("Deployment", "alice-model", "alice-qwen", api_version="apps/v1"),
        _live_object("Pod", "alice-model-0", "alice-qwen", ready=True),
    ]
    rounds = iter([objects, [], []])
    requested = []

    def fake_list_objects(config, resource_types, selector, **_):
        requested.append(tuple(resource_types))
        return list(next(rounds))

    monkeypatch.setattr(workflow, "list_objects", fake_list_objects)
    monkeypatch.setattr(workflow, "run", lambda cmd, **_: 0)

    assert main(["stop", "--namespace", "workload-ns", "--instance", "alice-qwen"]) == 0
    assert "Stopped alice-qwen." in capsys.readouterr().out
    full_set = workflow.DISCOVERY_RESOURCE_TYPES
    assert requested == [full_set, ("pods",), full_set]


def test_capture_retries_transient_cluster_failures(monkeypatch):
    attempts = []

    class Result:
        def __init__(self, returncode, stderr):
            self.returncode = returncode
            self.stdout = "ok"
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        attempts.append(kwargs.get("timeout"))
        if len(attempts) < 3:
            return Result(1, "Unable to connect to the server: dial tcp: i/o timeout")
        return Result(0, "")

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    monkeypatch.setattr(workflow.time, "sleep", lambda _: None)

    assert workflow.capture(["kubectl", "get", "pods"], timeout=30, retries=2) == "ok"
    assert attempts == [30, 30, 30]


def test_capture_does_not_retry_a_genuine_error(monkeypatch):
    attempts = []

    class Result:
        returncode = 1
        stdout = ""
        stderr = 'namespaces "missing" not found'

    def fake_run(cmd, **_):
        attempts.append(cmd)
        return Result()

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    monkeypatch.setattr(workflow.time, "sleep", lambda _: None)

    with pytest.raises(workflow.WorkflowError, match="not found"):
        workflow.capture(["kubectl", "get", "pods"], retries=3)
    assert len(attempts) == 1


def test_capture_surfaces_a_hung_cluster_read(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise workflow.subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    monkeypatch.setattr(workflow.time, "sleep", lambda _: None)

    with pytest.raises(workflow.WorkflowError, match="MANIFESTO_KUBECTL_TIMEOUT"):
        workflow.capture(["kubectl", "get", "pods"], timeout=5, retries=1)


def test_kubectl_read_budget_is_configurable(monkeypatch):
    monkeypatch.delenv("MANIFESTO_KUBECTL_TIMEOUT", raising=False)
    monkeypatch.delenv("MANIFESTO_KUBECTL_RETRIES", raising=False)
    assert workflow.kubectl_timeout() == workflow.DEFAULT_KUBECTL_TIMEOUT
    assert workflow.kubectl_retries() == workflow.DEFAULT_KUBECTL_RETRIES

    monkeypatch.setenv("MANIFESTO_KUBECTL_TIMEOUT", "600")
    monkeypatch.setenv("MANIFESTO_KUBECTL_RETRIES", "5")
    assert workflow.kubectl_timeout() == 600
    assert workflow.kubectl_retries() == 5
    # Slightly inside the process deadline, so kubectl reports its own diagnostic.
    assert workflow._request_timeout_flag(600) == ["--request-timeout=540s"]

    monkeypatch.setenv("MANIFESTO_KUBECTL_TIMEOUT", "0")
    assert workflow.kubectl_timeout() is None
    assert workflow._request_timeout_flag(None) == []


def test_teardown_allowlist_covers_every_rendered_kind():
    cluster = load_cluster(CLUSTER)
    objects = render(load_spec(MODEL, cluster), user="tester", cluster=cluster)

    assert {(obj["apiVersion"], obj["kind"]) for obj in objects} <= set(MANAGED_RESOURCE_TYPES)
