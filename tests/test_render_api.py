"""The public renderer produces a ready-to-write artifact for either platform."""

from pathlib import Path
import subprocess

import pytest
import yaml

from manifesto.cluster import load_cluster
from manifesto.render import render, render_kubernetes
from manifesto.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("platform", "profile"), [
    ("kubernetes", "example-stateless-b200"),
    ("kubernetes", "example-gb200"),
    ("kubernetes", "example-h200"),
    ("openshift", "example-stateless-b200"),
    ("slurm", "example-slurm"),
])
def test_public_render_selects_format_from_cluster(platform, profile):
    cluster = load_cluster(ROOT / "clusters" / f"{profile}.yaml")
    cluster.platform = platform
    spec = load_spec(ROOT / "models/qwen/qwen3-0.6b.yaml", cluster)
    assert spec.accelerator is None
    assert spec.accelerator_config(cluster) == cluster.accelerators.get()

    artifact = render(spec, user="tester", cluster=cluster, header=["test provenance"])

    assert isinstance(artifact, str)
    assert "# test provenance\n" in artifact
    if platform == "slurm":
        assert artifact.startswith("#!/bin/bash\n#SBATCH")
        assert "#SBATCH --gres=gpu:1\n" in artifact
        assert "Qwen/Qwen3-0.6B" in artifact
        subprocess.run(["bash", "-n"], input=artifact, text=True, check=True)
    else:
        assert artifact.startswith("# test provenance\n---\n")
        objects = list(yaml.safe_load_all(artifact))
        assert objects == render_kubernetes(spec, user="tester", cluster=cluster)
        assert {obj["kind"] for obj in objects} >= {"Deployment", "Service"}
        model = next(obj for obj in objects if obj["kind"] == "Deployment" and obj["metadata"]["name"].endswith("-decode"))
        resources = model["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert resources["requests"]["ephemeral-storage"] == "32Gi"
        assert resources["limits"]["ephemeral-storage"] == "32Gi"


def test_public_render_supports_kubernetes_routing_only():
    cluster = load_cluster(ROOT / "clusters/example-stateless-b200.yaml")
    spec = load_spec(ROOT / "models/qwen/aggregated.yaml", cluster)

    objects = list(yaml.safe_load_all(render(spec, user="tester", cluster=cluster, routing_only=True)))

    assert any(obj["kind"] == "InferencePool" for obj in objects)
    assert not any(obj["metadata"].get("labels", {}).get("app.kubernetes.io/component") == "model-server" for obj in objects)


def test_public_render_rejects_slurm_routing_only():
    cluster = load_cluster(ROOT / "clusters/example-slurm.yaml")
    spec = load_spec(ROOT / "models/qwen/qwen3-0.6b.yaml", cluster)

    with pytest.raises(ValueError, match="routing-only"):
        render(spec, user="tester", cluster=cluster, routing_only=True)


def test_kubernetes_object_renderer_requires_kubernetes_profile():
    cluster = load_cluster(ROOT / "clusters/example-slurm.yaml")
    spec = load_spec(ROOT / "models/qwen/qwen3-0.6b.yaml", cluster)

    with pytest.raises(ValueError, match="render_kubernetes requires"):
        render_kubernetes(spec, user="tester", cluster=cluster)
