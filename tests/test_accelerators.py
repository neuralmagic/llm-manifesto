"""Accelerator catalog and selection regression tests."""

from pathlib import Path

import pytest

from manifesto.cluster import load_cluster
from manifesto.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = load_cluster(ROOT / "clusters" / "example-gb200.yaml")
EXAMPLE_H200 = load_cluster(ROOT / "clusters" / "example-h200.yaml")


def test_clusters_define_their_accelerators_and_default():
    assert CLUSTER.accelerators.default == "gb200"
    assert {"gb200", "b200"} == set(CLUSTER.accelerators.profiles)
    assert CLUSTER.accelerators.get().gpu_arch == "gb200"
    assert EXAMPLE_H200.accelerators.default == "h200"
    assert EXAMPLE_H200.accelerators.get().torch_cuda_arch_list == "9.0"


def test_deployments_inherit_default_and_can_override_accelerator():
    default_spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    h200_spec = load_spec(ROOT / "models" / "qwen" / "h200-aggregated.yaml", EXAMPLE_H200)

    assert default_spec.accelerator is None
    assert default_spec.accelerator_config(CLUSTER).gpu_arch == "gb200"
    assert h200_spec.accelerator == "h200"
    assert h200_spec.accelerator_config(EXAMPLE_H200).gpu_arch == "h200"


def test_unknown_accelerator_is_rejected():
    with pytest.raises(ValueError, match="unknown accelerator"):
        CLUSTER.accelerators.get("quantum-gpu")
