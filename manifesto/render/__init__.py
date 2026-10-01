"""Render deployment artifacts using the platform selected by the cluster."""

from ..cluster import Cluster
from ..slurm import render_slurm
from ..spec import DeploymentSpec
from .emit import render_kubernetes, render_to_yaml

__all__ = ["render", "render_kubernetes", "render_to_yaml"]


def render(
    spec: DeploymentSpec,
    *,
    user: str,
    cluster: Cluster,
    routing_only: bool = False,
    header: list[str] | None = None,
) -> str:
    """Return Kubernetes YAML or a Slurm batch script, ready to write or submit."""
    if cluster.platform == "slurm":
        if routing_only:
            raise ValueError("Slurm does not support routing-only rendering")
        return render_slurm(spec, user=user, cluster=cluster, header=header)
    return render_to_yaml(
        render_kubernetes(spec, user=user, cluster=cluster, routing_only=routing_only),
        header=header,
    )
