"""Render namespace prerequisites declared by a cluster profile."""

from __future__ import annotations

from ..cluster import Cluster


def render_bootstrap(cluster: Cluster, namespace: str) -> list[dict]:
    """Render resources that Manifesto owns for first-time namespace setup."""
    claim = cluster.storage.shared_claim
    if claim is None:
        return []

    spec = {
        "accessModes": claim.access_modes,
        "resources": {"requests": {"storage": claim.size}},
    }
    if claim.storage_class_name is not None:
        spec["storageClassName"] = claim.storage_class_name

    return [
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": cluster.storage.shared_claim_name,
                "namespace": namespace,
                "labels": {"app.kubernetes.io/name": "manifesto"},
            },
            "spec": spec,
        }
    ]
