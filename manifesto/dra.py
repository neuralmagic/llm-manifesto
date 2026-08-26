"""Kubernetes Dynamic Resource Allocation helpers for accelerator workloads."""

from __future__ import annotations

import hashlib
import re
from typing import Any


DRA_API_VERSION = "resource.k8s.io/v1"
DRA_CLAIM_NAME = "accelerator"
DRA_REQUEST_NAME = "gpu"
RESOURCE_CLAIM_TEMPLATE_KIND = "ResourceClaimTemplate"


def accelerator_claim_template_name(
    workload_name: str,
    *,
    device_class_name: str,
    count: int,
) -> str:
    """Return a stable name that changes whenever the immutable claim spec does."""

    identity = f"{DRA_API_VERSION}\0{device_class_name}\0{count}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
    base = re.sub(r"[^a-z0-9-]+", "-", workload_name.lower()).strip("-")
    base = re.sub(r"-+", "-", base) or "workload"
    value = f"{base}-accelerator-{digest}"
    if len(value) <= 63:
        return value
    return f"{base[: 63 - len(digest) - 1].rstrip('-')}-{digest}"


def render_accelerator_claim_template(
    *,
    name: str,
    labels: dict[str, str],
    device_class_name: str,
    count: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": name}
    claim_metadata: dict[str, Any] = {}
    if labels:
        metadata["labels"] = dict(labels)
        claim_metadata["labels"] = dict(labels)
    return {
        "apiVersion": DRA_API_VERSION,
        "kind": RESOURCE_CLAIM_TEMPLATE_KIND,
        "metadata": metadata,
        "spec": {
            **({"metadata": claim_metadata} if claim_metadata else {}),
            "spec": {
                "devices": {
                    "requests": [
                        {
                            "name": DRA_REQUEST_NAME,
                            "exactly": {
                                "deviceClassName": device_class_name,
                                "allocationMode": "ExactCount",
                                "count": count,
                            },
                        }
                    ]
                }
            },
        },
    }


def attach_accelerator_claim(
    pod_spec: dict[str, Any],
    container: dict[str, Any],
    *,
    template_name: str,
) -> None:
    pod_claims = pod_spec.setdefault("resourceClaims", [])
    _require_available_claim_name(pod_claims, "pod resourceClaims")
    pod_claims.append(
        {
            "name": DRA_CLAIM_NAME,
            "resourceClaimTemplateName": template_name,
        }
    )

    container_claims = container.setdefault("resources", {}).setdefault("claims", [])
    _require_available_claim_name(container_claims, "container resources.claims")
    container_claims.append({"name": DRA_CLAIM_NAME})


def _require_available_claim_name(claims: list[dict[str, Any]], location: str) -> None:
    if any(claim.get("name") == DRA_CLAIM_NAME for claim in claims):
        raise ValueError(
            f"generated accelerator claim name {DRA_CLAIM_NAME!r} conflicts with "
            f"an existing entry in {location}"
        )
