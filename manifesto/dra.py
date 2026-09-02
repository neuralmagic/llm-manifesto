"""Kubernetes Dynamic Resource Allocation helpers for accelerator workloads."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


DRA_API_VERSION = "resource.k8s.io/v1"
DRA_CLAIM_NAME = "accelerator"
DRA_REQUEST_NAME = "gpu"
RESOURCE_CLAIM_TEMPLATE_KIND = "ResourceClaimTemplate"


def accelerator_claim_template_name(
    workload_name: str,
    *,
    labels: dict[str, str],
    device_class_name: str,
    count: int,
) -> str:
    """Return a stable name that changes whenever the immutable claim spec does."""

    claim_spec = _accelerator_claim_template_spec(
        labels=labels,
        device_class_name=device_class_name,
        count=count,
    )
    identity = json.dumps(claim_spec, sort_keys=True, separators=(",", ":"))
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
    if labels:
        metadata["labels"] = dict(labels)
    return {
        "apiVersion": DRA_API_VERSION,
        "kind": RESOURCE_CLAIM_TEMPLATE_KIND,
        "metadata": metadata,
        "spec": _accelerator_claim_template_spec(
            labels=labels,
            device_class_name=device_class_name,
            count=count,
        ),
    }


def _accelerator_claim_template_spec(
    *,
    labels: dict[str, str],
    device_class_name: str,
    count: int,
) -> dict[str, Any]:
    claim_metadata = {"labels": dict(labels)} if labels else None
    return {
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
    }


def attach_accelerator_claim(
    pod_spec: dict[str, Any],
    container: dict[str, Any],
    *,
    template_name: str,
) -> None:
    attach_resource_claim(
        pod_spec,
        container,
        name=DRA_CLAIM_NAME,
        template_name=template_name,
    )


def attach_resource_claim(
    pod_spec: dict[str, Any],
    container: dict[str, Any],
    *,
    name: str,
    template_name: str,
) -> None:
    """Attach a named ResourceClaimTemplate to one workload container."""

    pod_claims = pod_spec.setdefault("resourceClaims", [])
    _require_available_claim_name(pod_claims, name, "pod resourceClaims")
    pod_claims.append(
        {
            "name": name,
            "resourceClaimTemplateName": template_name,
        }
    )

    container_claims = container.setdefault("resources", {}).setdefault("claims", [])
    _require_available_claim_name(container_claims, name, "container resources.claims")
    container_claims.append({"name": name})


def _require_available_claim_name(
    claims: list[dict[str, Any]], name: str, location: str
) -> None:
    if any(claim.get("name") == name for claim in claims):
        raise ValueError(
            f"generated resource claim name {name!r} conflicts with an existing "
            f"entry in {location}"
        )
