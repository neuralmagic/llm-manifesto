"""Shared Kubernetes manifest helpers used by the renderers."""

from __future__ import annotations

from typing import Any


def env_list(values: dict[str, str]) -> list[dict[str, Any]]:
    return [{"name": key, "value": str(value)} for key, value in sorted(values.items())]


def secret_env(name: str, secret_name: str, key: str, *, optional: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {
            "secretKeyRef": {
                "name": secret_name,
                "key": key,
                "optional": optional,
            }
        },
    }


def field_ref_env(name: str, field_path: str) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {"fieldRef": {"fieldPath": field_path}},
    }
