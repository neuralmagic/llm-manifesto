"""YAML spec composition for base specs and small override files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .catalog import resolve_catalog_path


def load_spec_data(path: str | Path) -> dict[str, Any]:
    return _load_spec_data(Path(path), seen=[])


def load_routing_profile(value: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(resolve_catalog_path(str(value), "routing"))
    if not resolved.is_file():
        raise ValueError(f"unknown routing profile: {value}")
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"routing profile must be a YAML mapping: {resolved}")
    if data.get("kind") != "EndpointPickerConfig":
        raise ValueError(f"routing profile must have kind: EndpointPickerConfig: {resolved}")
    return resolved, data


def _load_spec_data(path: Path, *, seen: list[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(item) for item in [*seen, resolved])
        raise ValueError(f"circular spec extends chain: {chain}")

    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"spec must be a YAML mapping: {path}")

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data
    if not isinstance(parent_ref, str):
        raise ValueError(f"extends must be a relative path string: {path}")

    parent = _load_spec_data(resolved.parent / parent_ref, seen=[*seen, resolved])
    return merge_overrides(parent, data)


def merge_overrides(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if value == "$delete":
            merged.pop(key, None)
        elif key == "roles" and isinstance(value, dict):
            merged[key] = _merge_roles(merged.get(key, []), value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_overrides(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _merge_roles(base_roles: Any, role_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(base_roles, list):
        raise ValueError("role overrides require base roles to be a list")

    roles = []
    seen = set()
    for role in base_roles:
        if not isinstance(role, dict) or not isinstance(role.get("name"), str):
            raise ValueError("base roles must be mappings with string names")
        name = role["name"]
        seen.add(name)
        override = role_overrides.get(name)
        if override is None:
            roles.append(deepcopy(role))
            continue
        if not isinstance(override, dict):
            raise ValueError(f"role override must be a mapping: {name}")
        roles.append(merge_overrides(role, override))

    unknown = sorted(set(role_overrides) - seen)
    if unknown:
        raise ValueError(f"role override references unknown role(s): {', '.join(unknown)}")
    return roles
