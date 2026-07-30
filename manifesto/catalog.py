"""Resolution for bundled and user-owned Manifesto configuration catalogs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR_NAME = "llm-manifesto"
CATALOGS = {"models", "clusters", "routing"}


@dataclass(frozen=True)
class CatalogEntry:
    """A named YAML config visible through one of Manifesto's catalogs."""

    name: str
    path: Path
    source: str
    shadows: Path | None = None


def config_home() -> Path:
    if configured := os.environ.get("MANIFESTO_CONFIG_HOME"):
        return Path(configured).expanduser()
    if xdg_home := os.environ.get("XDG_CONFIG_HOME"):
        return Path(xdg_home).expanduser() / CONFIG_DIR_NAME
    return Path.home() / ".config" / CONFIG_DIR_NAME


def catalog_entries(catalog: str) -> list[CatalogEntry]:
    """Return effective catalog entries in resolution-precedence order."""

    if catalog not in CATALOGS:
        raise ValueError(f"unknown config catalog: {catalog}")

    entries: dict[str, CatalogEntry] = {}
    roots = ((config_home() / catalog, "user"), (ROOT / catalog, "bundled"))
    for root, source in roots:
        if not root.is_dir():
            continue
        for path in sorted((*root.rglob("*.yaml"), *root.rglob("*.yml"))):
            name = path.relative_to(root).with_suffix("").as_posix()
            existing = entries.get(name)
            if existing is None:
                entries[name] = CatalogEntry(name=name, path=path, source=source)
            elif existing.source == "user" and source == "bundled":
                entries[name] = CatalogEntry(
                    name=existing.name,
                    path=existing.path,
                    source=existing.source,
                    shadows=path,
                )
    return [entries[name] for name in sorted(entries)]


def resolve_catalog_path(value: str, catalog: str) -> str:
    if catalog not in CATALOGS:
        raise ValueError(f"unknown config catalog: {catalog}")
    path = Path(value).expanduser()
    variants = [path]
    if path.suffix.lower() not in {".yaml", ".yml"}:
        variants.extend((Path(f"{path}.yaml"), Path(f"{path}.yml")))

    for candidate in variants:
        if candidate.exists():
            return str(candidate)
    if path.is_absolute():
        return str(path)

    for root in (config_home() / catalog, ROOT / catalog):
        for candidate in variants:
            resolved = root / candidate
            if resolved.exists():
                return str(resolved)
    return value
