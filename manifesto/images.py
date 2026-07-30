"""Central image catalog loading and reference resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_CATALOG = ROOT / "config" / "images.yaml"


@dataclass(frozen=True)
class ImageCatalog:
    data: dict[str, Any]

    def get(self, ref: str, **format_values: str) -> str:
        value: Any = self.data
        for part in ref.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"unknown image ref: {ref}")
            value = value[part]
        if not isinstance(value, str):
            raise ValueError(f"image ref does not resolve to a string: {ref}")
        return value.format(**format_values)


def load_image_catalog(path: str | Path = DEFAULT_IMAGE_CATALOG) -> ImageCatalog:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"image catalog must be a YAML mapping: {path}")
    return ImageCatalog(data)


DEFAULT_IMAGES = load_image_catalog()


def apply_image_refs(data: dict[str, Any], images: ImageCatalog = DEFAULT_IMAGES) -> dict[str, Any]:
    normalized = dict(data)
    model = dict(normalized.get("model", {}))
    image_ref = model.pop("image_ref", None)
    if image_ref and "image" not in model:
        model["image"] = images.get(str(image_ref))
    normalized["model"] = model
    return normalized
