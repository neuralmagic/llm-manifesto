"""Public render API for converting loaded specs into Kubernetes manifests."""

from .emit import render, render_to_yaml

__all__ = ["render", "render_to_yaml"]
