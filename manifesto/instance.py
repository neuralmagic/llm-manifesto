"""Instance identity helpers for names, labels, selectors, and user-scoped paths."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

DNS_LABEL_MAX = 63


def _slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "default"


def _truncate_hash(value: str, limit: int = DNS_LABEL_MAX) -> str:
    if len(value) <= limit:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    head = value[: limit - len(digest) - 1].rstrip("-")
    return f"{head}-{digest}"


@dataclass(frozen=True)
class Instance:
    user: str
    release: str

    @property
    def user_slug(self) -> str:
        return _slug(self.user)

    @property
    def release_slug(self) -> str:
        return _slug(self.release)

    @property
    def instance_id(self) -> str:
        return _truncate_hash(f"{self.user_slug}-{self.release_slug}")

    def name(self, component: str, *, hostname_safe: bool = True, max_length: int | None = None) -> str:
        limit = max_length if max_length is not None else (DNS_LABEL_MAX if hostname_safe else 253)
        if limit < 10 or limit > 253:
            raise ValueError(f"name max_length must be between 10 and 253, got {limit}")
        return _truncate_hash(f"{self.instance_id}-{_slug(component)}", limit)

    def user_scoped_name(self, component: str, *, hostname_safe: bool = True) -> str:
        limit = DNS_LABEL_MAX if hostname_safe else 253
        return _truncate_hash(f"{self.user_slug}-{_slug(component)}", limit)

    def labels(self, component: str | None = None, role: str | None = None) -> dict[str, str]:
        labels = {
            "app.kubernetes.io/name": "manifesto",
            "app.kubernetes.io/instance": self.instance_id,
            "llm-d.ai/owner": self.user_slug,
        }
        if component:
            labels["app.kubernetes.io/component"] = _slug(component)
        if role:
            labels["llm-d.ai/role"] = _slug(role)
        return labels

    def pod_selector(self, role: str | None = None) -> dict[str, str]:
        selector = {"app.kubernetes.io/instance": self.instance_id}
        if role:
            selector["llm-d.ai/role"] = _slug(role)
        return selector
