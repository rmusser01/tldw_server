"""Capability helpers for managed vLLM instances."""

from __future__ import annotations

from typing import Any

_PROBE_REQUIRED_CAPABILITIES = frozenset({"embeddings", "vision", "audio", "multimodal"})


def normalize_capabilities(raw: dict[str, Any] | None) -> dict[str, bool]:
    return {str(key): bool(value) for key, value in (raw or {}).items()}


def derive_effective_capabilities(
    *,
    declared_capabilities: dict[str, Any] | None,
    probed_capabilities: dict[str, Any] | None,
) -> dict[str, bool]:
    """Combine declared and probed capability layers for routing decisions."""

    declared = normalize_capabilities(declared_capabilities)
    probed = normalize_capabilities(probed_capabilities)
    effective: dict[str, bool] = {}

    for key in sorted(set(declared) | set(probed)):
        declared_value = declared.get(key, False)
        probed_value = probed.get(key)
        if key in _PROBE_REQUIRED_CAPABILITIES:
            effective[key] = declared_value and bool(probed_value)
        else:
            effective[key] = declared_value if probed_value is None else declared_value and probed_value
    return effective
