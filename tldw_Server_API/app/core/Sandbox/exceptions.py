from __future__ import annotations

SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS: tuple[type[Exception], ...] = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

