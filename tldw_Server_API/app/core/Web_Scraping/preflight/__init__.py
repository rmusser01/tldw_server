"""Governed preflight facade contracts."""

from __future__ import annotations

from ..contracts import (
    PreflightAdvice,
    PreflightResult,
    RuntimeFailure,
    WebScrapingStatus,
)
from .context import (
    BudgetKind,
    PreflightConsumed,
    PreflightDeadlineExceeded,
    PreflightExecutionContext,
    PreflightLimits,
    PreflightRuntimeControls,
)
from .options import PreflightOptions, ScanDepth
from .probes import (
    BrowserProbe,
    BrowserProbeOptions,
    BrowserProbePage,
    ExternalToolProbe,
    ExternalToolResult,
    HttpProbe,
    ProbeBudgetExhausted,
    ProbeError,
    ProbeHttpRequest,
    ProbeHttpResponse,
    ProbeTimeout,
    ProbeUnavailable,
)
from .target import PreflightTarget

__all__ = [
    "BrowserProbe",
    "BrowserProbeOptions",
    "BrowserProbePage",
    "BudgetKind",
    "ExternalToolProbe",
    "ExternalToolResult",
    "HttpProbe",
    "PreflightAdvice",
    "PreflightConsumed",
    "PreflightDeadlineExceeded",
    "PreflightExecutionContext",
    "PreflightLimits",
    "PreflightOptions",
    "PreflightResult",
    "PreflightRuntimeControls",
    "PreflightTarget",
    "ProbeBudgetExhausted",
    "ProbeError",
    "ProbeHttpRequest",
    "ProbeHttpResponse",
    "ProbeTimeout",
    "ProbeUnavailable",
    "RuntimeFailure",
    "ScanDepth",
    "WebScrapingStatus",
]
