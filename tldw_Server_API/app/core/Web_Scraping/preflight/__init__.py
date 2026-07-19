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
from .facade import (
    PreflightAdapterOverrides,
    apply_preflight_advice,
    build_execution_context,
    evaluate_target,
    public_preflight_payload,
    run_preflight,
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
from .runner import (
    AnalysisOutput,
    gather_analysis,
    gather_analysis_with_context,
    run_analysis,
)
from .target import PreflightTarget

__all__ = [
    "AnalysisOutput",
    "BrowserProbe",
    "BrowserProbeOptions",
    "BrowserProbePage",
    "BudgetKind",
    "ExternalToolProbe",
    "ExternalToolResult",
    "HttpProbe",
    "PreflightAdvice",
    "PreflightAdapterOverrides",
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
    "apply_preflight_advice",
    "build_execution_context",
    "evaluate_target",
    "gather_analysis",
    "gather_analysis_with_context",
    "public_preflight_payload",
    "run_analysis",
    "run_preflight",
]
