"""Regression detection for the RAG pipeline.

Detects quality degradation by comparing current metric values against
stored baselines. Useful for catching regressions from:
- Configuration changes (new reranker model, different chunk size)
- Embedding model updates
- Pipeline parameter tuning

Integrates with the retrieval metrics module and quality gating system.

Ported from RAGnarok-AI's regression detection pattern, adapted for tldw_server2.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from .quality_gating import GatingConfig, MetricCategory

# Restrict baseline IDs to safe characters for filenames.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class MetricBaseline(BaseModel):
    """Stored baseline for a set of metrics.

    Attributes:
        baseline_id: Unique identifier for this baseline.
        created_at: ISO timestamp when the baseline was recorded.
        pipeline_config: Configuration snapshot at baseline time.
        metrics: Metric name -> value mapping.
        metadata: Additional metadata (model name, dataset, etc.).
    """

    model_config = {"frozen": True}

    baseline_id: str = Field(..., description="Unique baseline identifier")
    created_at: str = Field(..., description="Creation timestamp (ISO)")
    pipeline_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline configuration at baseline time",
    )
    metrics: dict[str, float] = Field(..., description="Metric name -> value")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class RegressionResult(BaseModel):
    """Result of regression detection for a single metric.

    Attributes:
        metric_name: Name of the metric.
        baseline_value: Value from the stored baseline.
        current_value: Current observed value.
        delta: Absolute difference (current - baseline).
        delta_percent: Percentage change relative to baseline.
        threshold: Degradation threshold used.
        regressed: Whether the metric has regressed beyond threshold.
        category: Stable or unstable metric category (quality gating).
    """

    model_config = {"frozen": True}

    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    delta_percent: float
    threshold: float
    regressed: bool
    category: MetricCategory = Field(
        default=MetricCategory.STABLE,
        description="Metric category (stable/unstable) from quality gating",
    )


class RegressionReport(BaseModel):
    """Full regression detection report.

    Attributes:
        baseline_id: The baseline compared against.
        timestamp: When this report was generated.
        results: Per-metric regression results.
        has_regression: Whether any stable metric regressed.
        has_warnings: Whether any unstable metric regressed (warning only).
        summary: Human-readable summary.
    """

    model_config = {"frozen": True}

    baseline_id: str
    timestamp: str
    results: list[RegressionResult]
    has_regression: bool
    has_warnings: bool = Field(
        default=False,
        description="Whether any unstable metrics regressed (warning only)",
    )
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "baseline_id": self.baseline_id,
            "timestamp": self.timestamp,
            "has_regression": self.has_regression,
            "has_warnings": self.has_warnings,
            "summary": self.summary,
            "results": [
                {
                    "metric": r.metric_name,
                    "baseline": r.baseline_value,
                    "current": r.current_value,
                    "delta": r.delta,
                    "delta_percent": r.delta_percent,
                    "threshold": r.threshold,
                    "regressed": r.regressed,
                    "category": r.category.value,
                }
                for r in self.results
            ],
        }


class RegressionDetector:
    """Detects quality regression by comparing metrics against baselines.

    Stores baseline metric snapshots after each validated deployment,
    and compares current metrics against the stored baseline to detect
    degradation.

    Example::

        detector = RegressionDetector()

        # After a validated deployment, store the baseline
        detector.save_baseline(
            metrics={"precision": 0.85, "recall": 0.80, "ndcg": 0.82},
            pipeline_config={"reranker": "flashrank", "chunk_size": 512},
        )

        # Before next deployment, check for regression
        report = detector.check_regression(
            current_metrics={"precision": 0.83, "recall": 0.75, "ndcg": 0.80},
        )
        if report.has_regression:
            print(report.summary)
    """

    DEFAULT_BASELINE_DIR = ".tldw/baselines"

    def __init__(
        self,
        baseline_dir: Path | str | None = None,
        default_threshold: float = 0.05,
        lower_is_better: set[str] | None = None,
        gating_config: GatingConfig | None = None,
        use_quality_gating: bool = True,
    ) -> None:
        """Initialize RegressionDetector.

        Args:
            baseline_dir: Directory for storing baselines.
            default_threshold: Default regression threshold (fractional).
                A metric is flagged as regressed if it degrades by more
                than this fraction of the baseline value.
            lower_is_better: Set of metric names where lower values are better.
        """
        self.baseline_dir = Path(baseline_dir or self.DEFAULT_BASELINE_DIR)
        self._dir_ensured = False
        self.default_threshold = default_threshold
        self.lower_is_better = lower_is_better or {"hallucination", "latency_p99_ms"}
        self.use_quality_gating = use_quality_gating
        self.gating_config = gating_config

    def _ensure_dir(self) -> None:
        """Create the baseline directory on first disk access."""
        if not self._dir_ensured:
            self.baseline_dir.mkdir(parents=True, exist_ok=True)
            self._dir_ensured = True

    def save_baseline(
        self,
        metrics: dict[str, float],
        pipeline_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        baseline_id: str | None = None,
    ) -> MetricBaseline:
        """Save a metric baseline snapshot.

        Args:
            metrics: Metric name -> value mapping.
            pipeline_config: Optional pipeline configuration snapshot.
            metadata: Optional additional metadata.
            baseline_id: Optional custom ID. Defaults to "latest".

        Returns:
            The saved MetricBaseline.
        """
        bid = baseline_id or "latest"
        now = datetime.now(timezone.utc).isoformat()

        baseline = MetricBaseline(
            baseline_id=bid,
            created_at=now,
            pipeline_config=pipeline_config or {},
            metrics=metrics,
            metadata=metadata or {},
        )

        self._save_atomic(baseline)
        logger.info(f"Saved metric baseline '{bid}' with {len(metrics)} metrics")
        return baseline

    def load_baseline(self, baseline_id: str = "latest") -> MetricBaseline | None:
        """Load a stored baseline.

        Args:
            baseline_id: Baseline identifier. Defaults to "latest".

        Returns:
            MetricBaseline if found, None otherwise.
        """
        path = self._get_baseline_path(baseline_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return MetricBaseline(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Failed to load baseline: {}", type(e).__name__)
            return None

    def check_regression(
        self,
        current_metrics: dict[str, float],
        baseline_id: str = "latest",
        thresholds: dict[str, float] | None = None,
    ) -> RegressionReport:
        """Check current metrics against a stored baseline.

        Args:
            current_metrics: Current metric values.
            baseline_id: Baseline to compare against.
            thresholds: Per-metric thresholds (fractional). Falls back to
                default_threshold for any metric not specified.

        Returns:
            RegressionReport with per-metric results.
        """
        baseline = self.load_baseline(baseline_id)
        if baseline is None:
            return RegressionReport(
                baseline_id=baseline_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                results=[],
                has_regression=False,
                summary=f"No baseline '{baseline_id}' found. Skipping regression check.",
            )

        thresholds = thresholds or {}
        gating_config: GatingConfig | None
        gating_config = self.gating_config or GatingConfig() if self.use_quality_gating else None

        lower_is_better = set(self.lower_is_better)
        if gating_config:
            lower_is_better.update(gating_config.lower_is_better)
        results: list[RegressionResult] = []
        has_regression = False
        has_warnings = False

        for name, current_value in current_metrics.items():
            if name not in baseline.metrics:
                continue

            baseline_value = baseline.metrics[name]
            threshold = thresholds.get(name, self.default_threshold)

            # Calculate delta
            delta = current_value - baseline_value

            if gating_config and name in gating_config.unstable:
                category = MetricCategory.UNSTABLE
            elif gating_config and name in gating_config.stable:
                category = MetricCategory.STABLE
            else:
                category = MetricCategory.STABLE

            # For "lower is better" metrics, a positive delta is bad
            # For "higher is better" metrics, a negative delta is bad
            if name in lower_is_better:
                # Higher current = worse
                regressed = delta > abs(baseline_value * threshold) if baseline_value != 0 else delta > 0
            else:
                # Lower current = worse
                regressed = (-delta) > abs(baseline_value * threshold) if baseline_value != 0 else delta < 0

            if baseline_value != 0:
                delta_percent = delta / baseline_value * 100
            elif delta != 0:
                # Baseline was zero but current is non-zero -- report as 100% change
                delta_percent = 100.0 if delta > 0 else -100.0
            else:
                delta_percent = 0.0

            results.append(
                RegressionResult(
                    metric_name=name,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    delta=delta,
                    delta_percent=delta_percent,
                    threshold=threshold,
                    regressed=regressed,
                    category=category,
                )
            )

            if regressed:
                if category == MetricCategory.UNSTABLE:
                    has_warnings = True
                else:
                    has_regression = True

        # Build summary
        if has_regression:
            regressed_names = [
                r.metric_name for r in results
                if r.regressed and r.category == MetricCategory.STABLE
            ]
            summary = f"Regression detected in stable metrics: {', '.join(regressed_names)}"
            if has_warnings:
                warn_names = [
                    r.metric_name for r in results
                    if r.regressed and r.category == MetricCategory.UNSTABLE
                ]
                summary += f". Warnings in unstable metrics: {', '.join(warn_names)}"
        elif has_warnings:
            warn_names = [
                r.metric_name for r in results
                if r.regressed and r.category == MetricCategory.UNSTABLE
            ]
            summary = (
                "Regression detected in unstable metrics (warning only): "
                + ", ".join(warn_names)
            )
        elif not results:
            summary = "No comparable metrics found between current and baseline."
        else:
            summary = "No regression detected. All metrics within threshold."

        return RegressionReport(
            baseline_id=baseline_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            results=results,
            has_regression=has_regression,
            has_warnings=has_warnings,
            summary=summary,
        )

    def list_baselines(self) -> list[MetricBaseline]:
        """List all stored baselines.

        Returns:
            List of baselines, sorted by creation time (newest first).
        """
        baselines: list[MetricBaseline] = []
        for path in self.baseline_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                baselines.append(MetricBaseline(**data))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        return sorted(baselines, key=lambda b: b.created_at, reverse=True)

    def _get_baseline_path(self, baseline_id: str) -> Path:
        """Get the file path for a baseline ID.

        Raises:
            ValueError: If baseline_id contains unsafe characters.
        """
        if not _SAFE_ID_RE.match(baseline_id):
            raise ValueError(f"Invalid baseline ID: {baseline_id!r}")
        return self.baseline_dir / f"{baseline_id}.json"

    def _save_atomic(self, baseline: MetricBaseline) -> None:
        """Save baseline atomically using temp file + rename."""
        self._ensure_dir()
        target_path = self._get_baseline_path(baseline.baseline_id)

        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="baseline_",
            dir=self.baseline_dir,
        )

        try:
            with os.fdopen(fd, "w") as f:
                json.dump(baseline.model_dump(), f, indent=2)
            Path(temp_path).replace(target_path)
        except Exception:
            temp_path_obj = Path(temp_path)
            if temp_path_obj.exists():
                temp_path_obj.unlink()
            raise
