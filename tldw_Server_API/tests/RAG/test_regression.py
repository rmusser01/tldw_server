"""
Tests for the RAG regression detection module.

Covers:
- MetricBaseline, RegressionResult, RegressionReport Pydantic models
- RegressionDetector save/load/check/list operations
- Path traversal rejection and invalid ID handling
- Lower-is-better metric semantics
- Division by zero edge case for delta_percent
- Atomic save behavior
- Report serialization via to_dict()
"""

import json
import time
from pathlib import Path

import pytest

from tldw_Server_API.app.core.RAG.rag_service import regression as regression_mod
from tldw_Server_API.app.core.RAG.rag_service.regression import (
    _SAFE_ID_RE,
    MetricBaseline,
    RegressionDetector,
    RegressionReport,
    RegressionResult,
)
from tldw_Server_API.app.core.RAG.rag_service.quality_gating import MetricCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(tmp_path: Path, **kwargs) -> RegressionDetector:
    """Create a RegressionDetector rooted in a temporary directory."""
    return RegressionDetector(baseline_dir=tmp_path / "baselines", **kwargs)


class _RecordingLogger:
    def __init__(self):
        self.warning_calls = []

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))


def _assert_warning_logs_are_sanitized(logger: _RecordingLogger):
    assert logger.warning_calls
    for args, kwargs in logger.warning_calls:
        assert not kwargs.get("exc_info")
        rendered = " ".join(str(arg) for arg in args)
        assert "/tmp/source" not in rendered
        assert "token=secret" not in rendered


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMetricBaseline:
    """Tests for the MetricBaseline frozen Pydantic model."""

    def test_creation(self):
        b = MetricBaseline(
            baseline_id="test-1",
            created_at="2025-01-01T00:00:00+00:00",
            metrics={"precision": 0.9},
        )
        assert b.baseline_id == "test-1"
        assert b.metrics == {"precision": 0.9}
        assert b.pipeline_config == {}
        assert b.metadata == {}

    def test_frozen(self):
        b = MetricBaseline(
            baseline_id="test-2",
            created_at="2025-01-01T00:00:00+00:00",
            metrics={"recall": 0.8},
        )
        with pytest.raises(Exception):
            b.baseline_id = "new-id"


@pytest.mark.unit
class TestRegressionResult:
    """Tests for the RegressionResult frozen Pydantic model."""

    def test_creation(self):
        r = RegressionResult(
            metric_name="precision",
            baseline_value=0.90,
            current_value=0.85,
            delta=-0.05,
            delta_percent=-5.56,
            threshold=0.05,
            regressed=True,
        )
        assert r.metric_name == "precision"
        assert r.regressed is True

    def test_frozen(self):
        r = RegressionResult(
            metric_name="recall",
            baseline_value=0.80,
            current_value=0.80,
            delta=0.0,
            delta_percent=0.0,
            threshold=0.05,
            regressed=False,
        )
        with pytest.raises(Exception):
            r.regressed = True


# ---------------------------------------------------------------------------
# RegressionReport tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRegressionReport:
    """Tests for RegressionReport model and its to_dict() method."""

    def _sample_report(self) -> RegressionReport:
        results = [
            RegressionResult(
                metric_name="precision",
                baseline_value=0.90,
                current_value=0.88,
                delta=-0.02,
                delta_percent=-2.22,
                threshold=0.05,
                regressed=False,
                category=MetricCategory.STABLE,
            ),
            RegressionResult(
                metric_name="recall",
                baseline_value=0.80,
                current_value=0.70,
                delta=-0.10,
                delta_percent=-12.5,
                threshold=0.05,
                regressed=True,
                category=MetricCategory.STABLE,
            ),
        ]
        return RegressionReport(
            baseline_id="v1",
            timestamp="2025-06-01T00:00:00+00:00",
            results=results,
            has_regression=True,
            has_warnings=False,
            summary="Regression detected in stable metrics: recall",
        )

    def test_to_dict_structure(self):
        """to_dict() should produce the expected keys and nested structure."""
        d = self._sample_report().to_dict()
        assert set(d.keys()) == {
            "baseline_id",
            "timestamp",
            "has_regression",
            "has_warnings",
            "summary",
            "results",
        }
        assert isinstance(d["results"], list)
        assert len(d["results"]) == 2

        result_keys = set(d["results"][0].keys())
        assert result_keys == {
            "metric",
            "baseline",
            "current",
            "delta",
            "delta_percent",
            "threshold",
            "regressed",
            "category",
        }

    def test_to_dict_values(self):
        """to_dict() values should reflect the underlying model data."""
        d = self._sample_report().to_dict()
        assert d["baseline_id"] == "v1"
        assert d["has_regression"] is True
        assert d["has_warnings"] is False

        recall_entry = next(r for r in d["results"] if r["metric"] == "recall")
        assert recall_entry["regressed"] is True
        assert recall_entry["baseline"] == 0.80
        assert recall_entry["current"] == 0.70
        assert recall_entry["category"] == "stable"

    def test_to_dict_is_json_serializable(self):
        """to_dict() output must be JSON serializable."""
        d = self._sample_report().to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# _SAFE_ID_RE tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSafeIdRegex:
    """Validate the _SAFE_ID_RE pattern for baseline IDs."""

    @pytest.mark.parametrize("valid_id", [
        "latest",
        "v1.0",
        "baseline_2025-01-01",
        "A.B_C-D",
        "123",
    ])
    def test_valid_ids(self, valid_id: str):
        assert _SAFE_ID_RE.match(valid_id) is not None

    @pytest.mark.parametrize("invalid_id", [
        "../../etc/passwd",
        "foo/bar",
        "foo bar",
        "",
        "id\x00null",
    ])
    def test_invalid_ids(self, invalid_id: str):
        assert _SAFE_ID_RE.match(invalid_id) is None


# ---------------------------------------------------------------------------
# RegressionDetector tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRegressionDetectorSaveLoad:
    """Save and load baseline roundtrip tests."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Saving a baseline and loading it back should return identical data."""
        detector = _make_detector(tmp_path)
        metrics = {"precision": 0.90, "recall": 0.80, "ndcg": 0.82}
        config = {"reranker": "flashrank", "chunk_size": 512}
        meta = {"dataset": "test-set-1"}

        saved = detector.save_baseline(
            metrics=metrics,
            pipeline_config=config,
            metadata=meta,
            baseline_id="v1",
        )
        loaded = detector.load_baseline("v1")

        assert loaded is not None
        assert loaded.baseline_id == saved.baseline_id
        assert loaded.created_at == saved.created_at
        assert loaded.metrics == metrics
        assert loaded.pipeline_config == config
        assert loaded.metadata == meta

    def test_save_default_id_is_latest(self, tmp_path: Path):
        """When no baseline_id is given, the default 'latest' should be used."""
        detector = _make_detector(tmp_path)
        saved = detector.save_baseline(metrics={"precision": 0.9})
        assert saved.baseline_id == "latest"

        loaded = detector.load_baseline("latest")
        assert loaded is not None
        assert loaded.metrics == {"precision": 0.9}

    def test_load_missing_baseline_returns_none(self, tmp_path: Path):
        """Loading a non-existent baseline should return None."""
        detector = _make_detector(tmp_path)
        assert detector.load_baseline("nonexistent") is None

    def test_load_invalid_baseline_log_is_sanitized(self, tmp_path: Path, monkeypatch):
        """Malformed baseline payloads return None without logging raw payload details."""
        detector = _make_detector(tmp_path)
        detector.baseline_dir.mkdir(parents=True)
        (detector.baseline_dir / "latest.json").write_text(
            json.dumps(
                {
                    "baseline_id": "latest",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "metrics": "invalid metrics /tmp/source token=secret",
                }
            )
        )
        recording_logger = _RecordingLogger()
        monkeypatch.setattr(regression_mod, "logger", recording_logger)

        assert detector.load_baseline("latest") is None
        _assert_warning_logs_are_sanitized(recording_logger)


@pytest.mark.unit
class TestRegressionDetectorCheckRegression:
    """Tests for check_regression behavior."""

    def test_no_regression_all_within_threshold(self, tmp_path: Path):
        """When all metrics stay within threshold, has_regression should be False."""
        detector = _make_detector(tmp_path, default_threshold=0.10)
        detector.save_baseline(
            metrics={"precision": 0.90, "recall": 0.80},
            baseline_id="latest",
        )

        report = detector.check_regression(
            current_metrics={"precision": 0.88, "recall": 0.78},
        )
        assert report.has_regression is False
        assert len(report.results) == 2
        assert all(not r.regressed for r in report.results)
        assert "No regression" in report.summary

    def test_regression_detected_beyond_threshold(self, tmp_path: Path):
        """When a metric degrades beyond threshold, it should be flagged."""
        detector = _make_detector(tmp_path, default_threshold=0.05)
        detector.save_baseline(
            metrics={"precision": 0.90, "recall": 0.80},
            baseline_id="latest",
        )

        # precision drops from 0.90 to 0.70 = ~22% decline, well beyond 5%
        report = detector.check_regression(
            current_metrics={"precision": 0.70, "recall": 0.79},
        )
        assert report.has_regression is True

        precision_result = next(
            r for r in report.results if r.metric_name == "precision"
        )
        assert precision_result.regressed is True
        assert precision_result.delta < 0

        recall_result = next(
            r for r in report.results if r.metric_name == "recall"
        )
        assert recall_result.regressed is False

    def test_unstable_metric_regression_is_warning(self, tmp_path: Path):
        """Unstable metric regressions should be reported as warnings."""
        detector = _make_detector(tmp_path, default_threshold=0.05)
        detector.save_baseline(
            metrics={"faithfulness": 0.90},
            baseline_id="latest",
        )

        report = detector.check_regression(
            current_metrics={"faithfulness": 0.60},
        )
        assert report.has_regression is False
        assert report.has_warnings is True

        result = report.results[0]
        assert result.metric_name == "faithfulness"
        assert result.regressed is True
        assert result.category == MetricCategory.UNSTABLE
        assert "warning" in report.summary.lower()

    def test_missing_baseline_no_regression(self, tmp_path: Path):
        """With no stored baseline, check_regression returns a benign report."""
        detector = _make_detector(tmp_path)
        report = detector.check_regression(
            current_metrics={"precision": 0.85},
            baseline_id="nonexistent",
        )
        assert report.has_regression is False
        assert report.results == []
        assert "No baseline" in report.summary

    def test_custom_per_metric_thresholds(self, tmp_path: Path):
        """Per-metric thresholds should override the default threshold."""
        detector = _make_detector(tmp_path, default_threshold=0.01)
        detector.save_baseline(
            metrics={"precision": 1.0, "recall": 1.0},
            baseline_id="latest",
        )

        # precision drops by 0.20 (20%), recall drops by 0.20 (20%)
        # threshold for precision = 0.50 (generous), recall = 0.01 (strict)
        report = detector.check_regression(
            current_metrics={"precision": 0.80, "recall": 0.80},
            thresholds={"precision": 0.50, "recall": 0.01},
        )
        precision_result = next(
            r for r in report.results if r.metric_name == "precision"
        )
        recall_result = next(
            r for r in report.results if r.metric_name == "recall"
        )

        assert precision_result.regressed is False  # 20% < 50% threshold
        assert recall_result.regressed is True  # 20% > 1% threshold

    def test_no_comparable_metrics(self, tmp_path: Path):
        """When current metrics have no overlap with baseline, report is benign."""
        detector = _make_detector(tmp_path)
        detector.save_baseline(
            metrics={"precision": 0.9},
            baseline_id="latest",
        )
        report = detector.check_regression(
            current_metrics={"f1_score": 0.85},
        )
        assert report.has_regression is False
        assert len(report.results) == 0
        assert "No comparable metrics" in report.summary


@pytest.mark.unit
class TestLowerIsBetterMetrics:
    """Tests for metrics where lower values indicate better performance."""

    def test_lower_is_better_no_regression(self, tmp_path: Path):
        """When a lower-is-better metric decreases, that is an improvement."""
        detector = _make_detector(
            tmp_path,
            default_threshold=0.10,
            lower_is_better={"latency_p99_ms"},
        )
        detector.save_baseline(
            metrics={"latency_p99_ms": 200.0},
            baseline_id="latest",
        )

        # latency dropped from 200 to 150 => improvement
        report = detector.check_regression(
            current_metrics={"latency_p99_ms": 150.0},
        )
        assert report.has_regression is False

    def test_lower_is_better_regression(self, tmp_path: Path):
        """When a lower-is-better metric increases significantly, it regresses."""
        detector = _make_detector(
            tmp_path,
            default_threshold=0.05,
            lower_is_better={"latency_p99_ms"},
        )
        detector.save_baseline(
            metrics={"latency_p99_ms": 200.0},
            baseline_id="latest",
        )

        # latency increased from 200 to 300 => 50% increase, regression
        report = detector.check_regression(
            current_metrics={"latency_p99_ms": 300.0},
        )
        assert report.has_regression is True
        result = report.results[0]
        assert result.regressed is True
        assert result.delta > 0  # positive delta = worse for lower-is-better

    def test_default_lower_is_better_set(self, tmp_path: Path):
        """The default lower_is_better set includes 'latency_p99_ms' and 'hallucination'."""
        detector = _make_detector(tmp_path)
        assert "latency_p99_ms" in detector.lower_is_better
        assert "hallucination" in detector.lower_is_better


@pytest.mark.unit
class TestPathTraversalAndInvalidIDs:
    """Tests for path traversal rejection and invalid baseline IDs."""

    def test_path_traversal_rejection(self, tmp_path: Path):
        """_get_baseline_path must reject IDs containing path separators."""
        detector = _make_detector(tmp_path)
        with pytest.raises(ValueError, match="Invalid baseline ID"):
            detector._get_baseline_path("../../etc/passwd")

    def test_slash_in_id_rejected(self, tmp_path: Path):
        detector = _make_detector(tmp_path)
        with pytest.raises(ValueError, match="Invalid baseline ID"):
            detector._get_baseline_path("foo/bar")

    def test_space_in_id_rejected(self, tmp_path: Path):
        detector = _make_detector(tmp_path)
        with pytest.raises(ValueError, match="Invalid baseline ID"):
            detector._get_baseline_path("foo bar")

    def test_save_baseline_with_invalid_id_raises(self, tmp_path: Path):
        """save_baseline should raise ValueError for unsafe baseline IDs."""
        detector = _make_detector(tmp_path)
        with pytest.raises(ValueError, match="Invalid baseline ID"):
            detector.save_baseline(
                metrics={"precision": 0.9},
                baseline_id="../../etc/passwd",
            )

    def test_load_baseline_with_invalid_id_raises(self, tmp_path: Path):
        """load_baseline should raise ValueError for unsafe baseline IDs."""
        detector = _make_detector(tmp_path)
        with pytest.raises(ValueError, match="Invalid baseline ID"):
            detector.load_baseline("../malicious")

    def test_valid_id_with_dots_and_dashes(self, tmp_path: Path):
        """IDs with dots and dashes (but no slashes) should be accepted."""
        detector = _make_detector(tmp_path)
        path = detector._get_baseline_path("baseline-2025.01.01_v2")
        assert path.name == "baseline-2025.01.01_v2.json"


@pytest.mark.unit
class TestListBaselines:
    """Tests for list_baselines ordering and content."""

    def test_list_baselines_returns_sorted_by_created_at(self, tmp_path: Path):
        """Baselines should be returned sorted by created_at, newest first."""
        detector = _make_detector(tmp_path)

        # Save three baselines with distinct timestamps (slight sleep not needed;
        # we rely on sequential datetime.now() being monotonically increasing).
        b1 = detector.save_baseline(metrics={"p": 0.80}, baseline_id="first")
        # Ensure distinct timestamps by a tiny sleep
        time.sleep(0.01)
        b2 = detector.save_baseline(metrics={"p": 0.85}, baseline_id="second")
        time.sleep(0.01)
        b3 = detector.save_baseline(metrics={"p": 0.90}, baseline_id="third")

        baselines = detector.list_baselines()
        assert len(baselines) == 3
        # Newest first
        assert baselines[0].baseline_id == "third"
        assert baselines[1].baseline_id == "second"
        assert baselines[2].baseline_id == "first"

    def test_list_baselines_empty_dir(self, tmp_path: Path):
        """An empty baseline directory should return an empty list."""
        detector = _make_detector(tmp_path)
        assert detector.list_baselines() == []

    def test_list_baselines_skips_corrupt_files(self, tmp_path: Path):
        """Corrupt JSON files should be silently skipped."""
        detector = _make_detector(tmp_path)
        detector.save_baseline(metrics={"p": 0.9}, baseline_id="good")

        # Write a corrupt JSON file into the baselines directory
        corrupt_path = detector.baseline_dir / "corrupt.json"
        corrupt_path.write_text("{invalid json content!!!")

        baselines = detector.list_baselines()
        assert len(baselines) == 1
        assert baselines[0].baseline_id == "good"


@pytest.mark.unit
class TestDivisionByZeroEdgeCase:
    """Test the edge case where baseline_value is 0."""

    def test_baseline_zero_current_nonzero_delta_percent(self, tmp_path: Path):
        """When baseline_value=0 and current_value=0.5, delta_percent should be 100.0."""
        detector = _make_detector(tmp_path, default_threshold=0.05)
        detector.save_baseline(
            metrics={"new_metric": 0.0},
            baseline_id="latest",
        )

        report = detector.check_regression(
            current_metrics={"new_metric": 0.5},
        )
        assert len(report.results) == 1
        result = report.results[0]
        assert result.baseline_value == 0.0
        assert result.current_value == 0.5
        assert result.delta_percent == 100.0

    def test_baseline_zero_current_zero_delta_percent(self, tmp_path: Path):
        """When both baseline and current are 0, delta_percent should be 0.0."""
        detector = _make_detector(tmp_path, default_threshold=0.05)
        detector.save_baseline(
            metrics={"stable_metric": 0.0},
            baseline_id="latest",
        )

        report = detector.check_regression(
            current_metrics={"stable_metric": 0.0},
        )
        result = report.results[0]
        assert result.delta == 0.0
        assert result.delta_percent == 0.0
        assert result.regressed is False

    def test_baseline_zero_current_negative_delta_percent(self, tmp_path: Path):
        """When baseline_value=0 and current_value=-0.5, delta_percent should be -100.0."""
        detector = _make_detector(tmp_path, default_threshold=0.05)
        detector.save_baseline(
            metrics={"weird_metric": 0.0},
            baseline_id="latest",
        )

        report = detector.check_regression(
            current_metrics={"weird_metric": -0.5},
        )
        result = report.results[0]
        assert result.delta_percent == -100.0


@pytest.mark.unit
class TestAtomicSave:
    """Tests for the atomic save mechanism."""

    def test_save_creates_json_file(self, tmp_path: Path):
        """save_baseline should create a .json file in the baseline directory."""
        detector = _make_detector(tmp_path)
        detector.save_baseline(metrics={"p": 0.9}, baseline_id="atomic-test")

        expected_path = detector.baseline_dir / "atomic-test.json"
        assert expected_path.exists()

        data = json.loads(expected_path.read_text())
        assert data["baseline_id"] == "atomic-test"
        assert data["metrics"]["p"] == 0.9

    def test_save_overwrites_existing_baseline(self, tmp_path: Path):
        """Saving with the same ID should overwrite the previous baseline."""
        detector = _make_detector(tmp_path)
        detector.save_baseline(metrics={"p": 0.8}, baseline_id="overwrite-test")
        detector.save_baseline(metrics={"p": 0.95}, baseline_id="overwrite-test")

        loaded = detector.load_baseline("overwrite-test")
        assert loaded is not None
        assert loaded.metrics["p"] == 0.95

    def test_no_temp_files_left_after_save(self, tmp_path: Path):
        """After a successful save, no temporary files should remain."""
        detector = _make_detector(tmp_path)
        detector.save_baseline(metrics={"p": 0.9}, baseline_id="clean-test")

        tmp_files = list(detector.baseline_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


@pytest.mark.unit
class TestRegressionDetectorInitialization:
    """Tests for detector initialization behavior."""

    def test_creates_baseline_dir_on_first_use(self, tmp_path: Path):
        """The baseline directory should be created lazily on first disk access."""
        baseline_dir = tmp_path / "new" / "nested" / "baselines"
        assert not baseline_dir.exists()
        detector = RegressionDetector(baseline_dir=baseline_dir)
        # Directory is NOT created eagerly in __init__ to avoid side-effects
        # when constructing detectors per-request in health endpoints.
        assert not baseline_dir.exists()
        # Saving a baseline triggers directory creation.
        detector.save_baseline(metrics={"p": 0.9}, baseline_id="test")
        assert baseline_dir.exists()

    def test_custom_default_threshold(self, tmp_path: Path):
        """The default_threshold parameter should be stored correctly."""
        detector = RegressionDetector(
            baseline_dir=tmp_path / "baselines",
            default_threshold=0.10,
        )
        assert detector.default_threshold == 0.10

    def test_custom_lower_is_better(self, tmp_path: Path):
        """A custom lower_is_better set should override the default."""
        detector = RegressionDetector(
            baseline_dir=tmp_path / "baselines",
            lower_is_better={"error_rate", "latency"},
        )
        assert detector.lower_is_better == {"error_rate", "latency"}
