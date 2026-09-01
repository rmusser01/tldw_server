"""Article extraction benchmark integration for the web scraping pipeline.

This module allows running the Scrapinghub article extraction benchmark
(https://github.com/scrapinghub/article-extraction-benchmark) against the
project's web scraping pipeline.  The evaluation logic for computing metrics is
adapted from the upstream MIT-licensed project, with minor modifications to fit
our infrastructure.
"""

from __future__ import annotations

import gzip
import json
import random
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.Web_Scraping.content import ContentMetadataHandler
from tldw_Server_API.app.core.Web_Scraping.extraction import extract_article_data_from_html

TP_FP_FN = tuple[float, float, float]


@dataclass
class BenchmarkMetrics:
    """Container for benchmark summary statistics."""

    f1: float
    precision: float
    recall: float
    accuracy: float
    f1_std: float
    precision_std: float
    recall_std: float
    accuracy_std: float

    def to_dict(self) -> dict[str, float]:
        return {
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "f1_std": self.f1_std,
            "precision_std": self.precision_std,
            "recall_std": self.recall_std,
            "accuracy_std": self.accuracy_std,
        }


class ArticleExtractionBenchmarkEvaluator:
    """Evaluate the web scraping pipeline using the Scrapinghub benchmark."""

    def __init__(
        self,
        dataset_root: Path,
        extractor: Callable[[str, str], str] | None = None,
        n_bootstrap: int = 1000,
        bootstrap_seed: int = 0,
    ) -> None:
        if type(n_bootstrap) is not int or n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be a positive integer")
        if type(bootstrap_seed) is not int:
            raise ValueError("bootstrap_seed must be an integer")

        self.dataset_root = Path(dataset_root)
        self.html_dir = self.dataset_root / "html"
        self.ground_truth_path = self.dataset_root / "ground-truth.json"
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seed = bootstrap_seed
        self.extractor = extractor or self._default_extractor

        if not self.html_dir.exists():
            raise FileNotFoundError(
                f"HTML snapshot directory not found at {self.html_dir}. "
                "Ensure the article-extraction-benchmark repository is checked out and available."
            )
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found at {self.ground_truth_path}."
            )

    def run(
        self,
        limit: int | None = None,
        output_predictions_path: Path | None = None,
    ) -> BenchmarkMetrics:
        full_ground_truth = self._load_ground_truth()
        predictions: dict[str, dict[str, str]] = {}

        item_ids = list(full_ground_truth.keys())
        if limit is not None:
            item_ids = item_ids[:limit]
            logger.info(
                "Running benchmark on {subset}/{total} documents (limit applied)",
                subset=len(item_ids),
                total=len(full_ground_truth),
            )
        else:
            logger.info("Running benchmark on {count} documents", count=len(item_ids))

        ground_truth = {key: full_ground_truth[key] for key in item_ids}

        for idx, item_id in enumerate(item_ids, start=1):
            truth = ground_truth[item_id]
            html = self._load_html(item_id)
            article_body = self.extractor(html, truth.get("url", ""))
            predictions[item_id] = {
                "articleBody": article_body,
                "url": truth.get("url", ""),
            }

            if idx % 25 == 0 or idx == len(item_ids):
                logger.info("Processed {done}/{total} documents", done=idx, total=len(item_ids))

        if output_predictions_path:
            output_predictions_path.write_text(
                json.dumps(predictions, indent=2, ensure_ascii=False)
            )
            logger.info("Saved predictions to {path}", path=output_predictions_path)

        metrics = evaluate_metrics(
            ground_truth,
            predictions,
            self.n_bootstrap,
            bootstrap_seed=self.bootstrap_seed,
        )
        logger.info(
            (
                "Benchmark complete - F1: {f1:.3f} (± {f1_std:.3f}), "
                "Precision: {precision:.3f} (± {precision_std:.3f}), "
                "Recall: {recall:.3f} (± {recall_std:.3f}), "
                "Accuracy: {accuracy:.3f} (± {accuracy_std:.3f})"
            ),
            **metrics,
        )

        return BenchmarkMetrics(
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            accuracy=metrics["accuracy"],
            f1_std=metrics["f1_std"],
            precision_std=metrics["precision_std"],
            recall_std=metrics["recall_std"],
            accuracy_std=metrics["accuracy_std"],
        )

    def _load_ground_truth(self) -> dict[str, dict[str, str]]:
        with self.ground_truth_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_html(self, item_id: str) -> str:
        html_path = self.html_dir / f"{item_id}.html.gz"
        if not html_path.exists():
            raise FileNotFoundError(f"HTML snapshot missing for item {item_id}: {html_path}")
        with gzip.open(html_path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()

    @staticmethod
    def _default_extractor(html: str, url: str) -> str:
        """Use the built-in Trafilatura-based pipeline for article extraction."""
        extraction = extract_article_data_from_html(html, url or "")
        content = extraction.get("content", "")
        if not content:
            return ""
        _, clean_content = ContentMetadataHandler.extract_metadata(content)
        return clean_content


# --- Evaluation helpers ---------------------------------------------------
# The functions below are adapted from the upstream article-extraction-
# benchmark project (MIT License).


def evaluate_metrics(
    ground_truth: dict[str, dict[str, str]],
    prediction: dict[str, dict[str, str]],
    n_bootstrap: int,
    *,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if type(n_bootstrap) is not int or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be an integer")
    if ground_truth.keys() != prediction.keys():
        raise ValueError("Prediction keys do not match ground truth")

    tp_fp_fns: list[TP_FP_FN] = []
    accuracies: list[float] = []
    for key in ground_truth:
        true = ground_truth[key].get("articleBody", "")
        pred = prediction[key].get("articleBody", "")
        tp_fp_fns.append(string_shingle_matching(true=true, pred=pred))
        accuracies.append(get_accuracy(true=true, pred=pred))

    metrics: dict[str, Any] = metrics_from_tp_fp_fns(tp_fp_fns)
    metrics["accuracy"] = statistics.mean(accuracies)

    bootstrap_values: dict[str, list[float]] = {}
    n_items = len(tp_fp_fns)
    indices_range = range(n_items)
    # Bootstrap resampling is deterministic evaluation, never security-sensitive.
    rng = random.Random(bootstrap_seed)  # nosec B311
    for _ in range(n_bootstrap):
        indices = [
            rng.randint(0, n_items - 1)
            for _ in indices_range
        ]
        sample_tp_fp_fns = [tp_fp_fns[i] for i in indices]
        sample_metrics = metrics_from_tp_fp_fns(sample_tp_fp_fns)
        for key, value in sample_metrics.items():
            bootstrap_values.setdefault(key, []).append(value)
        bootstrap_values.setdefault("accuracy", []).append(
            statistics.mean([accuracies[i] for i in indices])
        )

    for key, values in bootstrap_values.items():
        if len(values) <= 1:
            metrics[f"{key}_std"] = 0.0
        else:
            metrics[f"{key}_std"] = statistics.stdev(values)

    return metrics


def metrics_from_tp_fp_fns(tp_fp_fns: Iterable[TP_FP_FN]) -> dict[str, float]:
    tp_fp_fns_list = list(tp_fp_fns)
    precision_scores = [
        precision_score(tp, fp, fn)
        for tp, fp, fn in tp_fp_fns_list
        if tp + fp > 0
    ]
    recall_scores = [
        recall_score(tp, fp, fn)
        for tp, fp, fn in tp_fp_fns_list
        if tp + fn > 0
    ]

    precision = statistics.mean(precision_scores) if precision_scores else 0.0
    recall = statistics.mean(recall_scores) if recall_scores else 0.0
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def precision_score(tp: float, fp: float, fn: float) -> float:
    if fp == fn == 0:
        return 1.0
    if tp == fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_score(tp: float, fp: float, fn: float) -> float:
    if fp == fn == 0:
        return 1.0
    if tp == fn == 0:
        return 0.0
    return tp / (tp + fn)


def get_accuracy(true: str, pred: str) -> float:
    return float(_tokenize(true) == _tokenize(pred))


def string_shingle_matching(true: str, pred: str, ngram_n: int = 4) -> TP_FP_FN:
    true_shingles = _all_shingles(true, ngram_n)
    pred_shingles = _all_shingles(pred, ngram_n)
    tp = fp = fn = 0.0
    for key in (set(true_shingles) | set(pred_shingles)):
        true_count = true_shingles.get(key, 0)
        pred_count = pred_shingles.get(key, 0)
        tp += min(true_count, pred_count)
        fp += max(0, pred_count - true_count)
        fn += max(0, true_count - pred_count)
    total = tp + fp + fn
    if total > 0:
        tp, fp, fn = tp / total, fp / total, fn / total
    return tp, fp, fn


def _all_shingles(text: str, ngram_n: int) -> dict[tuple[str, ...], int]:
    return dict(_count_ngrams(_ngrams(text, ngram_n)))


def _ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    if len(tokens) < n:
        return [tuple(tokens)]
    return [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]


def _count_ngrams(ngrams: Iterable[tuple[str, ...]]) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for ngram in ngrams:
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


_TOKEN_RE = re.compile(
    r"\w+",
    re.UNICODE | re.MULTILINE | re.IGNORECASE | re.DOTALL,
)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text)


__all__ = [
    "ArticleExtractionBenchmarkEvaluator",
    "BenchmarkMetrics",
    "evaluate_metrics",
]
