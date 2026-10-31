#!/usr/bin/env python3
"""Run the Scrapinghub article extraction benchmark against the web scraper."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.Evaluations.article_extraction_benchmark import (
    ArticleExtractionBenchmarkEvaluator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the web scraping pipeline using the Scrapinghub article extraction benchmark.",
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the cloned article-extraction-benchmark repository.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of documents to evaluate.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals (default: 1000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Deterministic bootstrap seed (default: 0).",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional path to store the generated predictions JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = ArticleExtractionBenchmarkEvaluator(
        dataset_root=args.dataset_root,
        n_bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    metrics = evaluator.run(
        limit=args.limit,
        output_predictions_path=args.save_predictions,
    )

    logger.info("Final benchmark metrics:")
    for key, value in metrics.to_dict().items():
        logger.info("{key}: {value:.4f}", key=key, value=value)


if __name__ == "__main__":
    main()
