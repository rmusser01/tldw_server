# Web Scraping Article Extraction Benchmark

This document describes how to run the [Scrapinghub article extraction benchmark](https://github.com/scrapinghub/article-extraction-benchmark)
against the `tldw_server` web scraping pipeline.

## Overview

The benchmark provides 1,000+ frozen HTML snapshots and reference article bodies. We reuse the
benchmark's evaluation methodology (token shingles with bootstrap resampling) to score the
pipeline's extraction quality. Results are reported as F1, precision, recall, and exact-match
accuracy with confidence intervals.

A dedicated evaluator lives in `tldw_Server_API/app/core/Evaluations/article_extraction_benchmark.py`
and a convenience script is available at `Helper_Scripts/Evals/run_article_extraction_benchmark.py`.

## Prerequisites

1. Clone the upstream benchmark repository inside the repo (or anywhere accessible):

   ```bash
   git clone https://github.com/scrapinghub/article-extraction-benchmark tmp/article-extraction-benchmark
   ```

2. Ensure Python dependencies for the scraping pipeline are installed (see the main README).

No additional packages are required to compute the metrics.

## Running the Benchmark

Use the helper script to execute the evaluation:

```bash
python Helper_Scripts/Evals/run_article_extraction_benchmark.py \
    tmp/article-extraction-benchmark \
    --bootstrap 500 \
    --save-predictions tmp/article_predictions.json
```

Optional flags:

- `--limit N` &mdash; evaluate only the first `N` pages (useful for quick smoke checks).
- `--bootstrap N` &mdash; adjust the number of bootstrap samples (default: 1000).
- `--save-predictions PATH` &mdash; store the generated article body predictions for inspection.

The script logs progress and finishes with a metric summary. Example output:

```
2025-02-01 12:00:00.000 | INFO  | Benchmark complete - F1: 0.952 (± 0.006), Precision: 0.941 (± 0.008), Recall: 0.964 (± 0.006), Accuracy: 0.305 (± 0.034)
```

## Programmatic Usage

The evaluator can also be imported and used directly:

```python
from pathlib import Path
from tldw_Server_API.app.core.Evaluations.article_extraction_benchmark import (
    ArticleExtractionBenchmarkEvaluator,
)

evaluator = ArticleExtractionBenchmarkEvaluator(Path("tmp/article-extraction-benchmark"))
metrics = evaluator.run(limit=100)
print(metrics.to_dict())
```

The evaluator accepts a custom extraction callback if you want to compare different scraping
strategies. Pass a callable that receives the raw HTML and source URL and returns the extracted
article body.

## Interpreting Scores

- **F1** is the harmonic mean of precision and recall on 4-token shingles.
- **Precision/Recall** measure the overlap between the extracted body and reference text.
- **Accuracy** counts exact token-sequence matches.

Use these metrics to benchmark pipeline tweaks, new extractors, or regression tests across releases.
