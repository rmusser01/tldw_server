# Web Retrieval Quality Baselines

This project has two complementary web-retrieval evaluation paths. The fast checked baseline is a small offline contract and budget snapshot. The full Scrapinghub article benchmark measures extraction quality across an external frozen dataset. They answer different questions and should not be treated as interchangeable release gates.

## Fast offline retrieval baseline

Run the checked four-case baseline from the repository root:

```bash
source .venv/bin/activate
python Helper_Scripts/Evals/run_web_retrieval_quality_baseline.py \
  --json-out /tmp/web_retrieval_quality.json
```

The versioned input is `tldw_Server_API/tests/Web_Scraping/fixtures/retrieval_quality/v1.json`. The checked current-dev report is `Docs/Evals/baselines/web_retrieval_quality_v1.json`. The runner prints a concise stable summary and writes deterministic sorted JSON when `--json-out` is supplied.

The fixture contains one synthetic `.test` case for each shared measurement concern:

- article extraction with navigation text omitted from the observed article body;
- stable ordering of tied results from two search providers;
- an ordered cyclic crawl observation stopped at an explicit page limit;
- bounded retrieval provenance with safe URLs, versions, a fingerprint, selected tier, and truncation fields.

The fast runner validates and scores frozen observations. It does not call a production extractor, search provider, crawler, database, queue, LLM, browser, tokenizer, or public network. Its required pytest coverage is also fully offline.

### Versions and reproducibility

The checked contract uses:

- fixture schema: `web-retrieval-quality-fixture-v1`;
- report schema: `web-retrieval-quality-report-v1`;
- budget algorithm: `char-utf8-budget-v1`;
- crawl algorithm: `ordered-visit-stop-v1`;
- extraction algorithm: `token-shingle-f1-v1`;
- provenance algorithm: `required-field-recall-v1`;
- search-order algorithm: `position-match-v1`;
- token-estimate algorithm: `characters-ceil-div4-v1`.

Reports contain no timestamp, random identifier, host path, environment dump, or implicit Git lookup. Reordering fixture cases does not alter report case order. Incompatible fixture or report changes require a new schema version; scoring changes require a new algorithm version.

The checked `observed` values are a descriptive snapshot of reconciled current-dev behavior. They are not thresholds and do not replace feature behavior tests. Any intentional recapture must be linked to its Backlog task so the reason for the change remains reviewable.

Characters and UTF-8 bytes are the authoritative output budgets. Estimated tokens use `ceil(characters / 4)`, are explicitly marked `authoritative: false`, and add no tokenizer dependency.

## Full Scrapinghub article extraction benchmark

The [Scrapinghub article extraction benchmark](https://github.com/scrapinghub/article-extraction-benchmark) provides more than 1,000 frozen HTML snapshots and reference article bodies. The evaluator reuses its token-shingle methodology and reports F1, precision, recall, and exact token-sequence accuracy with bootstrap standard deviations.

The evaluator lives in `tldw_Server_API/app/core/Evaluations/article_extraction_benchmark.py`; its CLI is `Helper_Scripts/Evals/run_article_extraction_benchmark.py`.

### Prerequisite

Obtain the upstream dataset once, inside the repository or at another accessible path:

```bash
git clone https://github.com/scrapinghub/article-extraction-benchmark tmp/article-extraction-benchmark
```

Cloning requires network access. Benchmark execution itself reads the local frozen snapshots and does not use the public network.

### Run the benchmark reproducibly

```bash
source .venv/bin/activate
python Helper_Scripts/Evals/run_article_extraction_benchmark.py \
  tmp/article-extraction-benchmark \
  --bootstrap 500 \
  --bootstrap-seed 0 \
  --save-predictions tmp/article_predictions.json
```

Options:

- `--limit N` evaluates only the first `N` pages.
- `--bootstrap N` sets the positive bootstrap sample count; the default is 1000.
- `--bootstrap-seed N` selects the deterministic local bootstrap seed; the default is 0.
- `--save-predictions PATH` writes extracted article bodies for inspection.

Bootstrap sampling uses a local seeded random generator and never mutates Python's module-global random state. Invalid sample counts and non-integer seeds fail with a clear validation error.

### Programmatic usage

```python
from pathlib import Path

from tldw_Server_API.app.core.Evaluations.article_extraction_benchmark import (
    ArticleExtractionBenchmarkEvaluator,
)

evaluator = ArticleExtractionBenchmarkEvaluator(
    Path("tmp/article-extraction-benchmark"),
    n_bootstrap=500,
    bootstrap_seed=0,
)
metrics = evaluator.run(limit=100)
print(metrics.to_dict())
```

The evaluator also accepts a custom extraction callback receiving raw HTML and the source URL.

## Scope intentionally parked

This baseline does not add thresholds, feature-specific production adapters, fuzzing, soak infrastructure, PDF evaluation, or an external comparator. Those follow-up slices remain parked under `Docs/superpowers/specs/2026-08-27-agent-native-web-research-quality-provenance-roadmap.md` and their corresponding Backlog tasks.
