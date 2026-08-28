# Deterministic Web Retrieval Quality Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-13139.1 as a small, offline, deterministic baseline for extraction quality, output efficiency, ordered search results, crawl observations, and retrieval provenance.

**Architecture:** Keep the existing Scrapinghub article benchmark as the full extraction benchmark and make its bootstrap reproducible. Add one standard-library fixture/report module for the fast shared baseline. The fixture contains frozen current-dev observations rather than new production adapters; later feature tasks own the code that produces their observations and their behavior tests. The runner validates and scores the fixture, emits byte-stable JSON plus a stable human summary, and never accesses the public network.

**Tech Stack:** Python 3.10+, standard-library `json`, `random`, `statistics`, and `pathlib`; existing article extraction benchmark helpers; pytest; Loguru only where already used by the full benchmark.

**Spec:** `Docs/superpowers/specs/2026-08-27-agent-native-web-research-quality-provenance-roadmap.md` sections 3, 6, 7, 11-13, and TASK-13139.1.

## Global Constraints

- Work under Backlog task `TASK-13139.1`; set it to In Progress before source edits and keep its implementation notes current.
- This is fixture and measurement infrastructure, not a new extractor, search-fusion implementation, crawler, provenance store, fuzz harness, soak runner, or public-network comparator.
- The fast baseline must run without network access, external datasets, LLMs, tokenizers, services, or optional browser dependencies.
- Characters and UTF-8 bytes are authoritative output budgets. The token value is an explicitly non-authoritative estimate using `ceil(characters / 4)`.
- Reports contain no wall-clock timestamp, random identifier, host path, environment dump, or implicit git lookup. Identical fixture bytes and algorithm versions must produce identical report bytes.
- Frozen `observed` values document the reconciled current-dev baseline; they are not pass/fail thresholds. A feature task must link its Backlog record when intentionally recapturing an observation.
- Algorithm changes require a new algorithm-version string. Incompatible fixture or report changes require a new schema-version string.
- Use TDD: add one focused failing test, run it red, implement the minimum behavior, then run it green before moving on.
- Do not add dependencies or generated lockfile changes.

## File Map

- Modify: `tldw_Server_API/app/core/Evaluations/article_extraction_benchmark.py`
  - Make bootstrap sampling local and seedable; reject invalid bootstrap counts.
- Modify: `Helper_Scripts/Evals/run_article_extraction_benchmark.py`
  - Add `--bootstrap-seed` and pass it to the evaluator.
- Add: `tldw_Server_API/tests/Evaluations/test_article_extraction_benchmark.py`
  - Cover deterministic bootstrap behavior and invalid sample counts.
- Add: `tldw_Server_API/app/core/Evaluations/web_retrieval_quality.py`
  - Validate the versioned fixture and calculate deterministic metrics, budgets, reports, and summaries.
- Add: `Helper_Scripts/Evals/run_web_retrieval_quality_baseline.py`
  - Provide a thin CLI around the pure fixture/report module.
- Add: `tldw_Server_API/tests/Web_Scraping/fixtures/retrieval_quality/v1.json`
  - Store the minimal current-dev offline suite.
- Add: `tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py`
  - Cover schema validation, all four case kinds, Unicode budgets, stable serialization, CLI output, and checked baseline parity.
- Add: `Docs/Evals/baselines/web_retrieval_quality_v1.json`
  - Store the generated machine-readable current-dev report.
- Modify: `Docs/Evals/WebScraping_Article_Benchmark.md`
  - Distinguish the fast shared baseline from the external full extraction benchmark and document reproducibility/versioning.
- Modify: `Docs/Published/Evaluations/WebScraping_Article_Benchmark.md`
  - Keep the published documentation mirror synchronized.

## Fixture and Report Contract

The checked fixture has this exact top-level shape:

```json
{
  "schema_version": "web-retrieval-quality-fixture-v1",
  "suite_id": "current-dev-minimal-v1",
  "baseline_revision": "9fd2246157ce8a32ae6a6691a75efab788229f77",
  "cases": []
}
```

Every case has exactly these shared fields:

```json
{
  "id": "unique-stable-id",
  "kind": "extraction",
  "input": {},
  "expected": {},
  "observed": {
    "output_text": "the bounded agent-visible text for budget accounting"
  }
}
```

The four `kind` values and required kind-specific fields are:

| Kind | `input` | `expected` | `observed` |
| --- | --- | --- | --- |
| `extraction` | `url`, `html` | `text` | `text`, `output_text` |
| `search_order` | `provider_results` as ordered `{provider, url, title}` entries | `ordered_urls` | `ordered_urls`, `output_text` |
| `crawl_graph` | `start_url`, `links` mapping URL to an ordered URL list, and integer `page_limit` | `visited_urls`, `stop_reason` | `visited_urls`, `stop_reason`, `output_text` |
| `provenance` | `required_fields` | empty object | `record`, `output_text` |

Validation rules are deliberately small and strict:

- The schema version, suite ID, 40-character lowercase hexadecimal baseline revision, case list, unique non-empty case IDs, and supported kinds are required.
- Case arrays preserve their supplied order. Report cases are sorted by case ID so fixture mapping or case ordering cannot create report drift.
- URLs must be non-empty strings, but this offline schema does not perform DNS or network policy evaluation.
- The crawl graph is data only; the runner validates and scores the frozen current-dev visit observation but does not implement a second crawler.
- `output_text` is required for every observation so character and UTF-8 byte accounting is unambiguous.
- Unknown top-level or case fields are rejected to make accidental schema drift visible.

The report has this exact top-level contract:

```python
{
    "report_schema_version": "web-retrieval-quality-report-v1",
    "fixture_schema_version": "web-retrieval-quality-fixture-v1",
    "suite_id": suite["suite_id"],
    "baseline_revision": suite["baseline_revision"],
    "algorithm_versions": {
        "budget": "char-utf8-budget-v1",
        "crawl": "ordered-visit-stop-v1",
        "extraction": "token-shingle-f1-v1",
        "provenance": "required-field-recall-v1",
        "search_order": "position-match-v1",
        "token_estimate": "characters-ceil-div4-v1",
    },
    "cases": [...],
    "summary": {
        "case_count": 4,
        "mean_case_score": 0.0,
        "total_characters": 0,
        "total_utf8_bytes": 0,
        "estimated_tokens": {
            "value": 0,
            "algorithm": "characters-ceil-div4-v1",
            "authoritative": False,
        },
    },
}
```

Metrics are exact and intentionally uncomplicated:

- Extraction: existing 4-token-shingle precision, recall, and F1 plus token-sequence exact-match accuracy. The case score is F1.
- Search ordering: `position_match_ratio = same_url_at_same_index / max(expected_count, observed_count)`, with non-empty lists required. Also report `exact_order_match`. The case score is the position-match ratio.
- Crawl observation: set recall over expected visited URLs, the same position-match ratio over visit order, and exact stop-reason match. The case score is the arithmetic mean of those three values.
- Provenance: `required_field_recall = present_nonempty_required_fields / required_field_count`, with at least one required field. The case score is that recall.
- Output efficiency: `characters = len(output_text)`, `utf8_bytes = len(output_text.encode("utf-8"))`, and estimated tokens `= (characters + 3) // 4` with `authoritative: false`.
- Summary score: arithmetic mean of the four case scores. Summary estimated tokens use the same formula once over total characters, not a sum of per-case estimates. Round reported floating-point values to six decimal places at the report boundary.

## Task 1: Make the Existing Extraction Benchmark Reproducible

**Files:**

- Add: `tldw_Server_API/tests/Evaluations/test_article_extraction_benchmark.py`
- Modify: `tldw_Server_API/app/core/Evaluations/article_extraction_benchmark.py`
- Modify: `Helper_Scripts/Evals/run_article_extraction_benchmark.py`

**Success Criteria:** Bootstrap confidence values are repeatable for a supplied seed, evaluation does not mutate module-global random state, and zero or negative bootstrap counts fail with a clear `ValueError`.

- [ ] **Step 1: Mark TASK-13139.1 In Progress**

Use the official Backlog.md workflow to set `TASK-13139.1` to In Progress and add this plan path to its documentation.

- [ ] **Step 2: Add failing deterministic-bootstrap tests**

Create a three-item ground-truth/prediction sample with exact, partial, and empty matches. Add these tests:

```python
def test_evaluate_metrics_is_repeatable_without_mutating_global_random_state() -> None:
    state = random.getstate()

    first = evaluate_metrics(GROUND_TRUTH, PREDICTIONS, 50, bootstrap_seed=7)
    second = evaluate_metrics(GROUND_TRUTH, PREDICTIONS, 50, bootstrap_seed=7)

    assert first == second
    assert random.getstate() == state


@pytest.mark.parametrize("count", [0, -1])
def test_evaluator_rejects_non_positive_bootstrap_count(tmp_path: Path, count: int) -> None:
    dataset = _minimal_dataset(tmp_path)

    with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
        ArticleExtractionBenchmarkEvaluator(dataset, n_bootstrap=count)
```

Also call `evaluate_metrics()` directly with zero and negative counts and assert the same `ValueError`, and reject a non-integer `bootstrap_seed` with `ValueError("bootstrap_seed must be an integer")`.

- [ ] **Step 3: Run the focused tests and confirm red**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/test_article_extraction_benchmark.py -q
```

Expected: FAIL because `bootstrap_seed` is not accepted and non-positive counts are not validated.

- [ ] **Step 4: Add the seed to the evaluator contract**

Use these signatures:

```python
class ArticleExtractionBenchmarkEvaluator:
    def __init__(
        self,
        dataset_root: Path,
        extractor: Callable[[str, str], str] | None = None,
        n_bootstrap: int = 1000,
        bootstrap_seed: int = 0,
    ) -> None: ...


def evaluate_metrics(
    ground_truth: dict[str, dict[str, str]],
    prediction: dict[str, dict[str, str]],
    n_bootstrap: int,
    *,
    bootstrap_seed: int = 0,
) -> dict[str, Any]: ...
```

Validate `type(n_bootstrap) is int and n_bootstrap > 0` in the evaluator and in the standalone function. Validate `type(bootstrap_seed) is int`. Construct `rng = random.Random(bootstrap_seed)` inside `evaluate_metrics()` and replace `random.randint(...)` with `rng.randint(...)`. Pass `self.bootstrap_seed` from `run()`.

- [ ] **Step 5: Add the CLI flag**

Add:

```python
parser.add_argument(
    "--bootstrap-seed",
    type=int,
    default=0,
    help="Deterministic bootstrap seed (default: 0).",
)
```

Pass `bootstrap_seed=args.bootstrap_seed` to `ArticleExtractionBenchmarkEvaluator`.

- [ ] **Step 6: Run the focused tests and confirm green**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/test_article_extraction_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the reproducibility slice**

```bash
git add tldw_Server_API/app/core/Evaluations/article_extraction_benchmark.py Helper_Scripts/Evals/run_article_extraction_benchmark.py tldw_Server_API/tests/Evaluations/test_article_extraction_benchmark.py
git commit -m "test(evals): make article benchmark deterministic (TASK-13139.1)"
```

## Task 2: Add the Pure Fixture and Report Contract

**Files:**

- Add: `tldw_Server_API/app/core/Evaluations/web_retrieval_quality.py`
- Add: `tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py`

**Success Criteria:** The module strictly validates all four case kinds, reports versioned deterministic metrics, and distinguishes Unicode character counts from UTF-8 byte counts without a tokenizer.

- [ ] **Step 1: Add failing contract and metric tests**

Add tests for:

1. wrong schema version;
2. duplicate case IDs;
3. unknown fields and unsupported case kinds;
4. missing kind-specific fields;
5. one valid case of each kind;
6. `output_text="é🙂"` producing `characters == 2`, `utf8_bytes == 6`, and estimated tokens `value == 1` with `authoritative is False`;
7. extraction case metrics matching the existing shingle helpers;
8. stable case sorting and report equality across two calls;
9. all six algorithm-version keys exactly matching the contract above.

Use a compact in-test suite mapping first; do not add the checked fixture until Task 3.

- [ ] **Step 2: Run the contract tests and confirm red**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py -q
```

Expected: FAIL because `web_retrieval_quality` does not exist.

- [ ] **Step 3: Implement the public interface**

The module exports exactly:

```python
FIXTURE_SCHEMA_VERSION = "web-retrieval-quality-fixture-v1"
REPORT_SCHEMA_VERSION = "web-retrieval-quality-report-v1"
ALGORITHM_VERSIONS: Mapping[str, str]

class FixtureValidationError(ValueError): ...

def load_fixture_suite(path: Path) -> dict[str, Any]: ...
def validate_fixture_suite(value: Mapping[str, Any]) -> dict[str, Any]: ...
def evaluate_fixture_suite(value: Mapping[str, Any]) -> dict[str, Any]: ...
def serialize_report(report: Mapping[str, Any]) -> str: ...
def render_human_summary(report: Mapping[str, Any]) -> str: ...
```

Implementation constraints:

- `validate_fixture_suite()` returns a defensive plain-dict/list/string/int copy and never retains caller-owned mutable objects.
- Reject booleans where integers are required.
- Reject non-finite floats if a future observed metric reaches validation; do not emit NaN or Infinity.
- Reuse `string_shingle_matching`, `precision_score`, `recall_score`, and `get_accuracy` from `article_extraction_benchmark.py`; do not duplicate tokenization.
- Serialize with `json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"`.
- The human summary is stable, contains no timestamp or color codes, and uses this exact line grammar (with six decimal places for scores):

```python
f"suite={suite_id} baseline={baseline_revision} cases={case_count}"
f"case={case_id} kind={kind} score={score:.6f} characters={characters} utf8_bytes={utf8_bytes}"
f"total mean_case_score={mean_case_score:.6f} characters={total_characters} utf8_bytes={total_utf8_bytes} estimated_tokens={estimated_tokens} authoritative=false"
```

  Emit one `case=` line per case sorted by ID and end the returned summary string without an embedded trailing newline; the CLI's `print()` supplies the newline.

- [ ] **Step 4: Run the contract tests and confirm green**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py -q
```

Expected: PASS for the in-test fixture cases.

- [ ] **Step 5: Commit the fixture/report core**

```bash
git add tldw_Server_API/app/core/Evaluations/web_retrieval_quality.py tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py
git commit -m "feat(evals): add web retrieval quality contract (TASK-13139.1)"
```

## Task 3: Check In the Current-Dev Fixture, Runner, and Baseline

**Files:**

- Add: `tldw_Server_API/tests/Web_Scraping/fixtures/retrieval_quality/v1.json`
- Add: `Helper_Scripts/Evals/run_web_retrieval_quality_baseline.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py`
- Add: `Docs/Evals/baselines/web_retrieval_quality_v1.json`

**Success Criteria:** The checked fixture contains one representative case per kind, the runner is offline and deterministic, and regenerating the report exactly matches the checked baseline artifact.

- [ ] **Step 1: Add the four-case fixture**

Use only reserved `.test` URLs and short synthetic content. The extraction case must include navigation text in the HTML and omit it from the observed article text. The search case must contain at least two providers and a deliberate tie/order observation. The crawl graph must contain a cycle and stop at an explicit page limit. The provenance record must include safe source/final URLs, content fingerprint, retrieval/extraction version, selected tier, and truncation fields, with no headers, cookies, credentials, raw HTML, or query secrets.

Set `baseline_revision` exactly to `9fd2246157ce8a32ae6a6691a75efab788229f77`.

- [ ] **Step 2: Add failing checked-artifact and CLI tests**

Extend the test module to assert:

```python
def test_checked_fixture_generates_the_checked_baseline() -> None:
    suite = load_fixture_suite(FIXTURE_PATH)
    report = evaluate_fixture_suite(suite)

    assert serialize_report(report) == BASELINE_PATH.read_text(encoding="utf-8")


def test_cli_writes_json_and_stable_human_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "report.json"

    assert main(["--fixture", str(FIXTURE_PATH), "--json-out", str(destination)]) == 0
    assert destination.read_text(encoding="utf-8") == BASELINE_PATH.read_text(encoding="utf-8")
    assert capsys.readouterr().out == render_human_summary(
        evaluate_fixture_suite(load_fixture_suite(FIXTURE_PATH))
    ) + "\n"
```

- [ ] **Step 3: Run the new tests and confirm red**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py -q
```

Expected: FAIL because the CLI and baseline artifact do not exist yet.

- [ ] **Step 4: Implement the thin CLI**

Use this interface:

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

Arguments:

- `--fixture PATH`, defaulting to `tldw_Server_API/tests/Web_Scraping/fixtures/retrieval_quality/v1.json` resolved from the repository root;
- `--json-out PATH`, optional; when supplied, create its parent directory and write `serialize_report(report)` using UTF-8;
- no network, refresh, threshold, comparator, or mutation flag.

Always print `render_human_summary(report)` to stdout and return zero. Let validation and file errors fail visibly; do not replace them with an empty success report.

- [ ] **Step 5: Generate and inspect the checked baseline**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python Helper_Scripts/Evals/run_web_retrieval_quality_baseline.py --json-out Docs/Evals/baselines/web_retrieval_quality_v1.json
git diff -- Docs/Evals/baselines/web_retrieval_quality_v1.json
```

Expected: four cases, all schema/algorithm versions, character/byte totals, a non-authoritative token estimate, and no timestamps, absolute paths, secrets, or network-derived fields.

- [ ] **Step 6: Run the fixture and CLI tests and confirm green**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py -q
```

Expected: PASS, including byte-for-byte baseline equality.

- [ ] **Step 7: Commit the checked baseline slice**

```bash
git add Helper_Scripts/Evals/run_web_retrieval_quality_baseline.py tldw_Server_API/tests/Web_Scraping/fixtures/retrieval_quality/v1.json tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py Docs/Evals/baselines/web_retrieval_quality_v1.json
git commit -m "test(web): record retrieval quality baseline (TASK-13139.1)"
```

## Task 4: Document, Verify, Review, and Finalize

**Files:**

- Modify: `Docs/Evals/WebScraping_Article_Benchmark.md`
- Modify: `Docs/Published/Evaluations/WebScraping_Article_Benchmark.md`
- Modify through official Backlog workflow: `TASK-13139.1`

**Success Criteria:** Operators can distinguish the fast offline fixture baseline from the full external benchmark, every focused check passes, Bandit finds no new touched-scope issue, and the task record contains exact evidence.

- [ ] **Step 1: Update the evaluation documentation**

Document:

- the fast baseline command and checked report path;
- the exact fixture/report and algorithm version strings;
- frozen observations versus behavior tests and thresholds;
- character/UTF-8 byte authority and approximate token estimate;
- the full Scrapinghub benchmark command with `--bootstrap-seed 0`;
- that neither the fast baseline nor its required tests use the public network;
- that fuzz, soak, PDF, and external comparator work remains parked under the roadmap.

Keep the source and published mirror content synchronized.

- [ ] **Step 2: Run focused verification**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/test_article_extraction_benchmark.py tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py -q
python Helper_Scripts/Evals/run_web_retrieval_quality_baseline.py --json-out /tmp/web_retrieval_quality_task_13139_1.json
cmp /tmp/web_retrieval_quality_task_13139_1.json Docs/Evals/baselines/web_retrieval_quality_v1.json
python -m bandit -r tldw_Server_API/app/core/Evaluations/article_extraction_benchmark.py tldw_Server_API/app/core/Evaluations/web_retrieval_quality.py Helper_Scripts/Evals/run_article_extraction_benchmark.py Helper_Scripts/Evals/run_web_retrieval_quality_baseline.py -f json -o /tmp/bandit_task_13139_1.json
git diff --check
```

Expected: pytest passes; `cmp` exits zero; Bandit exits zero with no new finding in touched code; `git diff --check` emits nothing.

- [ ] **Step 3: Perform self-review**

Confirm:

- every TASK-13139.1 acceptance criterion has a test or documentation line;
- report serialization is independent of random state, locale, clock, machine path, and mapping order;
- the baseline is descriptive and no threshold silently became a release gate;
- no production web retrieval behavior changed except deterministic evaluation sampling;
- no dependency, network call, tokenizer, crawler, database, queue, or persistence path was added;
- docs and checked report contain no secret, raw cookie/header, unsafe URL query, or absolute developer path.

- [ ] **Step 4: Request code review and address findings**

Use `superpowers:requesting-code-review`. Apply `superpowers:receiving-code-review` before changing code in response to findings. Rerun the focused verification after any change.

- [ ] **Step 5: Finalize TASK-13139.1**

Through the official Backlog workflow:

- check all acceptance criteria and Definition of Done items;
- record the exact pytest, baseline parity, Bandit, and `git diff --check` results;
- document any environment-only skip explicitly;
- add a concise final summary and set the task to Done only when all required checks pass.

- [ ] **Step 6: Commit documentation and task finalization**

```bash
git add Docs/Evals/WebScraping_Article_Benchmark.md Docs/Published/Evaluations/WebScraping_Article_Benchmark.md backlog/tasks/task-13139.1\ -\ Establish-a-minimal-deterministic-web-retrieval-quality-baseline.md
git commit -m "docs(evals): explain retrieval quality baseline (TASK-13139.1)"
```

## Final Acceptance Checklist

- [ ] The checked fixture covers extraction, search order, crawl graph/stop observation, and safe provenance entirely offline.
- [ ] Repeated runs produce byte-identical JSON and stable human output.
- [ ] Character and UTF-8 byte budgets are authoritative; token estimates are labeled approximate.
- [ ] Every scoring and budget calculation carries an explicit algorithm version.
- [ ] The external extraction benchmark uses a local seeded RNG and rejects invalid bootstrap counts.
- [ ] No feature-specific production adapter, threshold, fuzz/soak system, external comparator, or new dependency was introduced.
- [ ] Focused tests, baseline parity, Bandit, and whitespace verification pass and are recorded in TASK-13139.1.
