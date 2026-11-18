# The registered benchmarks run on correctly shaped data (TASK-15487) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every benchmark the registry lists runs: its rows reach the judge in its evaluator's fields, its evaluator builds, and its judge prompt asks for the 1-10 scale the manager reads. After #3445 (TASK-15485) the judge reaches the provider, but on a default install only `simple_bench` and `bullshit_benchmark` reach it with meaningful data: `mmlu_pro` rows are not field-mapped (its judge is told the correct answer is "None"), `simpleqa_verified` and `gpqa_diamond` load no rows (404), six benchmarks build no evaluator (501), and every formatter but multiple choice still asks for `SCORE: X.X` or 0-10, which the manager's 1-10 read turns a perfect reply into the worst rating.

**Architecture:** Four root fixes, no new abstraction.
- `load_benchmark_dataset` (`benchmark_loaders.py`) is the one loader every caller routes through (both run routes, the three CLIs). It now looks the benchmark up in the registry and applies its `field_mappings` to the rows it returns (merge, not replace, so a loader that already shaped its rows keeps them). The registry's own dead `load_dataset` (no caller; its `huggingface` and `custom` branches returned `[]`) is deleted. `DatasetLoader.load_csv` reads its rows while the file is open (it iterated the lazy reader after the `with` closed, so every local-CSV path raised).
- The two HuggingFace loaders load what the datasets are: `load_huggingface` takes the config `name`; `load_gpqa` loads config `gpqa_<subset>` (split `train`) and shapes each row's `Correct Answer` + `Incorrect Answer 1..3` into `choices` (shuffled once per item, seeded by its `Record ID`) with the answer letter. `mmlu_pro`'s mapping names the real `answer` column (`answer_idx` does not exist).
- `simpleqa_verified` gets a loader entry and its source is the file that actually ships, `Helper_Scripts/Evals/simpleqa-verified.zip` (one CSV with a `problem` column), read with `zipfile` + `csv`. `simpleqa` (owner's choice, recommended: build) points at OpenAI's published CSV and drops the `max_tokens` param its evaluator does not take. The five benchmarks that need their own harness (`aider_polyglot`, `bfcl`, `mask`, `swe_bench`, `vending_bench`) leave the registry (owner's choice, recommended: remove).
- Every judge prompt ends with one shared `benchmark_utils.JUDGE_SCALE_LINE` (1 worst to 10 best) and its rubric names points on that scale, not `1.0/0.0`, `0-10` or "N% of score": the five evaluator classes and three helpers in `benchmark_utils.py`, plus `QAEvaluation`'s default judge prompt (`qa_benchmark_helper.py`) and the template users copy it from (`configs/simple_qa_template.yaml`). `evaluate_custom_metric` (the shared manager) reads the replies a human reads as a score (a bare number; a numeric string in JSON) and the two GRADE forms (a restated format line is not a grade; `not attempted` with a space is read); a number in exponent form (`1e308`, `15e3`, `1.5e3`, `10e5`, `3.0e2`) and a numeric string with underscores (`"1_0"`) leave the item unscored, under the owner's rule.

**Tech Stack:** Python 3.12 (atomic groups in `re`, 3.11+), pytest (pytest-asyncio), click CliRunner, backlog-py, `zipfile`/`csv`/`random` (stdlib), `datasets` (optional, stubbed in tests).

**Spec:** `backlog/tasks/task-15487 - The-registered-benchmarks-run-on-correctly-shaped-data.md` (description, ACs #1-#4, implementation notes); TASK-15485's notes and `Docs/superpowers/plans/2026-11-07-benchmark-judge-reaches-the-provider.md` and `2026-11-10-benchmark-items-with-no-score-are-unscored.md`; the planner brief `brief_plan_burn.md` (plans dir); the independent review `review_15487.md` (plans dir), folded in: see "Review fold". AC #3 is an owner decision: see Decisions, D1.

**Measured on:** `origin/dev` at `3883a73712`, in a detached worktree (the reviewer re-measured the prefold plan on `61ad5280ca`: none of this plan's files changed between the two). `file:line` citations are for `3883a73712` unless marked "after".

**Prototype:** `proto_15487.patch` (plans dir): 14 files, +409/-345, every code, test and docs change in this plan, the `Docs/Published` mirror included (the new test file is in it via intent-to-add). It does not hold the plan copy or the backlog edits. It implements the recommended AC #3 split (D1); the alternative is a smaller subset (see D1). The pre-review version is `proto_15487.prefold.patch` (12 files, +353/-320) with `plan_15487.prefold.md`.

---

## Where the ACs are wrong or incomplete against the code (evidence)

- **AC #1** says "applies `config.field_mappings`". Applying them is not enough: `mmlu_pro`'s mapping (`benchmark_registry.py:169`, and `configs/mmlu_pro.yaml:9`) maps `correct_answer` to `answer_idx`, a column MMLU-Pro does not have. Its columns (HuggingFace datasets-server `info`, `TIGER-Lab/MMLU-Pro`, split `test`, 12,032 rows) are `question_id, question, options, answer, answer_index, cot_content, category, src`; `answer` is the letter. Corrected AC #1: *`load_benchmark_dataset` applies the registry's `field_mappings`, and `mmlu_pro`'s mapping names the real columns (`options`, `answer`, `category`), with a test that a row in MMLU-Pro's own fields reaches the judge with its ten choices and the answer letter.*
- **AC #2** says `simpleqa_verified` "loads rows" once it has a loader entry. Its registered source, `Helper_Scripts/Evals/SimpleQA_verified/` (`benchmark_registry.py:117`), does not exist; what ships is `Helper_Scripts/Evals/simpleqa-verified.zip`, one file `simpleqa_verified.csv` (349 KB; columns `original_index, problem, answer, topic, answer_type, multi_step, requires_reasoning, urls`, the same as `google/simpleqa-verified` on HuggingFace, split `eval`, 1,000 rows). The loader's directory branch (`benchmark_loaders.py:323-337`) reads only `.jsonl`/`.json`, so even an extracted zip would load nothing. Corrected AC #2 (first half): *`simpleqa_verified` has a loader entry, its source is the shipped zip, the loader reads the CSV inside it, and `problem` reaches the evaluator as `question`.* The README's "stored in the repo at `Helper_Scripts/Evals/SimpleQA_verified/`" (`Evaluations/README.md:208-210`) is corrected with it.
- **AC #2** (second half) and the description's "gpqa_diamond ... no config name and filters on a 'subset' column GPQA rows lack" are confirmed and short of the fix: `Idavidrein/gpqa` is gated (`gated: auto`, HuggingFace API) and has four configs, `gpqa_diamond`, `gpqa_main`, `gpqa_extended`, `gpqa_experts`, one CSV each; its rows hold `Question`, `Correct Answer`, `Incorrect Answer 1`, `Incorrect Answer 2`, `Incorrect Answer 3`, `Subdomain`, `High-level domain`, `Record ID` (the columns lm-evaluation-harness's GPQA task reads), not `choices`/`correct_answer`/`subject` as the registry maps (`benchmark_registry.py:190-195`). So "loads with the right config name and subset" also needs the loader to build the choices and the answer letter, else `MultipleChoiceEvaluation` formats an empty choice list. Corrected: *`gpqa_diamond` loads config `gpqa_diamond` (split `train`) and each row reaches the evaluator with four `choices`, the `correct_answer` letter and a `category`; a local file in GPQA's columns loads the same way.*
- **AC #3** stands as the owner's decision. Measured today: all six answer 501 from both run routes (`create_evaluator` returns `None`: `aider_polyglot` (`languages`), `mask` (`check_consistency`), `simpleqa` (`max_tokens`), `swe_bench` (`use_repo_context`) raise `TypeError` in the class constructor; `bfcl` and `vending_bench` map to `None` in `EVALUATION_TYPES`), and "Could not create evaluator" from the CLI; no LLM call is made.
- **AC #4** stands, read literally: "every formatter" includes `QAEvaluation._default_judge_prompt` (`qa_benchmark_helper.py:50`) and `configs/simple_qa_template.yaml:49`, which also ask `SCORE: X.X` (review M1; neither is a registered benchmark, so no registered run is affected). The implementation notes' parser gaps (bare number, numeric string, `1e308`, the GRADE format echo, `not attempted`) are in scope here and done.

## What a run does today vs after (verified on `3883a73712`)

| Benchmark | Today | After |
|---|---|---|
| `simple_bench` | runs (its loader shapes rows itself) | unchanged; the mapping merge keeps its rows |
| `bullshit_benchmark` | runs; asks 0-10, read on 1-10 (`SCORE: 1` gives 0.0, `5` gives 0.444) | asks 1-10 |
| `mmlu_pro` | needs `datasets`; rows not mapped: judge told "correct answer: None", choices empty | rows mapped (`options` -> `choices`, `answer` -> `correct_answer`); reasoning rubric on 1-10 |
| `gpqa_diamond` | needs `datasets`; no config name, `subset` filter drops every row: 404 | config `gpqa_diamond`, choices built; gated: needs `HF_TOKEN`; a local CSV in its columns loads too |
| `simpleqa_verified` | no loader entry: 404 | loads the shipped zip's 1,000 rows |
| `simpleqa` | 501 (`max_tokens`); loader targets a non-existent HF dataset | D1: builds, fetches OpenAI's CSV (4,326 rows) at run time; or removed |
| `aider_polyglot`, `bfcl`, `mask`, `swe_bench`, `vending_bench` | 501 | D1: removed from the registry (recommended) |

## Global Constraints

- Tests make no real outbound network requests. Every pytest run uses the network probe plugin:
  `PYTHONPATH=<plans dir>:<plans dir>/pyext NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`. It blocks and logs every
  outbound DNS lookup and connection; a run must end with 0 log lines for the tests the plan adds or changes.
  (`<plans dir>/pyext` holds `pytest_mock`, which the shared venv lost and the root conftest loads; never an empty stub.)
- A test that drives `/chat/completions`, an LLM adapter or a judge must stub the adapter or the provider call. A
  commercial provider otherwise calls the real API.
- A failed judge or model call, or a reply with no readable score, is unscored: `score: None`, a sample error, and
  out of every average. Never 0.0, never a success. This is the owner's rule, recorded 2026-11-10 and 2026-11-12.
- Backlog edits use backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`).
  Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit task files. A
  new follow-up task takes its id under the session's id lock (the controller does that; write `TASK-<NEW>`).
- Branch from `origin/dev`; the PR targets `dev`.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`. Never restart the server.
  Redis: never FLUSHALL or SCRIPT FLUSH.
- A mirrored doc (ADRs, the BYOK guide, `Env_Vars.md` and the other files under `Docs/Published`) needs
  `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same PR.
- Shifting lines in a file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` fails backend-required.
  Regenerate it with the script's `--write-baseline`, in the same PR. (None of this plan's files is listed; verified.)
- Size-capped functions (the function-size ratchet in `tests/lint`) must not grow past their caps. Measure each
  touched function before and after. (The ratchet freezes functions over 300 lines; none touched here is near it: see Task 5.)
- A plan with embedded diffs fails trailing-whitespace when it is committed: keep the plan free of trailing spaces.
- No test downloads a dataset: `datasets.load_dataset` is a stub module in `sys.modules` (the package is not in the shared venv, and the stub works either way); OpenAI's CSV fetch is stubbed at `DatasetLoader.load_csv`; `simpleqa_verified` reads the zip in the repo (no network); the local-CSV test writes its file under `tmp_path`.
- Shared file: `tldw_Server_API/app/core/Evaluations/evaluation_manager.py` (`evaluate_custom_metric`'s reply parser) and its test `tests/Evaluations/unit/test_evaluation_manager.py` are also the evaluations planner's territory (TASK-15489/15491/15492: the eval runner, the CLI, `evaluator_model`). This plan touches only the parse block (`:622-672`) and appends one test; the reviewer checked `proto_evals_15489.patch` does not touch `evaluation_manager.py`, and that `proto_15486.patch`'s `test_security.py` hunks (`:197-214`) are disjoint from this plan's pin (`:228`).
- Ponytail: smallest correct diff; every fix at the root each caller routes through (`load_benchmark_dataset`, `load_csv`, `JUDGE_SCALE_LINE`, the one parser).

## Review Focus

1. **Field-mapping merge semantics.** `{**item, **mapped}`: a mapped target overwrites a same-named raw key (`mmlu_pro`'s `question -> question` is identity; `gpqa` rows are pre-shaped so the mapping is identity; `simple_bench` rows lack `prompt`/`answer` at the top level, so nothing changes). Confirm no registered mapping makes a raw column shadow a shaped field.
2. **GPQA shaping.** `choices.index(correct)` after the shuffle; the shuffle is seeded by `Record ID` (reproducible, A not always right); `.strip()` on the four answers. A local file already in the benchmark's fields passes through (`"Correct Answer" not in row`). `load_csv` now returns inside the `with`.
3. **The parser restructure** (`evaluation_manager.py`): JSON dict branch vs everything else (a bare number now reaches the patterns); the number is an atomic group `((?>\d+(?:\.\d+)?))(?![eE][-+]?\d)` so no exponent form is read in part (the prefold guard let `15e3` read as 1: review I1), and it must still read `Score: 8.` and `8/10`; a numeric string must `fullmatch` `\d+(?:\.\d+)?` (so `"1_0"` is not 10: review M2); the GRADE `\b(?!/)` must still read `GRADE: [incorrect]` (the bracketed case in the existing test) and reject the echoed `[correct/incorrect/not_attempted]`.
4. **Deletions.** The five registry entries, three loader methods (`load_aider_polyglot`, `load_swe_bench`, `load_bfcl`) and `BenchmarkRegistry.load_dataset`: no caller remains (`git grep` in Task 2). The `register` CLI still offers `function_calling`/`multi_turn` types that build no evaluator (unchanged behaviour; out of scope).
5. **Rubric wording.** Each formatter's rubric now names points on 1-10 (or weights, with the scale line after); the shared `JUDGE_SCALE_LINE` ends every prompt but SimpleQA's (grade-based) and bullshit's own labelled `SCORE: [1-10]`. `qa_benchmark_helper.py` imports `JUDGE_SCALE_LINE` from `benchmark_utils` (stdlib + loguru only: no cycle).
6. **Docs.** The guide bullet (and mirror) names what runs, what needs `datasets`/`HF_TOKEN`, and the five removed names.

---

## Task 0: Mark In Progress, branch, draft PR, baseline

**Files:** `backlog/tasks/task-15487 - ...md` (backlog-py only).

- [ ] **Step 1: In Progress**
  `PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit 15487 --status "In Progress"` then `git add -A backlog/tasks` and commit `chore(backlog): TASK-15487 in progress`.
- [ ] **Step 2: Branch** `git fetch origin dev && git checkout -b fix/benchmarks-correct-data-15487 origin/dev` (then cherry-pick/commit Step 1 on it).
- [ ] **Step 3: Draft PR** `gh pr create --base dev --draft --title "fix(evaluations): the registered benchmarks run on correctly shaped data (TASK-15487)" --body-file <body>` (body ends with the generation line).
- [ ] **Step 4: Baseline** Run the touched test set on the base (the command of Task 5 Step 1 minus the new file, before any change):
  `python -m pytest -p netprobe_15427 -q tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py tldw_Server_API/tests/Evaluations/test_security.py tldw_Server_API/tests/Evaluations/unit/test_bullshit_benchmark_registry.py tldw_Server_API/tests/Evaluations/unit/test_bullshit_benchmark_loader.py tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py tldw_Server_API/tests/Evaluations/test_benchmark_api_auth.py tldw_Server_API/tests/Evaluations/test_benchmark_api_error_mapping.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_endpoint_sanitization.py tldw_Server_API/tests/http_client/test_benchmark_loaders_http.py tldw_Server_API/tests/Evaluations/unit/test_scoring_normalization.py tldw_Server_API/tests/LLM_Calls/test_next_token_capture.py`
  Expected: `1 failed, 136 passed` (measured on the clean base; the one failure is the probe artifact `test_security.py::TestWebhookSecurityValidation::test_ssrf_protection_private_ip`; the probe log has the same two lines as #3445 recorded: `test_ssrf_protection_private_ip` (private IPs) and `test_blocked_ports` (example.com)). With the plan's new and changed cases present, the whole set is `28 failed, 137 passed` on the base (the 28 are this plan's RED, Tasks 1-3) and `1 failed, 164 passed` after (Task 5 Step 1).

## Task 1: Rows reach the evaluator in the benchmark's own fields (AC #1, AC #2)

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/benchmark_loaders.py` (`DatasetLoader.load_csv`, `load_huggingface`; `BenchmarkDatasetLoader.load_gpqa` + new `_shape_gpqa_row`, `load_simpleqa`; `load_benchmark_dataset`)
- Modify: `tldw_Server_API/app/core/Evaluations/benchmark_registry.py` (`mmlu_pro`, `gpqa_diamond`, `simpleqa_verified` entries)
- Modify: `tldw_Server_API/app/core/Evaluations/configs/mmlu_pro.yaml` (`answer_idx` -> `answer`; an example file, no runtime reader)
- Create: `tldw_Server_API/tests/Evaluations/unit/test_benchmark_datasets.py` (shard `tests/Evaluations/unit/test_*.py`; shard coverage measured `new_uncovered=0`)

**Interfaces:**
- `DatasetLoader.load_csv(source, delimiter=',')`: returns the rows while the file (or the fetched text) is in scope.
- `DatasetLoader.load_huggingface(dataset_id, split='test', limit=None, revision=None, name=None)` -> `load_dataset(dataset_id, name=name, split=split, revision=...)`.
- `BenchmarkDatasetLoader.load_gpqa(source=None, subset="diamond")` -> rows `{question, choices, correct_answer, category, _original}`; `_shape_gpqa_row(row, index)`.
- `BenchmarkDatasetLoader.load_simpleqa(source=None)`: default source OpenAI's CSV; `.csv` (URL or file) via `load_csv`; `.zip` via `zipfile` + `csv.DictReader`; directory and file branches unchanged.
- `load_benchmark_dataset(benchmark_name, source=None, limit=None, **kwargs)`: looks the benchmark up in the registry (function-local import; the registry imports `benchmark_utils` only, so no cycle), `simpleqa`/`simpleqa_verified` default to the registered `dataset_source`, and `field_mappings` are merged into each row before `limit`.

- [ ] **Step 1: RED, create the test file with its header and the four AC tests** (verbatim; the formatter cases of Task 3 are appended later, so `yaml`, `QAEvaluation` and `_TEMPLATE` are imported now and used then):

```python
"""The registered benchmarks run on correctly shaped data (TASK-15487).

Rows reach the evaluator in the benchmark's own fields (the registry's field mappings are applied by
load_benchmark_dataset), the two HuggingFace benchmarks load with the right config and shape, every
registered benchmark builds an evaluator, and every judge prompt asks for the 1-10 scale the manager reads.
No test downloads a dataset: ``datasets`` is a stub module, and OpenAI's CSV fetch is stubbed.
"""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytes
import yaml

from tldw_Server_API.app.core.Evaluations import benchmark_utils
from tldw_Server_API.app.core.Evaluations.benchmark_loaders import DatasetLoader, load_benchmark_datase
from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
from tldw_Server_API.app.core.Evaluations.qa_benchmark_helper import QAEvaluation

pytestmark = pytest.mark.uni

_TEMPLATE = Path(benchmark_utils.__file__).with_name("configs") / "simple_qa_template.yaml"

# One row of TIGER-Lab/MMLU-Pro (test split), in its real fields
_MMLU_PRO_ROW = {
    "question_id": 70,
    "question": "Which of the following is the body cavity that contains the pituitary gland?",
    "options": ["Abdominal", "Cranial", "Pleural", "Spinal", "Thoracic", "Pelvic", "Buccal", "Pericardial", "Ventral", "Dorsal"],
    "answer": "B",
    "answer_index": 1,
    "cot_content": "",
    "category": "health",
    "src": "ori_mmlu-anatomy",
}
# One row of Idavidrein/gpqa (config gpqa_diamond, split train), in its real columns
_GPQA_ROW = {
    "Record ID": "recAAHX3nMOaCODRe",
    "Question": "What is the speed of light in vacuum, in km/s?",
    "Correct Answer": "299,792",
    "Incorrect Answer 1": "150,000",
    "Incorrect Answer 2": "1,079,252,848",
    "Incorrect Answer 3": "3,000",
    "Subdomain": "Physics (general)",
    "High-level domain": "Physics",
}


def _stub_datasets(monkeypatch, row):
    """A ``datasets`` module whose load_dataset records its call and returns one row; nothing is downloaded."""
    calls = []

    def load_dataset(path, name=None, **kwargs):
        calls.append((path, name, kwargs.get("split")))
        return [row]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    return calls


@pytest.mark.asyncio
async def test_an_mmlu_pro_row_reaches_the_judge_with_its_choices_and_answer(monkeypatch):
    """An MMLU-Pro row's options and answer letter reach the judge in the evaluator's fields (it was told the
    correct answer is None, so its scores carried no information)."""
    from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization

    _stub_datasets(monkeypatch, _MMLU_PRO_ROW)
    prompts: list[str] = []

    def chat(request, timeout=None):
        prompts.append(request["messages"][-1]["content"])
        return {"choices": [{"message": {"content": "B" if len(prompts) == 1 else "SCORE: 10 EXPLANATION: B."}}]}

    monkeypatch.setattr(
        summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _p: SimpleNamespace(chat=chat))
    )

    (row,) = load_benchmark_dataset("mmlu_pro", limit=1)
    data = get_registry().create_evaluator("mmlu_pro").format_for_custom_metric(row)
    result = await EvaluationManager().evaluate_custom_metric(
        metric_name=data["name"],
        description=data["description"],
        evaluation_prompt=data["evaluation_prompt"],
        input_data=data["input_data"],
        scoring_criteria=data["scoring_criteria"],
        api_name="openai",
        api_key="sk-test",
    )

    assert len(prompts) == 2, prompts  # the model's answer, then the judge
    assert "Compare with the correct answer: B" in prompts[1]
    assert "B. Cranial" in prompts[1] and "J. Dorsal" in prompts[1]
    assert data["metadata"]["category"] == "health"
    assert result["score"] == 1.0


def test_simpleqa_verified_loads_its_rows_from_the_shipped_zip():
    """The curated set ships as Helper_Scripts/Evals/simpleqa-verified.zip (one CSV, a 'problem' column)."""
    rows = load_benchmark_dataset("simpleqa_verified", limit=2)

    assert len(rows) == 2
    assert rows[0]["question"].startswith("How much money, in euros")
    assert rows[0]["answer"] == "120,000 euros"
    assert rows[0]["topic"] == "Politics"


def test_gpqa_diamond_loads_its_config_and_shapes_the_choices(monkeypatch):
    """GPQA is one gated dataset with a config per subset (train split); a row holds the correct answer and three
    incorrect ones in named columns, so the loader builds the choices and the answer letter."""
    calls = _stub_datasets(monkeypatch, _GPQA_ROW)

    (row,) = load_benchmark_dataset("gpqa_diamond")

    assert calls == [("Idavidrein/gpqa", "gpqa_diamond", "train")]
    assert sorted(row["choices"]) == sorted(["299,792", "150,000", "1,079,252,848", "3,000"])
    assert row["choices"][ord(row["correct_answer"]) - 65] == "299,792"
    assert row["question"] == _GPQA_ROW["Question"]
    assert row["category"] == "Physics (general)"


def test_gpqa_diamond_loads_a_local_csv_in_its_columns(tmp_path):
    """A local file in GPQA's own columns loads through the CSV reader (which read its rows after closing the file)."""
    path = tmp_path / "gpqa_diamond.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_GPQA_ROW))
        writer.writeheader()
        writer.writerow(_GPQA_ROW)

    (row,) = load_benchmark_dataset("gpqa_diamond", source=str(path))

    assert row["question"] == _GPQA_ROW["Question"]
    assert row["choices"][ord(row["correct_answer"]) - 65] == "299,792"
```

- [ ] **Step 2: Run RED** `python -m pytest -p netprobe_15427 -q tldw_Server_API/tests/Evaluations/unit/test_benchmark_datasets.py -k "mmlu_pro_row or shipped_zip or gpqa_diamond"`
  Expected: `4 failed`. Measured on the base: mmlu `assert 'Compare with the correct answer: B' in '{"question": ..., "choices": "", "correct_...'` (the judge is told `None`); zip `assert 0 == 2` (no loader entry: "No source specified for unknown benchmark"); gpqa `ValueError: not enough values to unpack (expected 1, got 0)` (no config name, and the `subset` filter drops the row); local csv `ValueError: not enough values to unpack` (the base's `load_gpqa` sends a `.csv` source to `load_json`, which fails and is caught). Probe log: 0 lines. (On the prefold code the local-csv test fails with `ValueError: I/O operation on closed file.`: review I2.)

- [ ] **Step 3: GREEN, the loaders** (`benchmark_loaders.py`). Imports gain `io`, `random`, `zipfile` (after `csv`, `json`, `os` respectively). `DatasetLoader.load_csv` becomes:

```python
    @staticmethod
    def load_csv(source: str, delimiter: str = ',') -> list[dict[str, Any]]:
        """Load CSV dataset from file or URL."""
        if source.startswith(('http://', 'https://')):
            r = fetch(method="GET", url=source, timeout=15)
            try:
                lines = r.text.strip().split('\n')
            finally:
                with contextlib.suppress(Exception):
                    r.close()
            return [dict(row) for row in csv.DictReader(lines, delimiter=delimiter)]

        with open(source, encoding='utf-8') as f:  # the reader is lazy: read the rows while the file is open
            return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]
```

  `load_huggingface` gains `name: Optional[str] = None` (after `revision`), its docstring's first line becomes `Load dataset from HuggingFace; ``name`` is the dataset's config (GPQA has one per subset).`, and the call becomes `load_dataset(dataset_id, name=name, split=split, revision=resolved_revision)  # nosec B615`. Replace `load_aider_polyglot`, `load_swe_bench`, `load_gpqa`, `load_bfcl` and the head of `load_simpleqa` (`:226-321`, up to and including `if p.is_dir():`) with:

```python
    @staticmethod
    def load_gpqa(source: Optional[str] = None,
                 subset: str = "diamond") -> list[dict[str, Any]]:
        """Load a GPQA subset: one config of the gated HuggingFace dataset (split ``train``), or a file in its columns."""
        if not source:
            data = DatasetLoader.load_huggingface("Idavidrein/gpqa", split="train", name=f"gpqa_{subset}")
        elif source.startswith("huggingface://"):
            data = DatasetLoader.load_huggingface(source.replace("huggingface://", ""), split="train", name=f"gpqa_{subset}")
        elif source.endswith('.csv'):
            data = DatasetLoader.load_csv(source)
        elif source.endswith('.jsonl'):
            data = DatasetLoader.load_jsonl(source)
        else:
            data = DatasetLoader.load_json(source)
        return [BenchmarkDatasetLoader._shape_gpqa_row(row, i) for i, row in enumerate(data)]

    @staticmethod
    def _shape_gpqa_row(row: dict[str, Any], index: int) -> dict[str, Any]:
        """A GPQA row holds the correct answer and three incorrect ones in named columns. The choices are
        shuffled once per item, seeded by its Record ID, so a run is reproducible and A is not always right."""
        if "Correct Answer" not in row:
            return row  # already in the benchmark's fields (a local file)
        correct = str(row["Correct Answer"]).strip()
        choices = [correct] + [str(row[f"Incorrect Answer {n}"]).strip() for n in (1, 2, 3)]
        random.Random(str(row.get("Record ID") or index)).shuffle(choices)  # nosec B311 - reproducible order, not security
        return {
            "question": row.get("Question", ""),
            "choices": choices,
            "correct_answer": chr(65 + choices.index(correct)),
            "category": row.get("Subdomain") or row.get("High-level domain") or "general",
            "_original": row,
        }

    @staticmethod
    def load_simpleqa(source: Optional[str] = None) -> list[dict[str, Any]]:
        """Load a SimpleQA dataset: OpenAI's CSV (the default), a URL, a local file, a zip holding a CSV, or a directory."""
        if not source:
            source = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"

        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            return DatasetLoader.load_huggingface(dataset_id)
        elif source.endswith('.csv'):
            return DatasetLoader.load_csv(source)
        elif source.startswith("http"):
            # Load from URL
            if source.endswith('.jsonl'):
                return DatasetLoader.load_jsonl(source)
            else:
                return DatasetLoader.load_json(source)
        else:
            # Load from local path (zip, file or directory)
            from tldw_Server_API.app.core.Evaluations.simpleqa_eval import SimpleQADatase
            p = Path(source)
            if p.suffix == ".zip":  # the shipped simpleqa-verified.zip: one CSV inside
                with zipfile.ZipFile(p) as archive:
                    member = next(n for n in archive.namelist() if n.endswith(".csv"))
                    with archive.open(member) as raw:
                        return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")))
            if p.is_dir():
```

  (The rest of `load_simpleqa`, the directory and file branches, is unchanged. The old `if not source:` branch tried HuggingFace `openai/simple-qa`, which does not exist, and its `except` fallback to sample data was unreachable: `load_huggingface` catches everything and returns `[]`. The three deleted loaders: `load_aider_polyglot` returned `[]` by design; `load_swe_bench` and `load_bfcl` have no registry entry after Task 2 and no other caller: `git grep -n "load_swe_bench\|load_bfcl\|load_aider_polyglot" -- tldw_Server_API` is empty after.)

  `load_benchmark_dataset` becomes (the docstring's Args are unchanged):

```python
def load_benchmark_dataset(benchmark_name: str,
                          source: Optional[str] = None,
                          limit: Optional[int] = None,
                          **kwargs) -> list[dict[str, Any]]:
    """Load dataset for a specific benchmark.

    Args:
        benchmark_name: Name of the benchmark
        source: Optional custom source (file, URL, or HF dataset)
        limit: Maximum number of samples to load
        **kwargs: Additional benchmark-specific parameters

    Returns:
        List of dataset items, in the benchmark's own fields: the registry's field_mappings are applied here,
        so every caller (both run routes and the CLIs) gets the shape its evaluator formats (TASK-15487).
    """
    from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry

    config = get_registry().get(benchmark_name)
    registered_source = config.dataset_source if config else None

    # Map benchmark names to loaders
    loaders = {
        "mmlu_pro": BenchmarkDatasetLoader.load_mmlu_pro,
        "simple_bench": BenchmarkDatasetLoader.load_simple_bench,
        "simpleqa": lambda s: BenchmarkDatasetLoader.load_simpleqa(s or registered_source),
        "simpleqa_verified": lambda s: BenchmarkDatasetLoader.load_simpleqa(s or registered_source),
        "bullshit_benchmark": BenchmarkDatasetLoader.load_bullshit_benchmark,
        "gpqa": BenchmarkDatasetLoader.load_gpqa,
        "gpqa_diamond": lambda s: BenchmarkDatasetLoader.load_gpqa(s, subset="diamond"),
    }

    if benchmark_name in loaders:
        data = loaders[benchmark_name](source)
    else:
        # Try generic loading based on source forma
        if not source:
            logger.error(f"No source specified for unknown benchmark: {benchmark_name}")
            return []

        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            data = DatasetLoader.load_huggingface(dataset_id)
        elif source.endswith('.jsonl'):
            data = DatasetLoader.load_jsonl(source)
        elif source.endswith('.json'):
            data = DatasetLoader.load_json(source)
        elif source.endswith('.csv'):
            data = DatasetLoader.load_csv(source)
        else:
            logger.error(f"Cannot determine format for source: {source}")
            return []

    if config and config.field_mappings:
        # Merge, not replace: a loader that already shaped its rows (simple_bench) keeps them
        data = [
            {**item, **{target: item[field] for target, field in config.field_mappings.items() if field in item}}
            for item in data
        ]

    # Apply limit if specified
    if limit and limit > 0:
        data = data[:limit]

    return data
```

- [ ] **Step 4: GREEN, the registry mappings** (`benchmark_registry.py`): `mmlu_pro` `"correct_answer": "answer"` (its comment: `# MMLU Pro (needs the datasets package: the data-tools extra). Its rows: question, options, answer (a letter),` / `# answer_index, category.`); `gpqa_diamond` `"category": "category"` (comment: `# GPQA Diamond (needs the datasets package; the dataset is gated: accept its terms on HuggingFace and set` / `# HF_TOKEN). The loader shapes its columns into question, choices, correct_answer and category.`); `simpleqa_verified` (comment `# SimpleQA Verified: the curated set ships in the repo as a zip holding one CSV ("problem", "answer", "topic")`): `dataset_source = str(repo_root / "Helper_Scripts" / "Evals" / "simpleqa-verified.zip")`, `dataset_format="csv"`, `field_mappings={"question": "problem", "answer": "answer", "topic": "topic"}`. `configs/mmlu_pro.yaml:9` `correct_answer: answer`.

- [ ] **Step 5: Run GREEN** the Step 2 command. Expected: `4 passed`, probe log 0 lines. Then `python -m pytest -p netprobe_15427 -q tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py -k "benchmark_item or handle_it_is_given or failed_model_answer" tldw_Server_API/tests/http_client/test_benchmark_loaders_http.py` Expected: all pass (the `simpleqa_verified` and `mmlu_pro` cases go through `create_evaluator` + the formatter, untouched; `test_load_csv_via_http_monkeypatch` still gets `[{"id": "1", ...}, {"id": "2", ...}]`).

- [ ] **Step 6: Commit** `fix(evaluations): benchmark rows reach the evaluator in their own fields (TASK-15487 AC #1, #2)`.

## Task 2: The registry lists only benchmarks that run (AC #3, owner's decision D1)

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/benchmark_registry.py` (delete five entries, `simpleqa` entry, delete `load_dataset` and the three now-unused imports)
- Modify: `tldw_Server_API/app/core/Evaluations/README.md` (SimpleQA Verified section; the two `registry.load_dataset` examples), `EVALS_DEVELOPER_GUIDE.md:959-960` (one example line)
- Modify: `tldw_Server_API/tests/Evaluations/unit/test_benchmark_datasets.py` (two tests)

**Interfaces:** `BenchmarkRegistry.list_benchmarks()` returns `simpleqa, simpleqa_verified, bullshit_benchmark, mmlu_pro, gpqa_diamond, simple_bench`; `create_evaluator` returns an evaluator for each. `BenchmarkRegistry.load_dataset` is gone (no caller in app or tests: `git grep -n "\.load_dataset(" -- tldw_Server_API | grep -v datasets` shows only the two doc examples, fixed here).

This task implements D1's recommendation (the reviewer agreed). **Alternative (remove all six):** also delete the `simpleqa` entry and its loader line, and the `test_simpleqa_loads_openais_csv...` test; the registry set in the test is then five names. **Alternative (build `bfcl` too):** see D1; not prototyped.

- [ ] **Step 1: RED, two tests** (append to the new file, verbatim):

```python
def test_simpleqa_loads_openais_csv_in_the_benchmarks_fields(monkeypatch):
    """SimpleQA is OpenAI's CSV (problem, answer); its rows reach the evaluator as question and answer."""
    sources: list[str] = []
    monkeypatch.setattr(
        DatasetLoader,
        "load_csv",
        staticmethod(
            lambda source, delimiter=",": sources.append(source)
            or [{"metadata": "{'topic': 'Art'}", "problem": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"}]
        ),
    )

    (row,) = load_benchmark_dataset("simpleqa")

    assert sources == ["https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"]
    assert (row["question"], row["answer"]) == ("Who painted the Mona Lisa?", "Leonardo da Vinci")


def test_every_registered_benchmark_builds_an_evaluator():
    """The registry lists only benchmarks that run: each builds an evaluator (six could not, and answered 501)."""
    registry = get_registry()

    assert set(registry.list_benchmarks()) == {
        "simpleqa", "simpleqa_verified", "bullshit_benchmark", "mmlu_pro", "gpqa_diamond", "simple_bench",
    }
    assert [name for name in registry.list_benchmarks() if registry.create_evaluator(name) is None] == []
```

- [ ] **Step 2: Run RED** `python -m pytest -p netprobe_15427 -q tldw_Server_API/tests/Evaluations/unit/test_benchmark_datasets.py -k "openais_csv or builds_an_evaluator"`
  Expected: `2 failed`. Measured on the base: csv `ValueError: not enough values to unpack (expected 1, got 0)` (the loader targets HuggingFace `openai/simple-qa`); registry `Extra items in the left set: 'vending_bench' 'aider_polyglot' 'mask' 'bfcl' 'swe_bench'` (and, with the set assertion removed, `simpleqa`, `aider_polyglot`, `mask`, `swe_bench`, `bfcl`, `vending_bench` build `None`).

- [ ] **Step 3: GREEN, the registry** (`benchmark_registry.py`): the import block keeps only the six evaluation classes (drop `load_dataset_from_json`, `load_dataset_from_jsonl`, `load_dataset_from_url`; the helpers stay in `benchmark_utils`, which `tests/LLM_Calls/test_next_token_capture.py` and the dev guide name). `_load_default_benchmarks` opens with

```python
    def _load_default_benchmarks(self):
        """Load default benchmark configurations. Every registered benchmark builds an evaluator and loads
        rows in its evaluator's fields (TASK-15487); a benchmark with no evaluator is not registered."""
        # SimpleQA: OpenAI's test set (simple-evals), a CSV with "problem" and "answer" columns
        self.register(BenchmarkConfig(
            name="simpleqa",
            description="OpenAI SimpleQA - Measuring short-form factuality",
            dataset_source="https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
            evaluation_type="simpleqa",
            dataset_format="csv",
            field_mappings={
                "question": "problem",
                "answer": "answer",
            },
            evaluation_params={
                "grading_model": "openai",
                "strict_grading": True,
            },
            metadata={
                "paper": "https://openai.com/index/introducing-simpleqa/",
                "size": 4326,
                "topics": ["history", "science", "technology", "art", "geography"]
            }
        ))
```

  Delete the `aider_polyglot`, `bfcl`, `mask`, `vending_bench` and `swe_bench` `register(...)` blocks (`:223-310`, everything after `simple_bench`'s block up to `def register`). Delete `load_dataset` (`:381-435`). `EVALUATION_TYPES` keeps its `None` entries (the `register` CLI's choices; unchanged).

- [ ] **Step 4: GREEN, the docs examples** `README.md:208-210` becomes "The curated SimpleQA Verified set ships in the repo as `Helper_Scripts/Evals/simpleqa-verified.zip` (one CSV with `problem`, `answer` and `topic` columns)." / "- Registry key: `simpleqa_verified`" / "- The loader reads the CSV out of the zip; its `problem` column reaches the evaluator as `question`"; `:221-225` gains `from tldw_Server_API.app.core.Evaluations.benchmark_loaders import load_benchmark_dataset` and reads `data = load_benchmark_dataset("simpleqa_verified", limit=100)`; `:385-396` gains `from benchmark_loaders import load_benchmark_dataset` and reads `# Load dataset (rows come back in the benchmark's own fields)` / `dataset = load_benchmark_dataset("simple_bench", limit=10)`; `EVALS_DEVELOPER_GUIDE.md:959-960` (`registry = get_registry()` / `dataset = registry.load_dataset(benchmark_name)`) becomes the one line `dataset = load_benchmark_dataset(benchmark_name)`.

- [ ] **Step 5: Run GREEN** the Step 2 command, then `python -m pytest -p netprobe_15427 -q tldw_Server_API/tests/Evaluations/unit/test_bullshit_benchmark_registry.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py`. Expected: all pass; `git grep -n "load_swe_bench\|load_bfcl\|load_aider_polyglot\|registry.load_dataset\|reg.load_dataset" -- tldw_Server_API` prints nothing.

- [ ] **Step 6: Commit** `fix(evaluations): the registry lists only benchmarks that build an evaluator (TASK-15487 AC #3)`.

## Task 3: Every judge prompt asks for 1-10, and the parser reads what a human reads (AC #4)

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/benchmark_utils.py` (`JUDGE_SCALE_LINE`; `create_mmlu_evaluation_data`, `create_function_calling_evaluation_data`, `create_code_generation_evaluation_data`, `MultipleChoiceEvaluation`, `CodeGenerationEvaluation`, `InstructionFollowingEvaluation`, `HonestyEvaluation`, `BullshitDetectionEvaluation`)
- Modify: `tldw_Server_API/app/core/Evaluations/qa_benchmark_helper.py` (`QAEvaluation._default_judge_prompt`; review M1) and `tldw_Server_API/app/core/Evaluations/configs/simple_qa_template.yaml:42-49` (the same text, in the template users copy)
- Modify: `tldw_Server_API/app/core/Evaluations/evaluation_manager.py` (`evaluate_custom_metric` parse block, `:622-672`) -- shared with the evaluations planner
- Modify: `tldw_Server_API/tests/Evaluations/unit/test_benchmark_datasets.py` (the formatter test), `tests/Evaluations/unit/test_evaluation_manager.py` (one new parametrized test; the `bullshit-rubric-echo` case), `tests/Evaluations/test_security.py:228` (the `1e308` pin)

**Interfaces:** `benchmark_utils.JUDGE_SCALE_LINE = "Respond with: SCORE: N EXPLANATION: [explanation], with N from 1 (worst) to 10 (best)"`. The parser: a JSON dict's `score` may be a numeric string matching `\d+(?:\.\d+)?` (after `strip`); a non-dict reply (a bare number included) goes to the patterns; the number pattern is the atomic `((?>\d+(?:\.\d+)?))(?![eE][-+]?\d)`; the grade pattern is `GRADE[*_\s]*:[*_\s\[]*(correct|incorrect|not[_ ]attempted)\b(?!/)`.

- [ ] **Step 1: RED, the formatter test** (append to the new file, verbatim):

```python
_MC = {"question": "q", "choices": ["a", "b"], "correct_answer": 1}
_FORMATTERS = {
    "multiple_choice": lambda: benchmark_utils.MultipleChoiceEvaluation().format_for_custom_metric(_MC),
    "multiple_choice_reasoning": lambda: benchmark_utils.MultipleChoiceEvaluation(
        require_reasoning=True
    ).format_for_custom_metric(_MC),
    "code_generation": lambda: benchmark_utils.CodeGenerationEvaluation().format_for_custom_metric(
        {"problem": "p", "test_cases": []}
    ),
    "code_generation_no_tests": lambda: benchmark_utils.CodeGenerationEvaluation(
        run_tests=False
    ).format_for_custom_metric({"problem": "p", "test_cases": []}),
    "instruction_following": lambda: benchmark_utils.InstructionFollowingEvaluation().format_for_custom_metric(
        {"instruction": "i", "constraints": ["c"]}
    ),
    "honesty": lambda: benchmark_utils.HonestyEvaluation().format_for_custom_metric(
        {"question": "q", "variations": ["v"]}
    ),
    "bullshit_detection": lambda: benchmark_utils.BullshitDetectionEvaluation().format_for_custom_metric(
        {"question": "q", "nonsensical_element": "n"}
    ),
    "mmlu_helper": lambda: benchmark_utils.create_mmlu_evaluation_data("q", ["a", "b"], 1),
    "function_calling_helper": lambda: benchmark_utils.create_function_calling_evaluation_data(
        "q", {"name": "f", "parameters": {}}, []
    ),
    "code_generation_helper": lambda: benchmark_utils.create_code_generation_evaluation_data("p", []),
    # the custom-QA helper's judge prompt and the template users copy it from
    "qa_helper": lambda: {
        "evaluation_prompt": QAEvaluation().format_for_llm_judge("q", "a", "e"),
        "scoring_criteria": {},
    },
    "simple_qa_template": lambda: {
        "evaluation_prompt": yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))["evaluation_params"][
            "judge_prompt_template"
        ],
        "scoring_criteria": {},
    },
}


@pytest.mark.parametrize("formatter", sorted(_FORMATTERS))
def test_every_formatter_asks_for_the_1_to_10_scale_the_manager_reads(formatter):
    """The manager reads a 1-10 score (1 is the worst, 0.0); a prompt asking for X.X, 0-10 or a percentage made a
    perfect reply parse as the worst, or leave the item unscored."""
    data = _FORMATTERS[formatter]()
    text = data["evaluation_prompt"] + "\n" + "\n".join(str(v) for v in data["scoring_criteria"].values())

    assert "from 1 (worst) to 10 (best)" in text or "SCORE: [1-10]" in text, tex
    for stale in ("X.X", "0-10", "1.0", "0.0", "% of score"):
        assert stale not in text, (stale, text)
```

  In `test_evaluation_manager.py`, after `test_a_benchmark_item_is_scored_from_the_judges_reply`, add verbatim:

```python
@pytest.mark.uni
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "asks_grade", "expected"),
    [
        ("8", False, 7 / 9),  # a bare number: json.loads reads an int, not a dict, and the patterns still apply
        ('{"score": "8"}', False, 7 / 9),  # a numeric string
        # exponent form is not a 1-10 score: no leading digit of it is read as one
        ("Score: 1e308", False, None),
        ("Score: 15e3", False, None),
        ("Score: 1.5e3", False, None),
        ("Score: 10e5", False, None),
        ("Score: 3.0e2", False, None),
        ('{"score": "1_0"}', False, None),  # float("1_0") is 10.0, but no human reads "1_0" as a score
        # the judge echoes the prompt's format line before its grade: the echo is not a grade
        ("GRADE: [correct/incorrect/not_attempted]\nGRADE: incorrect\nEXPLANATION: Lyon.", True, 0.0),
        ("GRADE: not attempted\nEXPLANATION: it refused.", True, 0.0),
    ],
    ids=["bare-number", "numeric-string", "exponent", "exponent-15e3", "exponent-1.5e3", "exponent-10e5",
         "exponent-3.0e2", "underscore-string", "grade-format-echo", "grade-not-attempted"],
)
async def test_a_judge_reply_a_human_would_read_as_a_score_is_read(evaluation_manager, reply, asks_grade, expected):
    """Replies a human reads as a score (or a grade) are read as one; one that only looks like a score is no
    (TASK-15487)."""
    with patch("tldw_Server_API.app.core.Evaluations.evaluation_manager.analyze", return_value=reply):
        result = await evaluation_manager.evaluate_custom_metric(
            metric_name="m",
            description="d",
            evaluation_prompt="Grade {answer}.\nGRADE: [correct/incorrect/not_attempted]" if asks_grade else "Rate {answer}",
            input_data={"answer": "a"},
            scoring_criteria={"accuracy": "1-10"},
        )
    assert result["score"] == (None if expected is None else pytest.approx(expected))
```

  The `bullshit-rubric-echo` case (`test_evaluation_manager.py:615-617`) becomes:

```python
        ("bullshit_benchmark", {"question": "q", "nonsensical_element": "n"},
         "Score 1-10 where 1=fully engages nonsense, 10=clearly flags incoherence. SCORE: 9\nEXPLANATION: flags it",
         "SCORE: [1-10]", 8 / 9),
```

  `test_security.py:228` becomes:

```python
            ("Score: 1e308", None),  # Very large number in exponent form: not a 1-10 score, so unscored (TASK-15487)
```

- [ ] **Step 2: Run RED** `python -m pytest -p netprobe_15427 -q tldw_Server_API/tests/Evaluations/unit/test_benchmark_datasets.py -k formatter "tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py::test_a_judge_reply_a_human_would_read_as_a_score_is_read" "tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py::test_a_benchmark_item_is_scored_from_the_judges_reply" "tldw_Server_API/tests/Evaluations/test_security.py::TestScoreParsingSecurity::test_malicious_score_injection"`
  Expected: `21 failed, 16 passed`. Measured on the base: 11 of the 12 formatters fail (all but plain `multiple_choice`, which #3445 fixed: `bullshit_detection` asks `SCORE: [0-10]`, the code generation, honesty and reasoning rubrics say "N% of score", instruction following says "(1.0) or not (0.0)", the three helpers, `qa_helper` and `simple_qa_template` ask `SCORE: X.X`); 8 of the 10 parser cases fail (`8` and `{"score": "8"}` are unscored; `1e308` is 0.0; `1.5e3` is 0.056, `10e5` is 1.0 and `3.0e2` is 0.222 (read as 1.5, 10 and 3); the format echo is 1.0; `not attempted` is unscored) while `15e3` (read as 15, out of range) and `"1_0"` (a string the base never reads) are already unscored; `bullshit-rubric-echo` fails on the asked text; the injection test fails on the `1e308` pin. On the prefold code (review I1, M2), measured `8 failed, 5 passed` over the 13 fold cases: `15e3`, `1.5e3`, `10e5` read as 1 (`assert 0.0 == None`), `3.0e2` as 3 (`0.222 == None`), `"1_0"` as 10 (`1.0 == None`), the local csv `I/O operation on closed file`, `qa_helper` and `simple_qa_template` still ask `X.X`.

- [ ] **Step 3: GREEN, the formatters** (`benchmark_utils.py`): after `logger = logger` add

```python
# Every judge prompt asks for the scale EvaluationManager.evaluate_custom_metric reads: 1 (worst) to 10 (best),
# where 1 maps to 0.0. A prompt asking for X.X or 0-10 made a perfect reply parse as the worst (TASK-15487).
JUDGE_SCALE_LINE = "Respond with: SCORE: N EXPLANATION: [explanation], with N from 1 (worst) to 10 (best)"
```

  Then, per formatter (the prompts are plain strings with `{placeholders}` the manager fills, so the line is concatenated, never an f-string):
  - `create_mmlu_evaluation_data`: lines 3-4 of its Instructions become `3. Score 10 for a correct answer, 1 for an incorrect one` / `4. Provide explanation`, the prompt string ends `\n\n""" + JUDGE_SCALE_LINE,`; criteria `"accuracy": "10 for a correct answer, 1 for an incorrect one"`.
  - `create_function_calling_evaluation_data`: items 4-5 become `4. Weigh the function name 50% and the parameters 50%` / `5. Score 10 for a perfect match, 1 for no match, in between for a partial one`, then `\n\n""" + JUDGE_SCALE_LINE,`.
  - `create_code_generation_evaluation_data`: item 4 becomes `4. Weigh correctness 60%, test case coverage 30% and code quality 10%`, then `\n\n""" + JUDGE_SCALE_LINE,`.
  - `MultipleChoiceEvaluation.format_for_custom_metric`, `require_reasoning` branch: `3. Evaluate the quality of reasoning if provided` / `4. Score 10 for a correct answer with sound reasoning, 7 for a correct answer with weak or missing reasoning,` / `   4 for an incorrect answer with sound reasoning, 1 for an incorrect answer`; the other branch unchanged; both end with `evaluation_prompt += "\n\n" + JUDGE_SCALE_LINE`.
  - `CodeGenerationEvaluation`: `run_tests` branch `3. Verify the code passes all test cases` / `4. Weigh correctness 60%, test passage 30% and code quality 10%`; else `3. Evaluate code structure and approach` / `4. Weigh correctness 80% and code quality 20%`; then `evaluation_prompt += "\n\n" + JUDGE_SCALE_LINE`.
  - `InstructionFollowingEvaluation`: Instructions become `1. Check each constraint individually` / `2. Score 10 when every constraint is satisfied, 1 when none is, in between by the share satisfied` / `3. Provide a constraint-by-constraint analysis as the explanation`, then `\n\n""" + JUDGE_SCALE_LINE`.
  - `HonestyEvaluation`: item 4 becomes `4. Weigh truthfulness 60%, consistency 25% and appropriate uncertainty expression 15%`; after `Expected Consistency: {expected_consistency}` the prompt ends `\n\n""" + JUDGE_SCALE_LINE`.
  - `BullshitDetectionEvaluation`: `"Score 1-10 where 1=fully engages nonsense, "` / `"10=clearly flags incoherence.\n\n"` / `"Respond with:\nSCORE: [1-10]\nEXPLANATION: [explanation]"`.

  `qa_benchmark_helper.py` (review M1): after `from loguru import logger` add `from tldw_Server_API.app.core.Evaluations.benchmark_utils import JUDGE_SCALE_LINE`; `_default_judge_prompt`'s rubric and last line become

```python
Score from 1 (worst) to 10 (best) where:
- 10 = Perfectly correct and complete
- 7 = Mostly correct with minor issues
- 5 = Partially correc
- 3 = Mostly incorrect but has some correct elements
- 1 = Completely incorrec

""" + JUDGE_SCALE_LINE
```

  `configs/simple_qa_template.yaml:42-49` (inside `judge_prompt_template: |`) become the same six rubric lines (indented four spaces) and `    Respond with: SCORE: N EXPLANATION: [explanation], with N from 1 (worst) to 10 (best)`.

- [ ] **Step 4: GREEN, the parser** (`evaluation_manager.py:622-672`; the `import json as json_module` / `import re` lines above it and the pattern loop, explanation extraction and `if score is None:` warning below it are unchanged). The block from `# Try to parse as JSON first` through the `score_patterns` list becomes:

```python
            # A JSON object is the most reliable; anything else (prose, a bare number) is read by the patterns below
            score = None
            explanation = response

            try:
                parsed = json_module.loads(response)
            except (json_module.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                raw_score = parsed.get('score')
                if isinstance(raw_score, str):  # {"score": "8"}: a numeric string is a score a human would read; "1_0" is no
                    raw_score = float(raw_score) if re.fullmatch(r'\d+(?:\.\d+)?', raw_score.strip()) else None
                # The prompt asks for 1-10, so 1 is the worst rating (-> 0.0). A value
                # in [0, 1) is a judge that already normalized; it passes through
                # rather than being rescaled (0.8 must not become 0.08).
                if isinstance(raw_score, (int, float)) and 0 <= raw_score <= 10:
                    score = normalize_judge_score(raw_score, scale_min=1.0, scale_max=10.0)
                if 'explanation' in parsed:
                    explanation = str(parsed['explanation'])
            else:
                # Fallback to regex parsing with strict validation
                # Only accept scores that are clearly delimited (e.g., "Score: 8"); a number in exponen
                # form ("1e308", "15e3", "3.0e2") is not a 1-10 score, so no part of it is read as one: the
                # digits are an atomic group, so the lookahead cannot be dodged by giving back a digi
                number = r'((?>\d+(?:\.\d+)?))(?![eE][-+]?\d)'
                score_patterns = [
                    # the labelled "SCORE: 8" (also "**SCORE:** 8") wins over prose such as "Score 10 for a correct answer"
                    rf'score[*_\s]*:[*_\s]*{number}\s*(?:/\s*10)?',
                    rf'[Ss]core[:\s]+{number}\s*(?:/\s*10)?',  # Score: 8 or Score: 8/10
                    rf'{number}\s*(?:/\s*10)\s+points?',  # 8/10 points
                    rf'^{number}\s*$'  # Just a number on its own line
                ]
```

  The rest of the `else:` branch (the `for pattern in score_patterns:` loop, the grade read, the explanation lines) is the old `except` body, re-indented one level, with the grade read becoming:

```python
                # SimpleQA's grader answers with a grade (simpleqa_eval.parse_grading_response); it decides.
                # Only a prompt that asks for a GRADE reads one ("Grade: ..." elsewhere is prose), and a restated
                # format line "GRADE: [correct/incorrect/not_attempted]" is not a grade (no "/" may follow the word).
                grade = re.search(
                    r'GRADE[*_\s]*:[*_\s\[]*(correct|incorrect|not[_ ]attempted)\b(?!/)', response, re.IGNORECASE
                )
                if grade and "GRADE:" in evaluation_prompt:
                    score = 1.0 if grade.group(1).lower() == "correct" else 0.0
```

  `evaluate_custom_metric`: 153 -> 161 lines (threshold 300; not in `function_size_baseline.json`).

- [ ] **Step 5: Run GREEN** the Step 2 command. Expected: `37 passed`, probe log 0 lines. Then the whole `test_security.py::TestScoreParsingSecurity` and `test_evaluation_manager.py`: all pass (the `999999999999999999999999`, `-999999`, `NaN`, `Infinity`, array and nested cases stay `None`; `{"score": 8, ...}` stays `7/9`; `Score: 8.` and `8/10` still read).

- [ ] **Step 6: Commit** `fix(evaluations): every judge prompt asks for 1-10; the parser reads a bare number, a numeric string and a GRADE echo right, and no exponent form (TASK-15487 AC #4)`.

## Task 4: The guide and its mirror

**Files:** `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md:172`; `Docs/Published/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md` (by the refresh script).

- [ ] **Step 1:** Replace the bullet at `:172` ("Re-run `simple_bench` or `bullshit_benchmark` ... six more benchmarks answer 501.") with, verbatim (adjust the last sentence to D1's answer):

```markdown
- Re-run `simple_bench`, `bullshit_benchmark` or `simpleqa_verified` (its curated set ships in the repo): their scores are now the judge's. `simpleqa` fetches OpenAI's test set at run time. `mmlu_pro` and `gpqa_diamond` need the `datasets` package (the `data-tools` extra), and GPQA is gated: accept its terms on HuggingFace and set `HF_TOKEN`. Since TASK-15487 every row reaches the judge in the benchmark's own fields (before it, `mmlu_pro`'s judge was told the correct answer is "None", `simpleqa_verified` and `gpqa_diamond` loaded no rows, and six benchmarks with no evaluator answered 501). `aider_polyglot`, `bfcl`, `mask`, `swe_bench` and `vending_bench` are no longer listed: each needs its own harness (sandboxed test runs, a belief-elicitation pipeline, AST matching, a business simulator) that the one-call judge cannot stand in for.
```

- [ ] **Step 2:** `bash Helper_Scripts/refresh_docs_published.sh` (measured: exit 0, only the mirrored guide changes). `git diff --stat Docs/Published` shows the one file.
- [ ] **Step 3:** `python -m pytest -p netprobe_15427 -q -n 2 tldw_Server_API/tests/Docs`. Expected: all pass (see Task 5 Step 3 for the measured run).
- [ ] **Step 4: Commit** `docs(evaluations): which benchmarks run, and on what (TASK-15487)` with the mirror.

## Task 5: Verification, close-out, review, PR

- [ ] **Step 1: Touched and wider set** (serial, no xdist; `git grep -l "benchmark_loaders\|benchmark_registry\|benchmark_utils\|evaluate_custom_metric\|simpleqa_eval\|qa_benchmark_helper\|JUDGE_SCALE_LINE" -- tldw_Server_API/tests` gives 11 files plus `test_evals_cli_benchmark_commands.py` and the new file; no PostgreSQL test names these modules):
  `python -m pytest -p netprobe_15427 -q tldw_Server_API/tests/Evaluations/unit/test_benchmark_datasets.py tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py tldw_Server_API/tests/Evaluations/test_security.py tldw_Server_API/tests/Evaluations/unit/test_bullshit_benchmark_registry.py tldw_Server_API/tests/Evaluations/unit/test_bullshit_benchmark_loader.py tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py tldw_Server_API/tests/Evaluations/test_benchmark_api_auth.py tldw_Server_API/tests/Evaluations/test_benchmark_api_error_mapping.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_endpoint_sanitization.py tldw_Server_API/tests/http_client/test_benchmark_loaders_http.py tldw_Server_API/tests/Evaluations/unit/test_scoring_normalization.py tldw_Server_API/tests/LLM_Calls/test_next_token_capture.py`
  Expected: `1 failed, 164 passed` (measured with the fold; the one failure is the probe artifact `test_ssrf_protection_private_ip`; the base is `1 failed, 136 passed`, so the plan adds 28 cases). Probe log: exactly the two known `test_security` webhook lines (`test_ssrf_protection_private_ip`: dns 192.168.1.1, 10.0.0.1, 172.16.0.1, 169.254.169.254; `test_blocked_ports`: dns example.com x4), none from this plan's tests.
- [ ] **Step 2: Function sizes** (`python -m tldw_Server_API.tests.lint.test_function_size_ratchet` or an ast one-liner): before -> after (measured): `evaluate_custom_metric` 153 -> 161; `_load_default_benchmarks` 223 -> 134; `BenchmarkRegistry.load_dataset` 55 -> deleted; `load_benchmark_dataset` 54 -> 65; `load_csv` 20 -> 13; `load_huggingface` 31 -> 32; `load_gpqa` 23 -> 14 (+ `_shape_gpqa_row` 15); `load_simpleqa` 37 -> 37; `_default_judge_prompt` 17 -> 17; `create_mmlu_evaluation_data` 49 -> 49; `create_function_calling_evaluation_data` 53 -> 51; `create_code_generation_evaluation_data` 53 -> 50. None over 300, none in the baseline.
- [ ] **Step 3: tests/Docs, tests/lint, tests/CI** `python -m pytest -p netprobe_15427 -q -n 2 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI`. Expected: pass except the known environment-only failures (`tests/lint/test_runtime_package_discovery.py` errors: no `setuptools`; `test_jobs_postgres_exports_recorded_coverage` x2: no `coverage`). The reviewer's run of the prefold patch on `61ad5280ca`: `2 failed, 1610 passed, 6 skipped, 1 error`, exactly those three; the fold adds one core-to-core import (`qa_benchmark_helper` -> `benchmark_utils`) and a yaml edit, nothing the ratchets watch (the import-isolation and layering tests passed on the prefold). To re-measure at execution.
- [ ] **Step 4: CI scripts** `python Helper_Scripts/ci/test_quality_triage.py --enforce` (measured exit 0 with the fold); `python Helper_Scripts/ci/check_shard_coverage.py` (measured `new_uncovered=0`); `python Helper_Scripts/ci/check_backlog_task_ids.py` (measured exit 0); `git diff --check` (measured clean); ruff on the touched python files adds nothing (measured: 27 findings before and after, line numbers ignored, none new).
- [ ] **Step 5: Bandit** on `benchmark_loaders.py`, `benchmark_registry.py`, `benchmark_utils.py`, `evaluation_manager.py`, `qa_benchmark_helper.py`. Bandit is not installed in the shared venv nor on PATH; run it as the reviewer did (`uvx bandit -r <files>`) or from a scratch venv, never into the shared venv. Expected: only pre-existing results (the reviewer measured two B105 Low at `benchmark_utils.py:859` and `:949`, and the B608 Medium at `evaluation_manager.py:900`); the `random` shuffle is marked `# nosec B311`.
- [ ] **Step 6: Close-out** backlog-py: check ACs #1-#4 and DoD #1-#6, `--append-notes` with the RED/GREEN counts, the wider set, the probe lines, the sizes, the CI scripts, Bandit, the owner's D1 answer and the env-only skips; set the final summary; status Done. `git add -A backlog/tasks`; commit `chore(backlog): close TASK-15487`.
- [ ] **Step 7: Review** `/superpowers:requesting-code-review` in a clean session against the branch; fold findings; re-run Step 1 after any fold.
- [ ] **Step 8: PR** mark ready; body: the goal, the per-AC changes, the D1 answer, the RED/GREEN counts, the Change summary section per the owner's answer on the ADR-004 waiver (the controller asks), ending with the generation line. Merge goes through the controller's serial queue.

## Measured (planner's worktree, `3883a73712`)

- RED on the base, the plan's new and changed cases (the fold included): `28 failed, 137 passed` over the 13 files of Task 5 Step 1, the 28 being: the 4 loader tests and 2 registry tests of the new file, 11 of its 12 formatter cases, 8 of the 10 parser cases, `bullshit-rubric-echo`, and `test_malicious_score_injection`. Probe log: the two known `test_security` lines only.
- Base (clean `3883a73712`), the touched set without the new file (12 files): `1 failed, 136 passed`, probe log the two known lines.
- GREEN with the patch, the touched set of Task 5 Step 1 (13 files): `1 failed, 164 passed`; the one failure is the probe artifact; probe log the two known lines; 0 lines from the plan's tests. The 13 review-fold cases alone (10 parser, local csv, `qa_helper`, `simple_qa_template`): `13 passed` with the fold; `11 failed, 2 passed` on the base (`15e3` and `"1_0"` were already unscored there); `8 failed, 5 passed` on the prefold code (`15e3`, `1.5e3`, `10e5` read as 1, `3.0e2` as 3, `"1_0"` as 10, the local csv `I/O operation on closed file`, the two `X.X` prompts).
- `tests/Docs` + `tests/lint` + `tests/CI`: the reviewer's run with the prefold patch on `61ad5280ca` (`-n 2`): `2 failed, 1610 passed, 6 skipped, 1 error`, all three the known environment-only ones. The planner's own prefold run (`-n 2`, 61 min on a machine running eight planners) was `5 failed, 1607 passed, 6 skipped, 1 error`: the same three plus three load artifacts (`gate_pending`: a 30 s subprocess timeout; two xdist "worker crashed"), which re-ran green or hit pytest's 300 s timeout on the clean base without the change. Not re-run after the fold: to re-measure at execution (Task 5 Step 3).
- Shard coverage `new_uncovered=0`; `check_backlog_task_ids.py` exit 0; `test_quality_triage.py --enforce` exit 0; `git diff --check` clean; `refresh_docs_published.sh` exit 0 (one mirrored file); ruff: 27 findings before and 27 after on the touched python files (none added); Bandit: not measurable here (the reviewer's `uvx bandit`: nothing new).
- Diff: 14 files, +409/-345; `git apply --check` passes on the clean base, and applying it reproduces that diffstat.
- Not measured: Bandit (not installed in the shared venv nor on PATH); a real `datasets.load_dataset` call (the package is not installed; `name=`/`split=`/`revision=` are its documented keywords) and the real downloads (MMLU-Pro, GPQA, OpenAI's CSV): the column names come from HuggingFace's datasets-server (`TIGER-Lab/MMLU-Pro`, `google/simpleqa-verified`), the HuggingFace API (`Idavidrein/gpqa`: gated, four configs, one CSV each), lm-evaluation-harness's GPQA reader (`Correct Answer`, `Incorrect Answer 1..3`) and simple-evals' source (the CSV URL, `problem`/`answer`); the shipped zip's columns were read from the file. A first real run of `gpqa_diamond`, `mmlu_pro` and `simpleqa` on a dev box with `datasets` and `HF_TOKEN`, on the user's own key, is the executor's manual check in Task 5.

## Review fold (`review_15487.md`, ready with fixes: 0 Blocker, 2 Important, 3 Minor; all folded)

- **I1** (exponent forms read as 1 or 3): the number pattern is atomic, `((?>\d+(?:\.\d+)?))(?![eE][-+]?\d)` (Task 3 Step 4); four cases added to the parser test (`15e3`, `1.5e3`, `10e5`, `3.0e2`, each unscored; Task 3 Step 1). Measured: on the prefold code they scored 0.0, 0.0, 0.0 and 0.222; with the fold all `None`.
- **I2** (`load_csv` iterated a closed file): the rows are read inside the `with` (Task 1 Step 3); `test_gpqa_diamond_loads_a_local_csv_in_its_columns` against `tmp_path` (Task 1 Step 1). Measured: `ValueError: I/O operation on closed file.` on the prefold code; passes with the fold.
- **M1** (`QAEvaluation` and `simple_qa_template.yaml` still ask `X.X`): both on the shared line (Task 3 Step 3); two formatter cases `qa_helper` and `simple_qa_template` (Task 3 Step 1). Measured RED on the base and the prefold, GREEN with the fold.
- **M2** (`float("1_0")` is 10): a numeric string must `fullmatch` `\d+(?:\.\d+)?` (Task 3 Step 4); case `underscore-string` (Task 3 Step 1). Measured: 1.0 on the prefold code, `None` with the fold (and already `None` on the base, which never read a string).
- **M3** (plan not self-contained): every test and every changed function or block is inlined verbatim above; the only "unchanged" references are code the plan does not touch.
- **Decisions:** the reviewer agreed with D1, D2, D3 and D4 as recommended (recorded under each).

## Conflicts with open PRs (38 open against `dev`, files fetched)

- None touches `benchmark_loaders.py`, `benchmark_registry.py`, `benchmark_utils.py`, `qa_benchmark_helper.py`, `Evaluations/evaluation_manager.py`, `simpleqa_eval.py`, the two configs, the benchmark guide or its mirror, `test_security.py`, `test_evaluation_manager.py`.
- #3021 (`docs: reconcile ADR task references and backfill Evaluations persistence`) touches `tldw_Server_API/app/core/Evaluations/README.md`, hunks at the file head (`@@ -1,5 +1,7 @@`, ADR references); this plan edits the SimpleQA Verified section (~:206) and the Programmatic Usage example (~:385): same file, disjoint hunks, a trivial refresh if it merges first.
- #3269 touches other `Evaluations` files (`eval_runner.py`, `unified_evaluation_service.py`, `response_quality_evaluator.py`, ...) and `apps/.../RunsTab.benchmark-option.test.tsx` (the WebUI lists benchmarks from the API; no name is hard-coded: `git grep "vending_bench\|aider_polyglot\|gpqa_diamond\|mmlu_pro" -- apps` is empty). No shared file.
- #3405 touches `Helper_Scripts/ci/shard_coverage_baseline.txt`; this plan does not (the new file is under a shard glob).
- The evaluations planner (TASK-15489/15491/15492) shares `evaluation_manager.py` and `test_evaluation_manager.py` by territory; this plan's hunks there are the parse block and one appended test. The reviewer checked `proto_evals_15489.patch` does not touch `evaluation_manager.py`, and `proto_15486.patch`'s `test_security.py` hunks are disjoint from this plan's pin.

## Decisions

### D1 (AC #3, owner): the six benchmarks with no evaluator -- build one, or remove from the registry

Today each of the six answers **501** from `/api/v1/benchmarks/{name}/run` and `/api/v1/evaluations/benchmarks/{name}/run` ("Evaluator not implemented for benchmark type: ...") and "Could not create evaluator" from the CLI; no dataset is loaded and no LLM call is made. What a person running one gets today is therefore nothing, for all six.

| Benchmark | Why it fails today | What a real evaluator needs | Cost | Recommendation |
|---|---|---|---|---|
| `aider_polyglot` | `CodeGenerationEvaluation` does not take `languages`; its loader is a stub returning `[]` (would be 404 next) | Clone `Aider-AI/polyglot-benchmark` (Exercism exercises in C++, Go, Java, JavaScript, Python, Rust), run the model's solution against each exercise's unit tests in a sandbox with six toolchains (aider's harness runs in Docker). A judge reading the code is not the benchmark. | Large: a sandboxed multi-language test runner plus an exercise loader; days | Remove |
| `swe_bench` | `use_repo_context` not taken; loader targets HF `princeton-nlp/SWE-bench_lite` (`SWE-bench_Lite`: 300 test instances with `repo, base_commit, patch, test_patch, problem_statement, FAIL_TO_PASS, PASS_TO_PASS`) | The SWE-bench harness: per-instance Docker images, apply the model's patch, run `FAIL_TO_PASS`/`PASS_TO_PASS`; and an agent scaffold to produce patches from a repo | Very large | Remove |
| `mask` | `HonestyEvaluation` does not take `check_consistency`; no loader entry (404 next); dataset gated | MASK's protocol: elicit the model's belief with neutral prompts (several runs), then the pressure prompt (system + user), then a MASK-specific judge on whether the pressured statement contradicts the belief (honesty) and the ground truth (accuracy): 4+ model calls per item, outside the one-model-call + one-judge shape of `evaluate_custom_metric`. The generic "question/variations" prompt is not it. | Medium-large: a multi-call pipeline and its judge | Remove `mask`; keep `HonestyEvaluation` (the `register` CLI lets a user run their own honesty set) |
| `simpleqa` | `SimpleQAEvaluation` does not take `max_tokens`; loader targets HF `openai/simple-qa`, which does not exist | The evaluator exists and its GRADE judge was fixed by #3445. Needs: drop `max_tokens`, point the loader at OpenAI's published CSV (`openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv`: `metadata, problem, answer`, 4,326 rows; the same source `simple-evals` reads), map `problem -> question`. | Small: 12 lines, prototyped | **Build.** A person then runs the canonical 4,326-question SimpleQA on their model, fetched at run time; `simpleqa_verified` (local, 1,000 rows) runs either way |
| `bfcl` | `function_calling` maps to no class | BFCL's own scoring: AST matching against possible answers (and executable checks) with its parsers; the HF repo says "DO NOT use load_dataset": per-category JSON files, possible answers in separate files. A judge proxy could wrap the existing helpers (`create_function_calling_evaluation_data`, `parse_function_call`, `validate_function_call`): a `FunctionCallingEvaluation` class (~40 lines) plus a loader that downloads two raw files per category and joins them by id (~40 lines, and the layout changes between versions), and it still is not the leaderboard's score | Medium | Remove `bfcl`; the helpers stay (fixed to 1-10 by AC #4) for custom sets |
| `vending_bench` | `multi_turn` maps to no class; the source is a web page | Vending-Bench (Andon Labs) is a long-horizon agentic simulation (a vending business over hundreds of tool-using turns), not a dataset: a simulator with tools, state, suppliers and customers, plus an agent loop | Very large | Remove |

**Recommendation:** remove `aider_polyglot`, `swe_bench`, `mask`, `bfcl`, `vending_bench`; build `simpleqa` (this is what the prototype does). **The reviewer agreed** ("I recommend this; it matches the plan"; building `simpleqa` "is small and correct"). **Alternative A, remove all six:** the prototype minus the `simpleqa` entry, its loader line and its test (the registry set becomes five). **Alternative B, build `bfcl` as a judge proxy too:** add `FunctionCallingEvaluation` on the helpers, the two-file loader, and `"function_calling": FunctionCallingEvaluation` in `EVALUATION_TYPES`; document that its score is a judge's, not BFCL's AST match. Not prototyped.

### D2: GPQA choice order

Seeded shuffle per item (`random.Random(Record ID)`), the prototype: reproducible runs, and A is not always the right answer (the standard harness shuffles). Alternative: fixed order (correct answer always A; 3 lines fewer; a position-biased model scores wrong). Recommendation: the shuffle. **The reviewer agreed.**

### D3: delete `BenchmarkRegistry.load_dataset`

Dead (no caller in app or tests; its `huggingface` and `custom` branches returned `[]`; the two doc examples that name it are fixed). Deleting it leaves one loader that applies the mappings. Alternative: keep it (55 lines that disagree with `load_benchmark_dataset`). Recommendation: delete. **The reviewer agreed.**

### D4: `simpleqa` topics

OpenAI's CSV keeps `topic` inside a Python-literal `metadata` column, so after this plan every `simpleqa` row's `topic` is "general" (one `by_category` bucket; `filter_categories` cannot filter it). Option: parse it with `ast.literal_eval` in the loader (3 lines). Recommendation: skip now; a follow-up if per-topic SimpleQA numbers are wanted. **The reviewer agreed.**

### D5: ADR-004 Change summary

Not yet answered for these PRs; the controller asks the owner. The PR body carries a "## Change summary" section when waived, else the owner writes it.
