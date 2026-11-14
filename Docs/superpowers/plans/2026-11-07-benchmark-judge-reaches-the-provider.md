# The benchmark judge reaches the provider and its reply is scored (TASK-<JB>) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each benchmark item gets the score the judge gives the model's answer, not the 0.5 default. Also file TASK-<NT>, which takes every test that reaches a real provider or host offline, and, if the owner agrees (Decision 4), TASK-<BD>, which makes the registered benchmarks run on correctly shaped data.

**Architecture:** Four small edits and three guards, in two files. The judge's call reaches the provider, and the parser reads the reply that each benchmark's formatter asks for:
1. the judge's `analyze` call takes its JSON input as literal text;
2. the score regex ignores case;
3. the multiple-choice formatter asks for the manager's 1-10 scale;
4. a SimpleQA `GRADE:` decides the score.

The guards, from the plan review (Review fold below):
- G1: a failed judge call scores 0.0 with the error, not a silent 0.5;
- G2: a colon-anchored, markdown-tolerant score pattern is tried first, so a judge that restates the rubric, or writes `**SCORE:** 10`, parses right;
- G3: the grade is read only when the prompt asks for one, and `GRADE: [incorrect]` parses.

One parametrized test drives real registered benchmarks through their own formatter, the real `evaluate_custom_metric` and the real `analyze`, down to a stubbed adapter. Stored results are left alone. The benchmark guide says that runs from before the fix are 0.5 placeholders.

**Tech Stack:** Python 3.12, pytest (pytest-asyncio), backlog-py.

**Spec:** the planner brief `brief_plan_judge.md` (plans dir). Evidence:
- `review_plan_15426_ab.md` (finding B-4, probe `probe_judge.py`);
- `plan_15426_b.md` (V5, Decision 4, and the `TASK-<F_B>` draft that this PR replaces);
- `review_plan_15426_cd.md` ("Pre-existing, not caused by C or D").

**Measured on:** `origin/dev` at `7a020da14ab28e220fab652269723e3a2a4abbcb` (`7a020da14a`, "Merge pull request #3432"), in a detached worktree that was reverted afterwards. `file:line` citations are for `7a020da14a` unless marked "PR".

**Prototype:** `proto_judge.patch` (plans dir): 5 files, +103/-4. It holds every code, test and docs change in this plan, the `Docs/Published` mirror included. It does not hold the plan copy or the task files, and its docs text keeps the `TASK-<JB>` placeholder.

---

## Root cause (verified on `7a020da14a`)

| # | Finding | Evidence | Fix |
|---|---|---|---|
| R1 | **The judge never reaches the provider.** The judge's call `asyncio.to_thread(analyze, api_name, json.dumps(runtime_input_data), full_prompt, api_key, ...)` (`Evaluations/evaluation_manager.py:589-597`; the input at `:592`) passes no `input_is_literal_text`. So `analyze` (`LLM_Calls/Summarization_General_Lib.py:664-673`) calls `extract_text_from_input` (`:846-887`). That function parses the JSON string into a dict and keeps only `title`, `description`, `transcription`, `segments`, `text` or `content` (`:866-885`). The benchmark formatters build other keys:<br>- multiple choice: `question`, `choices`, `correct_answer`, `response` (`Evaluations/benchmark_utils.py:506-511`);<br>- SimpleQA: `question`, `expected_answer`, `model_response`, `topic` (`simpleqa_eval.py:68-73`);<br>- the bullshit benchmark: `question`, `nonsensical_element`, `response` (`benchmark_utils.py:806-810`).<br>The text is therefore empty. `analyze` returns `Error: Could not extract text content.` (`:673`) and calls no adapter. That string holds no score, so the item gets the default (`evaluation_manager.py:648-651`, `score = 0.5`). | **Measured** with `judge_probe.py` (scratchpad). Every registered benchmark that builds an evaluator went through its real formatter and the real manager to a stub adapter: `bullshit_benchmark`, `gpqa_diamond`, `mmlu_pro`, `simple_bench` and `simpleqa_verified`.<br>Each made **1 adapter call (the model's answer)** and scored **0.5** for every judge reply. `block_live_network()` logged no attempt.<br>The existing tests missed it, because every `evaluate_custom_metric` test patches `analyze` or `asyncio.to_thread` (`tests/Evaluations/unit/test_evaluation_manager.py:252-328`, `:565-581`; `test_scoring_normalization.py:124-140`). | `input_is_literal_text=True` on the judge's call |
| R2 | **With R1 alone, every item still scores 0.5.** Every formatter asks for `Respond with: SCORE: X.X ...` (`benchmark_utils.py:133`, `:351`, `:407`, `:500`, `:586`). The parser's first pattern is `[Ss]core[:\s]+...` (`evaluation_manager.py:624`), and it is searched with `re.MULTILINE` only (`:630`). It never matches `SCORE`. | **Measured** with R1 only:<br>- 2 calls (`model+judge`);<br>- 0.5 for `SCORE: 1.0`, `SCORE: 0.0` and `SCORE: 10`;<br>- only `Score: 8` parsed (0.778). | `re.MULTILINE \| re.IGNORECASE` |
| R3 | **With R1 and R2, a correct multiple-choice answer scores 0.0.** The formatter says `3. Score 1.0 for correct answer, 0.0 for incorrect` (`benchmark_utils.py:496`, the `require_reasoning=False` branch, which `simple_bench` uses, `benchmark_registry.py:219`). The manager appends `Provide a score from 1-10` and normalizes on 1-10 (`evaluation_manager.py:585`, `:617`, `:636`). `normalize_judge_score(1.0, scale_min=1, scale_max=10)` is 0.0, because exactly 1 is on the scale (`scoring.py:25-29`, `:55-64`, TASK-13328).<br>`mmlu_pro` and `gpqa_diamond` (`require_reasoning=True`, `:174`, `:198`) name no scale but ask for `X.X`. A judge that thinks in fractions writes `SCORE: 1.0` for a perfect answer, which also scores 0.0. | **Measured** by ablation (below). | the formatter asks for 10 / 1 and for `N from 1 (worst) to 10 (best)` |
| R4 | **SimpleQA answers with a grade.** Its prompt asks for `GRADE: [correct/incorrect/not_attempted]` (`simpleqa_eval.py:121`, `:142`), and the manager never reads a grade. `SimpleQAEvaluation.parse_grading_response` (`:251-292`) maps grades to scores, but no route calls it. **No route reaches SimpleQA today** (Decision 4), so this edit is correct but reaches no user. | **Measured** with R1 and R2: `GRADE: correct` gave 0.5. | a `GRADE:` match decides, only when the prompt asks for a `GRADE:` (G3): correct gives 1.0, anything else 0.0 (`parse_grading_response`'s mapping) |
| R5 | **A failed judge call scores a silent 0.5.** Once R1 lands, every item makes a real judge call, up to 16 items in parallel and two calls each. `analyze` returns `Error: ...` for a provider error, a timeout, a rate limit, a missing judge model, and a truncated or empty reply. No score pattern matches, so the item gets the 0.5 default and the route counts it a success. The model-response branch already returns 0.0 with the error (`evaluation_manager.py:565-571`). | **Measured** (review probe): a truncated reply (`finish_reason: length`) and an adapter that raises both gave 0.5. | G1: `if isinstance(response, str) and response.startswith("Error:")` returns 0.0 with the error text and `raw_output: None` |
| R6 | **The parser takes the first `score <n>` in the reply.** R3's new rubric line is "Score 10 for a correct answer, 1 for an incorrect one"; a judge that restates it before its own score line matches `[Ss]core[:\s]+(\d+)` on the rubric, not the score. Markdown labels (`**SCORE:** 10`, `Score: **10**`) match nothing and fall to 0.5. | **Measured** (review probe): "Score 10 for a correct answer, 1 for an incorrect one. ... SCORE: 1" gave 1.0 (a wrong answer scored right); `**SCORE:** 10` gave 0.5. | G2: `r'score[*_\s]*:[*_\s]*(\d+(?:\.\d+)?)\s*(?:/\s*10)?'` first; the existing patterns stay as fallbacks |

**Every edit is pinned.** An ablation undoes one edit at a time and runs the new test (13 cases; measured):

| Edit undone | Failing cases |
|---|---|
| `input_is_literal_text` | all 13 |
| `re.IGNORECASE` | `[mc-correct]`, `[mc-incorrect]`, `[mc-reasoning]`, `[rubric-echo]`, `[markdown-label]`, `[markdown-value]`, `[mc-grade-is-prose]` |
| the `GRADE:` branch | the four `[simpleqa-*]` |
| the grade as a fallback only (`if score is None and grade`) | `[simpleqa-grade-decides]` |
| the `with N from 1 (worst) to 10 (best)` clause | `[mc-reasoning]` |
| the 10 / 1 rubric line | `[mc-correct]`, `[mc-incorrect]`, `[no-score]`, `[judge-error]`, `[rubric-echo]`, `[markdown-label]`, `[markdown-value]`, `[mc-grade-is-prose]` |
| G1, the judge-error branch | `[judge-error]` |
| G2, the colon-anchored pattern | `[rubric-echo]`, `[markdown-label]`, `[markdown-value]` |
| G3, the `"GRADE:" in evaluation_prompt` gate | `[mc-grade-is-prose]` |
| G3, the bracket tolerance | `[simpleqa-bracketed-grade]` |

**Not changed (known, out of scope):**
- **`_generate_model_response`** (`evaluation_manager.py:736-757`) still sends its prompt through the JSON sniffing. Its prompt is plain text, unless a dataset's whole `question`, `instruction`, `query` or `prompt` value is a JSON object.
- **The bullshit benchmark** asks for 0-10 (`benchmark_utils.py:803`) and is read on 1-10. It is the most reachable benchmark (built in, and the guide's own example). 10 maps exactly, but 0 reads as the worst rating 1 (so `Score: 1` is 0.0, not 0.1), and a middle score is off by at most 0.06: 5 gives 0.444, not 0.5 (measured). No test pins its prompt. The one-line fix (`"Score 1-10 where 1=fully engages nonsense, 10=clearly flags incoherence.\nRespond with: SCORE: N EXPLANATION: [explanation]"`) is left to TASK-<BD> (Decision 4), with the other formatters below.
- **The other formatters' fractional rubrics.** Code generation (`aider_polyglot`, `swe_bench`), honesty (`mask`), instruction following, function calling and the module helpers `create_mmlu_evaluation_data`, `create_function_calling_evaluation_data` and `create_code_generation_evaluation_data` still ask `SCORE: X.X`, with fractional or percentage rubrics. With R2, a perfect `SCORE: 1.0` from one of these now parses to 0.0 (it was 0.5). None is reachable today (below). TASK-<BD> moves them to 1-10 when it makes those benchmarks runnable.
- **Which benchmarks reach the judge** (measured with `create_evaluator` for all 11 registered benchmarks and `load_benchmark_dataset`; the gpqa and Hugging Face rows are by code):
  - **Six get 501 before any LLM call.** `aider_polyglot` (`CodeGenerationEvaluation` takes no `languages`), `mask` (`HonestyEvaluation` takes no `check_consistency`), `simpleqa` (`SimpleQAEvaluation` takes no `max_tokens`) and `swe_bench` (no `use_repo_context`) build no evaluator. `bfcl` and `vending_bench` have no class at all (`benchmark_registry.py:80-81`).
  - **`simpleqa_verified` gets 404.** It builds an evaluator, but `load_benchmark_dataset("simpleqa_verified")` returns 0 rows (it has no loader entry, and the routes pass no source).
  - **`gpqa_diamond` most likely gets 404** (by code, not measured: needs the network). It calls `load_huggingface("Idavidrein/gpqa")` with no config name, then filters on an `item["subset"]` column that GPQA rows do not have.
  - **`mmlu_pro` and `gpqa_diamond` need the `datasets` package** (the `evaluation` extra). A default install lacks it, so both get 404 there.
  - **`mmlu_pro` reaches the judge with no answer key.** `load_benchmark_dataset` ignores `config.field_mappings`, so a real HF row reaches the formatter as `choices: ''` and `correct_answer: 'None'` (measured), and the judge is asked to compare with "the correct answer: None". Its scores look real and carry no information.
  - **So on a default install only `simple_bench` (it downloads from raw.githubusercontent.com) and `bullshit_benchmark` (built in) reach the judge with meaningful data.** See Decision 4.

## Stored results and what presents them

- **Stored rows.** `save_results` defaults to True on both routes (`benchmark_api.py:54`, `evaluations_benchmarks.py:41`). It writes `{"summary", "scores"}` and the first 10 items to `internal_evaluations` (`evaluations_benchmarks.py:205-217`, `benchmark_api.py:337`). Every score stored by `dev` before this fix is 0.5, or 0.0 for an item that failed.
- **What reads them back: nothing a user sees.**
  - `EvaluationManager.get_history`, `get_evaluation`, `list_evaluations` and `compare_evaluations` (`evaluation_manager.py:366`, `:816`, `:836`, `:456`) have no route caller. `git grep "manager\.(get_evaluation|list_evaluations|get_history|compare_evaluations)\("` finds only Prompt Studio's own manager.
  - The WebUI History tab reads `/api/v1/evaluations/history`, which lists the unified `evaluations` table (`Evaluations_DB.py:1161`), not `internal_evaluations`.
- **What presents scores: the live run's response only.**
  - The WebUI Runs tab prints the response as raw JSON (`apps/packages/ui/src/components/Option/Evaluations/tabs/RunsTab.tsx:619-621`).
  - The benchmark guide documents the response with an example `average_score` of 0.84 (`Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md:117`, mirrored in `Docs/Published`).
- **This PR changes nothing in code (Decision 2).** The guide gains a troubleshooting entry: results from before the fix are 0.5 placeholders, and the two benchmarks whose scores are now meaningful are `simple_bench` and `bullshit_benchmark`. TASK-<JB>'s notes say the same.

## Decisions

1. **ADR-004 Change summary (owner).**
   - This PR is AI-authored.
   - The 2026-11-07 waiver ("Yes, waive for all") covers the TASK-15426 PRs, and this PR is not one of them.
   - *Recommendation: extend the waiver to this PR.* It is a small bug fix with tests, in the same family. It was split out of TASK-15426's PR B on the owner's instruction of 2026-11-07.
   - Record the answer as a full `## Change summary` section: the waiver text, or the owner's own summary. The controller asks.
2. **Stored results: nothing in code (ruling; the brief's recommendation).**
   - Why:
     - a placeholder 0.5 cannot be told apart from a real 0.5;
     - no user-facing view reads those rows (see above);
     - re-scoring would need the model's answers, which were never stored.
   - The guide and the task notes say that results from before the fix are placeholders.
   - Alternatives:
     - flag the stored rows by date, which needs a migration and a merge date known in code;
     - delete them, which destroys a user's records.

     Both are larger than the docs entry, and both reach only people who read the guide anyway.
3. **R2-R4 belong in this PR (ruling).**
   - The brief names R1, but R1 alone leaves every item at 0.5 (R2).
   - R1 and R2 together turn a correct `simple_bench` answer into 0.0 (R3), which is worse than the placeholder.
   - "Every item scores 0.5" is fixed only when all four land. Each edit is one to four lines and has its own failing case.
   - R1, R2 and R3 ship together: R1+R2 alone turns a correct `simple_bench` answer into 0.0, and R1+R2+R3 is correct for every benchmark that can run (`simple_bench` and `bullshit_benchmark`). R4 reaches no user today (Decision 4); it stays because it is correct, tested, and a few lines, now gated on the prompt asking for a `GRADE:` (G3), so it cannot override a valid score in another benchmark.
4. **Benchmarks run on correctly shaped data (owner; recommendation: file it, at normal priority, as TASK-<BD>).**
   - After this PR the judge works, but most registered benchmarks still cannot run on correct data (see "Which benchmarks reach the judge" in Not changed): `mmlu_pro` produces plausible scores judged against "correct answer: None", `simpleqa_verified` gets 404, `gpqa_diamond` most likely gets 404, and six more get 501. On a default install only `simple_bench` and `bullshit_benchmark` reach the judge. R4's grade edit is therefore correct but reaches no user.
   - Scope of the follow-up:
     - apply `config.field_mappings` in `load_benchmark_dataset`;
     - a `simpleqa_verified` loader entry;
     - the gpqa config name and subset;
     - the six benchmarks with no evaluator (their `evaluation_params` name options the classes do not take: a product choice, not a typo fix);
     - move the remaining `SCORE: X.X` formatters, and the bullshit benchmark's 0-10, to 1-10.
   - Task 0 Step 3b files TASK-<BD> only if the owner agrees. If the owner declines, drop that step and every `TASK-<BD>` in the plan, and record the answer in TASK-<JB>'s close-out notes.
   - The guide must not advise re-running `mmlu_pro` as if its scores now mean something; it names `simple_bench` and `bullshit_benchmark` only.
5. **TASK-<NT> is filed, not fixed (brief).**
   - Its list is the measured one (Task 0 Step 3).
   - Two RAG stream cases are owned elsewhere: TASK-15482 owns `RAG/test_restricted_postgres_media_retrieval.py`'s stream tests, and TASK-15427 (#3439) owns `test_rag_stream_parity.py::test_rag_streaming_generation_provider_override`. They are named in its description, not included.

## Global Constraints

Verbatim from the brief:
- Tests make no real outbound network requests; stub the provider adapter or its transport; run every test you touch with the probe plugin (`PYTHONPATH=<plans dir> NETPROBE_LOG=<file> ... -p netprobe_15427`).
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions must not grow.
- Backlog: edit tasks with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`); `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files. New ids come from `zsh <plans dir>/task_next.sh` (prints the HIGHEST id in use; new = that + 1, then + 2) after `mkdir <scratchpad>/id_lock`; release once the PR's file list shows the tasks. Placeholders in the plan; do NOT allocate ids now.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- Redis: never FLUSHALL or SCRIPT FLUSH. PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`; drop only your own objects. The shared PostgreSQL at 127.0.0.1:5432 is down: do not restart it; say which PG cases you could not run.
- Mirrored docs (`Docs/Published`) need `bash Helper_Scripts/refresh_docs_published.sh` in the same PR.
- CI checks before the PR, in the last task: `tests/Docs`, `tests/lint`, `tests/CI`, backlog-py's tests; shard coverage; `test_quality_triage.py --enforce`; route-auth ratchet; OpenAPI export check; Bandit on touched source; every test file that names a touched symbol (git grep), baseline on origin/dev; `scope_predicate_baseline.txt`; `check_backlog_task_ids.py` exit 0.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific:
- **Paths.**
  - `$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Write it out in each call: shell variables do not survive between agent calls.
  - `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`.
  - `<scratchpad>` is the executing session's scratchpad.
- **Ids.**
  - Under one lock, `TASK-<JB>` (the judge bug) is the printed value + 1, `TASK-<NT>` (the network tests) is the printed value + 2, and `TASK-<BD>` (benchmarks on correct data, only if the owner agrees to Decision 4) is the printed value + 3.
  - `task_next.sh` does not take the lock. Take it with `mkdir`. If the directory exists, wait; never break another session's lock.
  - Write the real ids wherever this plan says `<JB>` or `<NT>`, in the guide text too.
  - Before the last commit, `git grep -n -E "TASK-<(JB|NT|BD)>" -- ':!Docs/superpowers/plans'` must print nothing.
- **`<pytest>`** stands for `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir> NETPROBE_LOG=<scratchpad>/netprobe_<label>.log TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -p netprobe_15427`. After each run, check the log: it must be empty, or hold only the base lines named here.
- **Probe ceilings.**
  - The plugin works per test window.
  - It counts `getaddrinfo` on a non-loopback IP literal as an attempt, although no lookup leaves the machine. So `test_security.py::TestWebhookSecurityValidation::test_ssrf_protection_private_ip` fails under it, on the base and with the PR alike. `review_plan_15426_ab.md` recorded it as a probe artifact.
  - A server subprocess (`e2e/`, `server_e2e_tests/`) is never probed.
- **Function sizes and scope baseline.**
  - `function_size_baseline.json` has no entry for `evaluation_manager.py`, `benchmark_utils.py` or `Summarization_General_Lib.py`. `evaluate_custom_metric` is not frozen; it grows by 13 lines.
  - `scope_predicate_baseline.txt` names none of the touched files (grep count 0).
- **Git.** Stage named files. `git diff --cached --check` must print nothing before each commit.
- **Go cache.** If a run leaves a read-only Go module cache, clean it with `chmod -R u+w tldw_Server_API/Config_Files/acp_runner_home && git clean -ffdx -- tldw_Server_API/Config_Files/acp_runner_home`. Never `rm -rf` the directory. The last `git status --short` must be empty.

## Review Focus

1. **A judge that writes `SCORE: 1.0` for a correct answer anyway** (it ignores the new instruction). That reply scores 0.0, because exactly 1 on a 1-10 scale is the worst rating (`scoring.py`, TASK-13328). The instruction itself is the only defence.
   - Expected: R3's prompt names the scale twice, in its rubric line and in `Respond with:`.
   - Pinned by `[mc-correct]`, by `[mc-reasoning]` (its `asked` assertion), and by the unchanged `test_evaluation_manager_custom_metric_uses_its_declared_1_to_10_scale[Score: 1]` (0.0).
2. **A reply with neither a score nor a grade.** Expected: the 0.5 default and the `Could not parse valid score` warning, as before. Pinned by `[no-score]`.
   - **A failed judge call** is not such a reply. Expected: 0.0 with the `Error: ...` text as the explanation (G1). Pinned by `[judge-error]`.
   - **A judge that restates the rubric** ("Score 10 for a correct answer ... SCORE: 1"), or writes `**SCORE:** 10`. Expected: the labelled line decides (G2). Pinned by `[rubric-echo]`, `[markdown-label]`, `[markdown-value]`.
3. **A SimpleQA reply with both a grade and a score.** The manager's own suffix asks for a 1-10 score, so a judge may write both.
   - Expected: the grade decides, because SimpleQA's metric is the share of correct answers.
   - Pinned by `[simpleqa-grade-decides]`: `GRADE: correct` with `Score: 7` gives 1.0, not 0.667.
   - A multiple-choice reply with "Grade:" in prose does not read as a grade (G3). Pinned by `[mc-grade-is-prose]`.
4. **Existing custom-metric replies keep their scores.** `Score: 8`, `{"score": 0.8}` and the other 1-10 and pass-through cases parse as before.
   - A judge input with a `text` key now sends the whole input JSON, not only `text`, followed by the same formatted prompt. TASK-15426 B's test still sees two calls.
   - Expected: no score changes.
   - Pinned by the unchanged `TestCustomMetrics` and `test_scoring_normalization.py` cases (both files run whole in Task 1 Step 5).
5. **A benchmark run now makes two LLM calls per item, not one.** The judge's call is new spend, and it is checked like any other call: the boundary's 402 and the call-site gate are unchanged.
   - Expected: a test that drove a real adapter through a benchmark would now try a second call.
   - Pinned by the wider set under the probe: the base and the PR show the same 2 attempt lines (Task 2 Step 2).

---

## Files

| File | Task | Change |
|---|---|---|
| `tldw_Server_API/app/core/Evaluations/evaluation_manager.py` | 1 | In `evaluate_custom_metric`: the judge's `analyze` call gets `input_is_literal_text=True` (PR `:597`); the score regex gets `re.IGNORECASE` (PR `:631`); a `GRADE:` match decides when the prompt asks for one (PR `:650-654`); G1: a failed judge call scores 0.0 (PR `:601-606`); G2: the colon-anchored pattern (PR `:631-632`) |
| `tldw_Server_API/app/core/Evaluations/benchmark_utils.py` | 1 | `MultipleChoiceEvaluation.format_for_custom_metric`: `:496` and `:500` ask for the 1-10 scale |
| `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py` | 1 | `test_a_benchmark_item_is_scored_from_the_judges_reply` (13 cases), appended at the end of the file |
| `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md` + its `Docs/Published` mirror | 1 | a troubleshooting entry, "Every item scored 0.5" (names the benchmarks whose scores are now meaningful, and `Error:` explanations) |
| `Docs/superpowers/plans/2026-11-07-benchmark-judge-reaches-the-provider.md`; `backlog/tasks/task-<JB> - ...md`, `task-<NT> - ...md`, and `task-<BD> - ...md` if the owner agrees | 0, 2 | this plan; the tasks |

## Test commands

- **Focused:** `<pytest> -q tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py::test_a_benchmark_item_is_scored_from_the_judges_reply`.
- **Wider set:** every test file that names a touched symbol. Run it with label `<label>`:

```zsh
git grep -l -E 'evaluate_custom_metric|format_for_custom_metric|MultipleChoiceEvaluation|benchmark_utils|benchmark_registry|EvaluationManager|evaluation_manager|benchmark_api|evaluations_benchmarks|/benchmarks' \
  -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$' > <scratchpad>/wider.txt
<pytest> -q -n 4 --tb=no -rfE $(cat <scratchpad>/wider.txt)
```

  Measured: 31 files, about 2 minutes. No PostgreSQL case among them runs here (`TLDW_TEST_NO_DOCKER=1`, PG down).

---

## Task 0: Ids, two tasks, branch, draft PR, baseline

**Files:** `backlog/tasks/` (two new tasks, three if the owner agrees to Decision 4; TASK-<JB> set to In Progress); `Docs/superpowers/plans/2026-11-07-benchmark-judge-reaches-the-provider.md`.

- [ ] **Step 1: Allocate the ids and branch.**
  - `mkdir <scratchpad>/id_lock` (Global Constraints).
  - `git fetch origin dev --quiet`, then run `zsh <plans dir>/task_next.sh` from the repository root. `<JB>` is the printed value + 1, `<NT>` is the printed value + 2, and `<BD>` is the printed value + 3 (used only if the owner agrees to Decision 4).
  - `ls backlog/tasks backlog/completed backlog/archive/tasks | grep -E "^task-(<JB>|<NT>|<BD>) "` must print nothing.
  - `git switch -c fix/benchmark-judge-<JB> origin/dev`.
- [ ] **Step 2: File TASK-<JB>** (write the ids literally):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Benchmark judge calls never reach the provider, so every benchmark item scores 0.5" \
  --id TASK-<JB> -l evaluations,backend,bug \
  -d "User-facing: every /benchmarks and /evaluations/benchmarks run on dev scores each item 0.5 whatever the model answered, so average_score, min_score, max_score and by_category carry no information. Found by TASK-15426 PR B's plan review (B-4); fixed before that PR on the owner's instruction (2026-11-07). Which benchmarks it reaches: on a default install only simple_bench and bullshit_benchmark (mmlu_pro and gpqa_diamond need the datasets package; mmlu_pro rows are not field-mapped, so its judge is told the correct answer is 'None'; simpleqa_verified loads 0 rows and gpqa_diamond most likely 404s; six benchmarks get 501). Root cause (7a020da14a): EvaluationManager.evaluate_custom_metric sends the judge json.dumps(runtime_input_data) through Summarization_General_Lib.analyze without input_is_literal_text (Evaluations/evaluation_manager.py:589-597), so extract_text_from_input (Summarization_General_Lib.py:846-887) parses it and keeps only title, description, transcription, segments, text or content. The benchmark formatters build question/choices/correct_answer/response (benchmark_utils.py:506-511; SimpleQA question/expected_answer/model_response/topic, simpleqa_eval.py:68-73), so analyze returns 'Error: Could not extract text content.' (:673) without a provider call, and the parser defaults to 0.5 (evaluation_manager.py:648-651). Behind it: the score regex is case-sensitive ([Ss]core, :624, searched at :630) while every formatter asks for 'SCORE: X.X'; the multiple-choice formatter asks for 1.0/0.0 (benchmark_utils.py:496) while the manager reads 1-10, where 1 is the worst rating (scoring.py, TASK-13328), so a correct answer would score 0.0; SimpleQA answers 'GRADE: ...' (simpleqa_eval.py:121), which the manager never reads. Measured: each registered benchmark that builds an evaluator made one adapter call per item (the model's answer) and scored 0.5 for every judge reply. Stored results (save_results, internal_evaluations) from before the fix are 0.5 placeholders; no route or WebUI view reads them back." \
  --ac "A benchmark item's judge call reaches the provider with the formatted prompt and the model's answer, with a test that runs a registered benchmark's own formatter through the real evaluate_custom_metric and analyze to a stubbed adapter and sees two adapter calls" \
  --ac "The item's score is read from the judge's reply in the format its formatter asks for (multiple choice on 1-10: 10 gives 1.0, 1 gives 0.0, also when the judge restates the rubric or writes markdown labels; a SimpleQA grade: correct 1.0, otherwise 0.0), not the 0.5 default, with tests; a reply with neither still scores 0.5; a failed judge call (provider error, timeout, rate limit, truncated or empty reply) scores 0.0 with the error text, not 0.5" \
  --ac "Benchmark results from before the fix are documented as 0.5 placeholders (the benchmark guide and its Docs/Published mirror), naming simple_bench and bullshit_benchmark as the benchmarks whose scores are now meaningful; no code rewrites stored results"
```

- [ ] **Step 3: File TASK-<NT>.** The list below was measured on `7a020da14a`. If `origin/dev` has moved since then, run the five in-scope files under the probe on the new branch, and add any log line that is not on the list to the description.

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Every test that reaches a real provider or host runs offline, and a session-wide blocker fails any new attempt" \
  --id TASK-<NT> -l tests,ci \
  -d "Owner request (2026-11-07), filed by TASK-<JB>. A test that reaches a real provider or host can spend a real key, flake with the network, and leak a request; on dev these tests mostly pass because the code under test swallows the failure. Measured on 7a020da14a by running each test inside tldw_Server_API/tests/helpers/offline_network.block_live_network() (a pytest plugin that blocks and logs every non-loopback DNS lookup and connect). 28 tests: (1) Prompt_Management_NEW/integration/test_prompt_improvement_api.py::test_llm_quota_refusal_answers_402_not_retryable_provider_unavailable: api.openai.com. (2) Media_Ingestion_Modification/test_media_processing.py, 15: TestProcessVideos::test_process_video_url_success and ::test_process_video_multi_status_mixed (www.youtube.com); TestProcessAudios::test_process_audio_url_success_no_analysis_no_chunking (cdn.pixabay.com; fails when blocked) and ::test_process_audio_multi_status_mixed (example.com); TestProcessPdfs::test_process_pdf_url_success (www.w3.org); TestProcessDocuments::test_process_doc_multiple_success and ::test_process_doc_overrides_and_options (raw.githubusercontent.com); TestProcessEbooks::test_process_ebook_url_success_no_analysis_no_chunking, ::test_process_ebook_multiple_success, ::test_process_ebook_multi_status_mixed, ::test_process_ebook_overrides, ::test_process_ebook_with_analysis_mocked and ::test_process_ebook_options_extraction_method[basic], [filtered], [markdown] (filesamples.com). (3) Media_Ingestion_Modification/test_media_contextual_integration.py, 7 (example.com): TestMediaEndpointContextualIntegration::test_add_media_contextual_defaults_from_config, ::test_add_media_with_contextual_chunking_enabled, ::test_add_media_with_contextual_chunking_disabled, ::test_contextual_chunking_preserves_other_options and ::test_contextual_chunking_with_different_media_types[document-sentences], [ebook-ebook_chapters], [pdf-sentences]. (4) Evaluations/test_security.py, 2: TestWebhookSecurityValidation::test_blocked_ports (example.com) and ::test_ssrf_protection_private_ip (getaddrinfo on 192.168.1.1, 10.0.0.1, 172.16.0.1 and 169.254.169.254: numeric hosts, so no lookup leaves the machine, but the blocker counts them and the test fails under it). (5) Evaluations/test_production_features_integration.py, 3 (example.com): TestWebhookManager::test_webhook_registration and ::test_webhook_unregistration, TestIntegration::test_full_evaluation_flow_with_features. Owned elsewhere, not in this task: RAG/test_restricted_postgres_media_retrieval.py's two stream tests (api.openai.com; the [sqlite] cases measured, the [postgresql] cases need PostgreSQL) by TASK-15482; RAG_NEW/integration/test_rag_stream_parity.py::test_rag_streaming_generation_provider_override (api.groq.com) by TASK-15427 (#3439). How many tests a session-wide blocker would fail today (a dry count; a full run of 5379 test files, about 45,000 test functions, was too slow locally): 31 tests attempted in the 137 files run under the probe (the files above, a 31-file Evaluations set and a random 100-file sample, seed 15427): the 28 above and the 3 owned elsewhere. Statically, 729 test files outside e2e/, server_e2e_tests/ and frontend_e2e/ hold a non-loopback URL literal (12,885 test functions); in the random sample 1 of 50 such files and 0 of 50 others attempted, which puts the unmeasured rest at roughly 15 files, i.e. tens of tests, not hundreds. The first CI run with the blocker gives the exact number. Probe ceilings: per test window only; a thread that outlives its test is not seen; a server subprocess is not probed." \
  --ac "Each listed test stubs its download, provider or DNS call and passes inside block_live_network() with no attempt" \
  --ac "A session-wide blocker in the test config (an autouse fixture in tldw_Server_API/tests/conftest.py around tests/helpers/offline_network.block_live_network(), asserting no attempt at teardown as LLM_Adapters/conftest.py's no_live_network does) makes a new outbound DNS lookup or connect fail loudly in CI, with a test that shows the failure; getaddrinfo on a numeric non-loopback host passes through (no lookup leaves the machine) while its connect stays blocked; loopback is decided with ipaddress.ip_address(host).is_loopback (as tests/helpers/netblock_13956 does), not the four exact strings of offline_network._LOOPBACK (127.0.0.2, ::ffff:127.0.0.1, 0.0.0.0 and the machine's own hostname count as live otherwise), and the host of a developer's configured TEST_DATABASE_URL / TEST_DB_HOST is allowed; a function-scoped autouse fixture does not cover session- or module-scoped fixture setup, so the AC names that ceiling and the notes say which such fixtures were checked" \
  --ac "Tests that legitimately need the network opt out: by a new registered live_network marker, by the existing external_api, local_llm_service and provider_inventory markers, and by directory for e2e against a live server (e2e/, server_e2e_tests/, frontend_e2e/); loopback is never blocked; the opt-out is documented in Docs/Development/Testing.md and tldw_Server_API/tests/README.md" \
  --ac "A full-suite CI run passes with the blocker on, and the notes record how many tests it failed before the stubs"
```

- [ ] **Step 3b: File TASK-<BD> only if the owner agrees to Decision 4** (write the ids literally; otherwise skip this step and drop every `TASK-<BD>` from the plan):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "The registered benchmarks run on correctly shaped data" \
  --id TASK-<BD> -l evaluations,backend \
  -d "Follow-up of TASK-<JB> (owner decision, 2026-11-07). After the judge fix, on a default install only simple_bench and bullshit_benchmark reach the judge with meaningful data. mmlu_pro rows are not field-mapped (load_benchmark_dataset ignores config.field_mappings), so its judge is told the correct answer is 'None' and its scores look real but carry no information. simpleqa_verified builds an evaluator but load_benchmark_dataset returns 0 rows (no loader entry), so it gets 404. gpqa_diamond calls load_huggingface('Idavidrein/gpqa') with no config name and filters on a 'subset' column GPQA rows lack, so it most likely gets 404. mmlu_pro and gpqa_diamond need the datasets package (the evaluation extra). aider_polyglot, mask, simpleqa and swe_bench cannot build an evaluator (evaluation_params the classes do not take: languages, check_consistency, max_tokens, use_repo_context) and bfcl and vending_bench have no class, so all six get 501. The formatters other than multiple choice still ask 'SCORE: X.X' with fractional rubrics (a perfect 1.0 now parses to 0.0 on the manager's 1-10 scale), and bullshit_benchmark asks 0-10 while it is read on 1-10 (Score: 1 gives 0.0, 5 gives 0.444)." \
  --ac "load_benchmark_dataset applies config.field_mappings, with a test that a real mmlu_pro row reaches the judge with its choices and correct answer" \
  --ac "simpleqa_verified has a loader entry and loads rows; gpqa_diamond loads with the right config name and subset; each with a test (stubbed loader, no network)" \
  --ac "The six benchmarks with no evaluator (aider_polyglot, mask, simpleqa, swe_bench, bfcl, vending_bench) build one and run, or are removed from the registry, as the owner decides" \
  --ac "Every formatter asks for the 1-10 scale the manager reads (code generation, honesty, instruction following, function calling, the module helpers, and bullshit_benchmark), with tests"
```

- [ ] **Step 4: Start TASK-<JB>:** `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-<JB> -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-11-07-benchmark-judge-reaches-the-provider.md. Fixed before TASK-15426 PR B, which touches evaluate_custom_metric too (owner, 2026-11-07)."` The owner's answers to Decisions 1 and 4 are recorded at close-out (Task 2 Step 4), not here: the notes are append-only.
- [ ] **Step 5: Plan copy, commit, draft PR.**
  - Copy this plan to `Docs/superpowers/plans/2026-11-07-benchmark-judge-reaches-the-provider.md` unchanged. The copy keeps the placeholders.
  - `git add Docs/superpowers/plans/2026-11-07-benchmark-judge-reaches-the-provider.md && git add -A backlog/tasks && git diff --cached --check`. It must print nothing.
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` must exit 0.
  - Commit (message 0).
  - `git push -u origin fix/benchmark-judge-<JB>`.
  - `gh pr create --draft --base dev --title "fix(evaluations): the benchmark judge reaches the provider and its reply is scored (TASK-<JB>)"`, with a one-line body that names TASK-<JB>, TASK-<NT> (and TASK-<BD>, if filed) and ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - Release `<scratchpad>/id_lock` once `gh pr view <n> --json files` lists both task files.
- [ ] **Step 6: Baseline** the wider set (label `base`) on the unchanged branch.
  - Expected, measured on `7a020da14a`: `1 failed, 665 passed` (about 1.5 minutes).
  - The failure is `test_security.py::TestWebhookSecurityValidation::test_ssrf_protection_private_ip`, the probe artifact (Global Constraints).
  - The log has 2 lines, both in `test_security.py`: `::test_ssrf_protection_private_ip` (four numeric hosts) and `::test_blocked_ports` (`example.com` ×4).
  - Record the result in the ledger. Compare any later failure with this baseline, and never dismiss one as environment-only without it.

## Task 1: The judge reaches the provider and its reply is scored (ACs #1-#3)

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/evaluation_manager.py` (`evaluate_custom_metric`, `:589-654` with the PR)
- Modify: `tldw_Server_API/app/core/Evaluations/benchmark_utils.py` (`MultipleChoiceEvaluation.format_for_custom_metric`, `:496`, `:500`)
- Test: `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py` (appended at the end of the file)
- Docs: `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md` and its mirror

**Interfaces:**
- Consumes:
  - `benchmark_registry.get_registry().create_evaluator(name) -> BaseEvaluation | None`;
  - `BaseEvaluation.format_for_custom_metric(item) -> dict`, with the keys `name`, `description`, `evaluation_prompt`, `input_data` and `scoring_criteria`;
  - `EvaluationManager.evaluate_custom_metric(metric_name, description, evaluation_prompt, input_data, scoring_criteria, api_name="openai", api_key=None) -> dict`, with `score`, `explanation` and `raw_output`;
  - the adapter seam `Summarization_General_Lib.get_registry().get_adapter(provider).chat(request, timeout=None)`. The prompt is in `request["messages"][-1]["content"]`.
- Produces: no signature change.

- [ ] **Step 1: Write the failing test** at the end of `test_evaluation_manager.py`:

```python
_MC_ITEM = {"question": "What is 2 + 2?", "choices": ["3", "4"], "correct_answer": 1}
_QA_ITEM = {"question": "What is the capital of France?", "answer": "Paris"}
_MC_ASKS = "Score 10 for a correct answer, 1 for an"
_QA_ASKS = "GRADE: [correct/incorrect"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("benchmark", "item", "reply", "asked", "expected"),
    [
        ("simple_bench", _MC_ITEM, "SCORE: 10 EXPLANATION: B.", _MC_ASKS, 1.0),
        ("simple_bench", _MC_ITEM, "SCORE: 1 EXPLANATION: A.", _MC_ASKS, 0.0),
        ("simple_bench", _MC_ITEM, "I cannot judge this.", _MC_ASKS, 0.5),
        ("mmlu_pro", _MC_ITEM, "SCORE: 8.2 EXPLANATION: B, thin.", "with N from 1 (worst) to 10 (best)", 0.8),
        ("simpleqa_verified", _QA_ITEM, "GRADE: correct\nEXPLANATION: Paris.", _QA_ASKS, 1.0),
        ("simpleqa_verified", _QA_ITEM, "GRADE: incorrect\nEXPLANATION: Lyon.", _QA_ASKS, 0.0),
        ("simpleqa_verified", _QA_ITEM, "GRADE: correct\nEXPLANATION: Paris.\nScore: 7", _QA_ASKS, 1.0),
        ("simpleqa_verified", _QA_ITEM, "GRADE: [incorrect]\nEXPLANATION: Lyon.", _QA_ASKS, 0.0),
        # a failed judge call is a 0.0 with the error, not a silent 0.5 counted as a success
        ("simple_bench", _MC_ITEM, RuntimeError("429 rate limited"), _MC_ASKS, 0.0),
        # the judge restates the rubric before its own score line: the labelled line wins
        ("simple_bench", _MC_ITEM,
         "Per the rubric: Score 10 for a correct answer, 1 for an incorrect one. It chose A.\nSCORE: 1", _MC_ASKS, 0.0),
        ("simple_bench", _MC_ITEM, "**SCORE:** 10\nEXPLANATION: B.", _MC_ASKS, 1.0),
        ("simple_bench", _MC_ITEM, "Score: **10**", _MC_ASKS, 1.0),
        # "Grade:" in a multiple-choice reply is prose; only a prompt that asks for a GRADE reads one
        ("simple_bench", _MC_ITEM, "Grade: Correct answer is B; the model chose A.\nSCORE: 1", _MC_ASKS, 0.0),
    ],
    ids=["mc-correct", "mc-incorrect", "no-score", "mc-reasoning", "simpleqa-correct", "simpleqa-incorrect",
         "simpleqa-grade-decides", "simpleqa-bracketed-grade", "judge-error", "rubric-echo", "markdown-label",
         "markdown-value", "mc-grade-is-prose"],
)
async def test_a_benchmark_item_is_scored_from_the_judges_reply(
    evaluation_manager, monkeypatch, benchmark, item, reply, asked, expected
):
    """A benchmark's own formatter, through the real analyze(): the model answers, the judge
    scores that answer on the scale it was asked for, and the item's score is the judge's, not
    the 0.5 default (TASK-<JB>)."""
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization

    prompts: list[str] = []

    def chat(request, timeout=None):
        prompts.append(request["messages"][-1]["content"])
        if isinstance(reply, Exception) and len(prompts) > 1:
            raise reply
        return {"choices": [{"message": {"content": "B. 4" if len(prompts) == 1 else str(reply)}}]}

    monkeypatch.setattr(
        summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _p: SimpleNamespace(chat=chat))
    )
    data = get_registry().create_evaluator(benchmark).format_for_custom_metric(item)
    result = await evaluation_manager.evaluate_custom_metric(
        metric_name=data["name"],
        description=data["description"],
        evaluation_prompt=data["evaluation_prompt"],
        input_data=data["input_data"],
        scoring_criteria=data["scoring_criteria"],
        api_name="openai",
        api_key="sk-test",
    )
    assert len(prompts) == 2, prompts  # the model's answer, then the judge
    assert "B. 4" in prompts[1] and asked in prompts[1]
    assert result["score"] == pytest.approx(expected)
    if isinstance(reply, Exception):
        assert result["explanation"].startswith("Error:")
```

  - `evaluation_manager` is the fixture at `tests/Evaluations/conftest.py:554`.
  - The model comes from the tracked `Config_Files/config.txt` (`[openai]`).
  - The test uses no handle, so it needs none of TASK-15426 B's helpers.
- [ ] **Step 2: Run it:** `<pytest> -q --tb=short <focused>`. Expected (measured): `13 failed`, each with `assert 1 == 2` (`len(prompts)`: only the model's answer was asked for). The probe log stays empty.
- [ ] **Step 3: Implement** the four edits, then the three guards. No signature changes.
  - `evaluation_manager.py`, in the judge's `asyncio.to_thread(analyze, ...)` (`:589-597`): end the `system_message=...` line with a comma, and add `input_is_literal_text=True,` as the last keyword.
  - `:630`: `match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)  # benchmarks ask "SCORE:"`.
  - After the `for pattern in score_patterns:` loop (`:629-639`), before `# Extract explanation` (`:641`) (G3 below replaces this block):

```python
                # SimpleQA's grader answers with a grade (simpleqa_eval.parse_grading_response); it decides.
                grade = re.search(r'GRADE:\s*(correct|incorrect|not_attempted)', response, re.IGNORECASE)
                if grade:
                    score = 1.0 if grade.group(1).lower() == "correct" else 0.0
```

  - `benchmark_utils.py:496` becomes `3. Score 10 for a correct answer, 1 for an incorrect one"""`, and `:500` becomes `Respond with: SCORE: N EXPLANATION: [explanation], with N from 1 (worst) to 10 (best)"""`.
  - Run the focused test now, before the guards. Expected (measured): `6 failed, 7 passed`. The six failures are the guards' cases: `[simpleqa-bracketed-grade]`, `[judge-error]`, `[rubric-echo]`, `[markdown-label]`, `[markdown-value]`, `[mc-grade-is-prose]`.
  - **G1.** Right after the judge's `analyze` call, before `# Parse response with strict validation`:

```python
            # A failed judge call (provider error, timeout, rate limit, truncated or empty reply)
            # is a 0.0 with the error, as for the model's own answer above; not a silent 0.5.
            if isinstance(response, str) and response.startswith("Error:"):
                logger.error(f"Judge call failed for metric '{metric_name}': {response}")
                return {"metric_name": metric_name, "score": 0.0, "explanation": response, "raw_output": None}
```

  - **G2.** First entry of `score_patterns` (the old `[Ss]core[:\s]+...` entry stays second):

```python
                    # the labelled "SCORE: 8" (also "**SCORE:** 8") wins over prose such as "Score 10 for a correct answer"
                    r'score[*_\s]*:[*_\s]*(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
```

  - **G3.** The grade block becomes:

```python
                # SimpleQA's grader answers with a grade (simpleqa_eval.parse_grading_response); it decides.
                # Only a prompt that asks for a GRADE reads one ("Grade: ..." elsewhere is prose).
                grade = re.search(r'GRADE[*_\s]*:[*_\s\[]*(correct|incorrect|not_attempted)\b', response, re.IGNORECASE)
                if grade and "GRADE:" in evaluation_prompt:
                    score = 1.0 if grade.group(1).lower() == "correct" else 0.0
```

- [ ] **Step 4: Run it.** Expected (measured): `13 passed`, and an empty probe log.
- [ ] **Step 5: Run the touched test files whole:** `<pytest> -q -n 4 tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py tldw_Server_API/tests/Evaluations/unit/test_scoring_normalization.py tldw_Server_API/tests/Evaluations/unit/test_bullshit_benchmark_registry.py`. Expected (measured): `78 passed`, and an empty log (the 72 of the base plus this test's 6 beyond its first 7).
- [ ] **Step 6: Commit** the three files (message 1).
- [ ] **Step 7: The guide (AC #3).** In `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md`, after the `### Run failures in summary` list (`:160-164`), add (with the real id):

```markdown
### Every item scored 0.5

- Before the November 2026 fix (TASK-<JB>), the judge's call never reached the provider, so every item got the 0.5 default. The `average_score`, `min_score`, `max_score` and `by_category` of those runs, and the results they stored (`save_results`), carry no information.
- Re-run `simple_bench` or `bullshit_benchmark`: their scores are now the judge's. Do not re-run the others expecting meaningful scores. `mmlu_pro` rows are not yet mapped to the benchmark's fields, so its judge is told the correct answer is "None", and its scores look real but carry no information. `simpleqa_verified` has no loader entry and `gpqa_diamond` most likely fails the same way (404), `mmlu_pro` and `gpqa_diamond` also need the `evaluation` extra (the `datasets` package), and six more benchmarks answer 501.
- Each item's score is the judge's reply mapped to 0-1: a 1-10 score (10 is 1.0, 1 is 0.0). An item that scores 0.5 got a reply with no score; the server logs `Could not parse valid score`. An item whose explanation starts with `Error:` is a failed judge call (provider error, timeout, rate limit or an empty or truncated reply); it scores 0.0, and re-running it may succeed.
```

  - Run `bash Helper_Scripts/refresh_docs_published.sh`. Only `Docs/Published/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md` may change (measured).
  - `tests/Docs` runs in Task 2 Step 3. It passed on the prototype.
  - Commit both files (message 2).

## Task 2: Verify, close, review, PR

- [ ] **Step 1: Focused set again.** Expected: `13 passed`, and an empty log.
- [ ] **Step 2: Wider set** with label `final`. Expected (measured on the prototype): `1 failed, 678 passed`.
  - That is the base's 665 plus this plan's 13.
  - The failure is the same probe artifact as the base's.
  - The log has the same 2 lines as the base's.
- [ ] **Step 3: CI checks.** Each expected result was measured on the prototype:

| Command | Expected |
|---|---|
| `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir> NETPROBE_LOG=<scratchpad>/netprobe_ci.log TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -p netprobe_15427 -q -n 4 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI tools/backlog-py/tests` | `1661 passed, 6 skipped` (about 6.5 minutes) and an empty log. This includes the function-size ratchet. It was measured without the plan copy and the task files. If `test_land_required_gates.py::...[gate_pending]` fails with a 30 s subprocess timeout, rerun that file alone (12 passed): it is load, not this change. |
| `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py` | `new_uncovered=0`, "OK — no newly-unshared test files." |
| `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce` | exit 0, "OK — no offenses beyond the baseline." |
| `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py` | "208 unauthenticated routes; baseline exact." |
| `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` | "OK — OpenAPI fingerprint matches the checked-in snapshot." and "OK — committed path union matches the canonical contract." |
| `/opt/homebrew/bin/uvx bandit -ll tldw_Server_API/app/core/Evaluations/evaluation_manager.py tldw_Server_API/app/core/Evaluations/benchmark_utils.py` | One finding, pre-existing: B608, Medium, at `evaluation_manager.py:858` on the base and `:873` with the PR (`# nosec` one line above). Nothing new. |
| `$PY -m ruff check` on the three touched Python files | 13 pre-existing findings in `test_evaluation_manager.py` (I001, F401 x5, F841 x2, F811 x4, B017), the same 13 on the base; none on the new lines; none in the two source files. |
| `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` | exit 0 |
| `git diff origin/dev --check` | nothing |

  Two more checks, which contain a `|`:
  - `grep -c -E "evaluation_manager|benchmark_utils|test_evaluation_manager|Benchmark_Creation" Helper_Scripts/ci/scope_predicate_baseline.txt` must print `0`.
  - `git grep -n -E "TASK-<(JB|NT|BD)>" -- ':!Docs/superpowers/plans'` must print nothing.

- [ ] **Step 4: Close TASK-<JB>** (fill every `<...>` first):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-<JB> \
  --append-notes "RED/GREEN (record the real counts): test_a_benchmark_item_is_scored_from_the_judges_reply 13 failed (assert 1 == 2: one adapter call) -> 6 failed, 7 passed with the four edits -> 13 passed with the three guards. Each edit is pinned by ablation: without input_is_literal_text all 13 fail; without IGNORECASE seven (mc-correct, mc-incorrect, mc-reasoning, rubric-echo, markdown-label, markdown-value, mc-grade-is-prose); without the GRADE branch the four simpleqa cases; with the grade as a fallback only, simpleqa-grade-decides; without the 1-10 clause of Respond with, mc-reasoning; without the 10/1 rubric line eight; without the judge-error branch, judge-error; without the colon-anchored pattern, rubric-echo, markdown-label and markdown-value; without the GRADE: prompt gate, mc-grade-is-prose; without the bracket tolerance, simpleqa-bracketed-grade. Wider set (31 files, under a socket probe) <base> -> <final>, the same two probe lines. tests/Docs + lint + CI + backlog-py <n> passed; shard coverage, test-quality ratchet, route-auth ratchet and OpenAPI unchanged; Bandit <result>." \
  --append-notes "Stored results: nothing in code. Every stored benchmark score from before this fix is a 0.5 placeholder (or 0.0 for a failed item); no route or WebUI view reads internal_evaluations back (EvaluationManager.get_history/get_evaluation/list_evaluations/compare_evaluations have no route caller; the History tab lists the unified evaluations table). The benchmark guide says so. Which benchmarks reach the judge: on a default install only simple_bench and bullshit_benchmark. mmlu_pro and gpqa_diamond need the datasets package; mmlu_pro rows are not field-mapped (its judge is told the correct answer is None, so its scores carry no information); simpleqa_verified loads 0 rows (404); gpqa_diamond most likely 404s; aider_polyglot, mask, simpleqa and swe_bench cannot build an evaluator (languages, check_consistency, max_tokens, use_repo_context) and bfcl and vending_bench have no class (all six 501). The SimpleQA grade edit is correct but reaches no user today. bullshit_benchmark asks 0-10 and is read on 1-10 (Score: 1 gives 0.0; 5 gives 0.444), and the code-generation, honesty, instruction-following and function-calling formatters still ask SCORE: X.X (a perfect 1.0 now parses to 0.0); all in the follow-up. Owner on Decision 1 (Change summary): <answer>. Owner on Decision 4 (benchmarks on correct data): <answer>; follow-up <TASK-BD or declined>." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "The benchmark judge now reaches the provider: evaluate_custom_metric passes its JSON input to analyze as literal text, so a formatter's question/choices/response keys no longer empty it. Its reply is read as each formatter asks: the score pattern ignores case ('SCORE:'), the multiple-choice formatter asks for the manager's 1-10 scale (it asked 1.0/0.0, and 1 is the worst rating on 1-10), and a SimpleQA grade decides (correct 1.0, else 0.0). One parametrized test (13 cases) drives simple_bench, mmlu_pro and simpleqa_verified through their own formatters, the real manager and analyze to a stubbed adapter. Results stored before the fix are 0.5 placeholders; the benchmark guide says so. A failed judge call scores 0.0 with its error, not 0.5, and a judge that restates the rubric or writes markdown labels is read right. Follow-ups filed: TASK-<NT> (tests that reach real hosts) and, if the owner agreed, TASK-<BD> (benchmarks on correctly shaped data)." \
  -s Done
```

  - Write the real ids, then check that `git grep -n -E "<(answer|base|final|n|result|TASK-BD or declined)>" -- 'backlog/tasks/task-<JB> *' 'backlog/tasks/task-<NT> *'` prints nothing.
  - `git add -A backlog/tasks && git diff --cached --check`, then commit (message 3).
- [ ] **Step 5: Review before the PR.**
  - Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev..HEAD`. Give it this plan's Root cause table, Decisions 2-3 and Review Focus.
  - Address every finding, then re-run the focused set.
  - Record each ruling in the ledger.
- [ ] **Step 6: Finish the PR and merge.**
  - `git push`. Replace the draft's body with `gh pr edit <n> --body-file <file>`, then run `gh pr ready <n>`.
  - The body has:
    - a Summary: the six root-cause rows, the four edits and the three guards;
    - the ACs, and how each is met;
    - the stored results (Decision 2) and what presents them;
    - TASK-<NT>, and TASK-<BD> if filed;
    - the Review Focus items;
    - the measured counts;
    - a full `## Change summary` (Decision 1's answer).

    It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - Diff the PR's Frontend E2E failures against a recent dev run. They are not required checks.
  - Check `gh variable get MERGE_QUEUE` (unset when this plan was written). When the reviews are settled and the Change summary is in, run `QUEUE="<n>:fix/benchmark-judge-<JB>" sh <plans dir>/pr_queue3.sh` in the background.
    - The script reads dev's 8 required checks from the ruleset, rebases server-side when the branch is BEHIND, and merges when all pass.
    - Wait for the license status that each body edit re-triggers before merging.
  - Confirm the PR is MERGED before deleting the branch. Then tell the TASK-15426 B executor to rebase (see Interplay).

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `chore(backlog): file TASK-<JB> and TASK-<NT> (and TASK-<BD>), start TASK-<JB> and add its plan`
- 1: `fix(evaluations): the benchmark judge reaches the provider and its reply is scored (TASK-<JB>)`. Body: R1-R6, one line each.
- 2: `docs(evaluations): benchmark results from before TASK-<JB> are 0.5 placeholders`
- 3: `chore(backlog): close TASK-<JB>`

## Interplay with TASK-15426 PR B

B (`plan_15426_b.md`, `proto_15426_b.patch`) is already folded to reference this fix as TASK-<JB>: its V5 says "fixed by TASK-<JB>", its Decision 4 lists what changes, and its `TASK-<F_B>` step is gone (`plan_15426_b.md:27`, `:36-42`, `:293`). The one item left is B's comment at `:161` ("`text` is a key `extract_text_from_input` keeps ..."), which is obsolete after this PR. This PR touches two of B's files.

- **`tldw_Server_API/app/core/Evaluations/evaluation_manager.py`.**
  - B's hunk `@@ -593,7 +599,8 @@` adds `provider_credentials=provider_credentials,` after the judge's `system_message`. This PR adds `input_is_literal_text=True,` after the same line. That is one textual conflict (measured by the review with `git apply --3way`): keep both keywords. Every other B file applied cleanly.
  - B's other hunks (the signature and docstring at `:553-566`, and `_generate_model_response` at `:739-760`) do not overlap this PR's hunks.
- **`tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py`.** B inserts `test_a_custom_metric_runs_both_calls_on_the_handle_it_is_given` before `class TestEvaluationComparison`, and this PR appends at the end of the file. No conflict.
  - B's test still passes on this fix: two calls, each with the handle (the review measured 44 passed across B's three test files with both changes).
- **No other file of B's** is touched here: not the two benchmark routes, `outputs_service.py`, `watchlists.py`, the route tests, or the docs.

## Conflicts with open PRs

None. On 2026-11-07, 41 open PRs targeted `dev`. Their files were checked with `gh api repos/rmusser01/tldw_server/pulls/<n>/files --paginate`, one PR at a time.
- No PR touches `Evaluations/evaluation_manager.py`, `benchmark_utils.py`, `test_evaluation_manager.py` or the benchmark guide.
- Near misses:
  - #3269 touches `core/Prompt_Management/prompt_studio/evaluation_manager.py`, a different file, and a WebUI `RunsTab.benchmark-option` test;
  - #3021 touches `core/Evaluations/README.md` and `Docs/Published/Evaluations/Evals-Plan-1.md`. This PR touches neither.
- #3439 (TASK-15427) adds only its plan and backlog files.
- The PRs can merge in either order.

## Not measured

- **A real provider's judge.** The stub replies in each formatter's format. It was not measured whether a live model obeys R3's scale or still writes `SCORE: 1.0` (Review Focus 1).
- **PostgreSQL cases.** None of the 31 wider files runs one here (PG down; `TLDW_TEST_NO_DOCKER=1`). The `[postgresql]` stream cases of `test_restricted_postgres_media_retrieval.py` (TASK-15482) were not run.
- **The blocker's full-suite count.** A full run was not made: 5379 test files, about 45,000 test functions, is too slow on this machine. TASK-<NT>'s description gives the dry count and its method.
- **A live judge on the new scale and the new patterns.** The stub replies in the formats the guards parse; whether a live model writes `SCORE:` lines at all was not measured.
- **The ids.** `task_next.sh` was not run; the ids are allocated at execution.
- **`gpqa_diamond`'s 404.** By code only: it needs the network and the `datasets` package.

## Review fold (2026-11-07)

Review: `review_plan_judge.md` (ready with fixes). Measured on `origin/dev` `7a020da14a` with the patch applied, every run under the socket probe (empty log except the two known `test_security.py` lines). The test grew from 7 to 13 cases. RED first: 13 failed on the base, and 6 failed, 7 passed with the four edits only. Then GREEN: 13 passed. The three touched test files give 78 passed; the wider set gives `1 failed, 678 passed`; `tests/Docs` + `lint` + `CI` + backlog-py give `1661 passed, 6 skipped`. The prototype was regenerated.

| Finding | Where it landed |
|---|---|
| I-1 failed judge call scores a silent 0.5 | G1 in Architecture, Root cause R5, Task 1 Step 3 (G1 code), `[judge-error]` case, Review Focus 2, guide entry (the `Error:` line, and the wrong "0.5 means neither" line rewritten), TASK-<JB> AC 2 |
| I-2 first-match parser, rubric echo, markdown labels | G2 in Architecture, Root cause R6, Task 1 Step 3 (G2 code), `[rubric-echo]`, `[markdown-label]`, `[markdown-value]` cases, Review Focus 2 |
| I-3 which benchmarks reach the judge; follow-up | Not changed ("Which benchmarks reach the judge"), Root cause R4, Decision 4 (rewritten, with the owner decision and `TASK-<BD>`), Task 0 Step 3b (files `TASK-<BD>` only if the owner agrees), guide entry (names `simple_bench` and `bullshit_benchmark`, warns off `mmlu_pro`), TASK-<JB> description and close-out notes, Ids, final grep |
| M-1 GRADE gate and spellings | G3 in Architecture, Root cause R4, Task 1 Step 3 (G3 code), `[simpleqa-bracketed-grade]`, `[mc-grade-is-prose]`, Review Focus 3 |
| M-2 Decision 3 wording | Decision 3 rewritten: R1-R3 ship together; R4 stays, gated |
| M-3 bullshit_benchmark 0-10 | Not changed (measured: `Score: 1` is 0.0, 5 is 0.444; the one-line fix is left to `TASK-<BD>`) |
| M-4 fractional formatters | Not changed ("The other formatters' fractional rubrics"); `TASK-<BD>` scope |
| M-5 ruff Expected line | Task 2 Step 3: 13 pre-existing findings, none new |
| M-6 `<answer>` placeholders in Task 0 Step 4 | Task 0 Step 4 drops the "Owner answers" clause; Task 2 Step 4 records them |
| M-7 stale Interplay | Interplay rewritten: B is already folded; only its `:161` comment is obsolete |
| M-8 TASK-<NT> blocker ACs | TASK-<NT> AC 2: `is_loopback`, the configured `TEST_DB_HOST`, and the fixture-scope ceiling |

Not folded: nothing. M-3's one-line prompt fix is deferred to `TASK-<BD>` on purpose (the review marked it optional). The review's optional note (the judge receives the input twice) is not taken.
