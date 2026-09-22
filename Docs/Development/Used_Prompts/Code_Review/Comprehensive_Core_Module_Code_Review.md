# Comprehensive Code Review: tldw_server Core Modules

> **How to use this prompt.** Paste the whole file into a fresh Claude Opus session opened at
> the repo root (`/Users/macbook-dev/Documents/GitHub/tldw_server`). It is self-contained: every
> path, command, count, and rule it depends on is stated inline. Do not summarize it before
> pasting — the measured figures and the do-not-report lists are what keep the review from
> returning noise.

## Your Role

You are a **Principal Software Engineer** running a maintainability and correctness audit on a
large Python FastAPI monolith. You have 15+ years on long-lived server codebases and you have
been paged at 3am by exactly the kind of drift this audit is looking for: the same helper
copy-pasted into six modules, five of which got a bug fix and one of which did not.

You are **read-only for this entire task.**

- Do not edit, create, or delete any source file.
- Do not run `git commit`, `git add`, `git checkout -b`, or any mutating git command.
- Do not apply fixes, even obvious one-liners. A fix you make is a fix nobody reviewed.
- You *may* freely run read commands: `rg`, `grep`, `find`, `wc`, `sed -n`, `git log`,
  `git diff`, `cat`, and read-only `pytest --collect-only`.
- You **may** write your review output files, which are documentation, not code. Those paths are
  specified in section 12. Nothing outside those paths.

Your deliverable is a ranked findings report plus a concrete consolidation migration plan. Not
a refactor. Not a style guide. Not an architecture essay.

## Context

**tldw_server** ("Too Long; Didn't Watch") is a self-hosted research-assistant and
media-analysis platform: media ingestion and transcription, RAG, an OpenAI-compatible chat and
audio API, evaluations, and an MCP server. FastAPI backend, Next.js WebUI, SQLite by default
with PostgreSQL supported.

The review was commissioned to find four things:

1. Latent bugs and correctness issues.
2. Efficiency gains.
3. Poor patterns.
4. **Primary:** functionality reproduced across modules and functions that should instead live
   in a shared utility module.

### Measured baseline — treat as established fact, do not re-derive

These were measured immediately before this prompt was written. Spending your context
recounting them is waste.

| Fact | Value |
|---|---|
| `tldw_Server_API/app` total | ~1,686,000 LOC |
| `app/core` total | ~1,274,000 LOC |
| `app/core/Utils/` total | **4,893 LOC** — 0.4% of core |
| Files over 2,000 LOC | **138** |
| Largest single file | `app/core/DB_Management/ChaChaNotes_DB.py` — **45,292 LOC** |
| `core/` → `app/api/` imports (layering inversion) | **161**, across **107** files |
| Ad-hoc `*_utils.py` / `*_helpers.py` modules outside `Utils/` | ~60 |
| Test files / test LOC | **4,730** / ~1,738,000 |
| `# noqa` in `app/` | **4,813** (TRY003 2,580; BLE001 1,332) |
| `# type: ignore` in `app/` | **1,614** |
| Ruff CI baseline / MyPy CI baseline | **1,419 / 10,191 errors**, both non-blocking |
| Files grandfathered out of `BLE001` | **1,050** |

Raw grep signal counts across `app/`, for orientation only: `def .*retry` 135,
`def .*paginat` 179, `def .*hash` 233, `def .*truncate` 67, `def sanitize` 62,
`def .*_to_dict` 91, `tempfile.mkdtemp` 22.

**These are signals, not findings.** A verification sweep already confirmed some and disproved
others; the verdicts are in sections 6 and 7. A grep tally is never admissible as a finding.

**On the line numbers in this prompt.** The seed clusters in section 6 were spot-verified by
opening the cited lines, and several citations had drifted by a few lines between the sweep and
verification — one cluster turned out ~3x larger than first scoped, and one claimed bug turned
out to be the *correct* implementation of a diverging pair. So: **re-confirm every line number
before you cite it in your own report.** Treat this prompt's citations as reliable pointers to
real code, not as line-exact addresses, and never copy one into a finding without opening it.

### Four things that invert the naive approach

Read these before you plan anything. Each one is a trap that a generic "find the duplication"
review falls into.

**1. A canonical utility layer already exists and is largely unused.** This is *not* a codebase
that lacks shared helpers. It is a codebase that has them and bypasses them.

- `app/core/Utils/Utils.py` defines `truncate_content`, `save_temp_file`, `generate_unique_id`,
  and `is_valid_url` — each with **zero importers anywhere in the repo**. Meanwhile ~15 private
  `_truncate_text` helpers exist in other modules.
- `app/api/v1/utils/datetime_utils.py:21` exists; ~60 private `_utc_now` definitions bypass it.
- `app/api/v1/utils/http_errors.py:145` exists; 4 copies of `get_http_status_from_exception`
  bypass it.
- `app/core/LLM_Calls/tokenizer_resolver.py:929` exists; the tiktoken fallback is rewritten 8x.
- `app/core/DB_Management/sqlite_policy.py:47` exists; 3 hand-rolled pragma blocks bypass it.

So the headline finding class is **adoption gaps and dead helpers**, not greenfield
consolidation. Frame it that way.

**2. The utility modules that did get adopted became junk drawers.** `app/core/http_client.py`
is **6,600 LOC** in a single module (132 importers). `Utils/Utils.py` is 1,110 LOC of unrelated
concerns — filesystem, downloads, URL parsing, transcript formatting, zip validation.

Therefore: **you are forbidden from recommending "move this into `Utils.py`"** or any variant
that grows an existing grab-bag. Every consolidation proposal must name a *cohesive*
destination module with a single responsibility, and must say what that module's one job is.
"Add a `_coerce_bool` to `Utils.py`" is a rejected recommendation. "Create
`app/core/Utils/coercion.py` owning scalar/env coercion, with these 4 functions" is accepted.

**3. Test coverage is strong, but tests are not findable by path — and the test tree has its own
duplication problem.** The suite is 4,730 files / ~1,738,000 LOC, more test code than
application code. Every in-scope module is well covered, measured by import:

| Module | test files importing it | Module | test files importing it |
|---|---|---|---|
| DB_Management | 1,238 | LLM_Calls | 188 |
| AuthNZ | 1,207 | TTS | 165 |
| Chat | 285 | Evaluations | 106 |
| RAG | 218 | Sync | 99 |
| Ingestion_Media_Processing | 191 | MCP_unified | 63 |

`tldw_Server_API/tests/` directory names **do not mirror** `app/core/` module names. Ingestion
tests live across `tests/MediaIngestion_NEW/`, `tests/Media_Ingestion_Modification/`,
`tests/Media/`, and `tests/Audio/`. So:

- **Locate a module's tests by import-grep, never by path:**
  `grep -rl "core\.<Module>" tldw_Server_API/tests`
- **Do not report any module as untested.** If a path-based search returns nothing, your search
  was wrong. Re-run it by import.

Separately, the parallel `*_NEW` test trees — `tests/Chat` *and* `tests/Chat_NEW`,
`tests/Character_Chat` *and* `tests/Character_Chat_NEW`, plus `tests/RAG_NEW`, `tests/TTS_NEW`,
`tests/MediaIngestion_NEW` — and the split AuthNZ trees (`AuthNZ`, `AuthNZ_Unit`,
`AuthNZ_SQLite`, `AuthNZ_Postgres`, `AuthNZ_Federation`) are themselves an instance of the
pattern under review. **The test suite is a review target, not only a safety net.**

**4. This review has been run twice before and the repo has a house format for it.** Do not
invent a format and do not re-derive existing findings.

- `Docs/superpowers/reviews/` holds per-module staged review ledgers.
- **`Docs/superpowers/reviews/rag/` and `Docs/superpowers/reviews/db-management/` are already
  complete** (stage1–stage6 plus synthesis). For those two modules your job is to **extend**:
  read the existing ledger, then report only what it does not already contain, and say
  explicitly which prior findings you are confirming as still-live versus already-addressed.
- Other existing ledgers to check before reviewing their subject areas:
  `auth-dependencies/`, `evals-module/`, `api-pagination/`, `api-response-envelope/`,
  `characters-backend/`, `moderation-backend/`, `web-scraping/`, `shared-api-client/`,
  `phase3-pilots/`, `phase4-parking-lot/`.

## Evaluation Criteria

### 1. Scope

In scope: the ten largest `app/core` modules plus the endpoints layer.

| Module | LOC | Module | LOC |
|---|---|---|---|
| `app/core/DB_Management/` | 233,293 | `app/core/TTS/` | 42,067 |
| `app/core/MCP_unified/` | 141,588 | `app/core/Evaluations/` | 35,805 |
| `app/core/AuthNZ/` | 74,539 | `app/core/Chat/` | 25,698 |
| `app/core/RAG/` | 58,804 | `app/core/LLM_Calls/` | 21,841 |
| `app/core/Ingestion_Media_Processing/` | 55,842 | `app/api/v1/endpoints/` | 247,913 |
| `app/core/Sync/` | 43,905 | | |

Out of scope: the frontends (`apps/`), `Helper_Scripts/`, `Dockerfiles/`, and any `app/core`
module not listed. You may *cite* an out-of-scope file as evidence for an in-scope finding —
for instance a duplicate living in `app/services/` — but do not audit it as a target.

### 2. Read this prior art before you review anything

Read these **selectively, by relevance to your scope** — not all seven every time. If you are
reviewing one module with no existing ledger, `Codeslop-Vibecheck-SKILL.md` and any ADR touching
your module are usually the only ones that change a finding. The ledger and audit-format docs
earn their cost when your module already has a ledger, or when you are producing the ranked
cross-module table.

| Path | Why |
|---|---|
| `Codeslop-Vibecheck-SKILL.md` (repo root) | **The rubric you apply.** Its three axes, per-axis flag / do-not-flag lists, severity guide, and "Common Mistakes" section. Follow it, with the two overrides in section 3. |
| `Docs/superpowers/reviews/rag/README.md` | The review rules and stage-file format you must reproduce. |
| `Docs/superpowers/reviews/rag/` + `db-management/` | Already-complete reviews. Extend, do not redo. |
| `audits/2026-07-04-test-suite-audit-round2.md` | The finding-ID and severity-ranking style to match. |
| `Docs/Architecture.md` | Layering ground truth. Quoted in section 9. |
| `Docs/ADR/` (31 ADRs) | Binding decisions. **A finding that contradicts an ADR is invalid** — check before asserting. |
| `Docs/Conventions/README_TEMPLATE.md` | Module README convention; 90 of 100 `app/core/*/` dirs comply. |
| **The Backlog ledger — `backlog search "<symbol>" --plain`** | **Mandatory, not optional.** ~3,110 tasks, many of them closed reviews of the exact code you are auditing. See the warning below. |

**Never cite your own output as prior art.** Before describing any ledger, audit, or stage file
as pre-existing corroboration, establish that it predates this run:

```
git log --oneline -1 -- <path>     # empty = never committed
git status --porcelain -- <path>   # '??' = created this session, possibly by you
```

An untracked file dated today is **not** prior art. This matters because the staged-ledger format
in section 12 has you writing stage1, then stage2, then summarizing — so a reviewer who files
stage-1 findings and later re-reads the directory will mistake them for an earlier reviewer's
work. Observed failure: in a 10-module validation run, two reviewers labelled their own stage-1
and stage-2 findings from twenty minutes earlier as "prior findings, confirmed still-live,"
manufacturing independent corroboration for themselves. A third ran `git status`, saw `??`, and
correctly reported the files as uncommitted output of the run in progress. Be the third one.

For any finding you mark as confirming prior work, name **which commit or tracked file** the
prior finding came from. If you cannot name one, it is a new finding — label it new.

**Search the Backlog before you write any finding.** Run
`backlog search "<the symbol or file you are about to report>" --plain` and read any hit that
looks related, including `Done` ones. This is not bookkeeping — a closed task may record that a
behaviour you are about to flag is **deliberate**, or, worse, that some odd-looking construct is
**load-bearing**.

> **Worked example, from this prompt's own validation run.** A reviewer found
> `extract_text_from_input` defined twice in
> `core/LLM_Calls/Summarization_General_Lib.py` (`:493` and `:917`), the second winning at
> runtime behind a `# noqa: F811`, and recommended deleting the duplicate. Correct-looking, and
> wrong. `backlog task 2425` records that an **arbitrary file read** in that module was assessed
> as "not active through `analyze()` **because** the file-reading helper is shadowed by a later
> `extract_text_from_input()` definition." The shadow is an accidental security control.
> Deleting it re-opens a file read reachable from 17+ core modules. The finding was real; the
> recommended fix would have introduced a vulnerability.

**The rule this gives you:** when a construct looks obviously wrong and obviously easy to
delete, that is exactly when to check whether something depends on it. Report the finding —
but search first, and say in the finding what the prior art says.

### 3. The rubric, and the two rules you must override

Apply `Codeslop-Vibecheck-SKILL.md` as written — its axes, its flag and do-not-flag lists, its
High/Medium/Low severity guide, and especially its Common Mistakes section, which already
forbids the two ways this review most easily fails:

> Do not report "duplication" just because two blocks look similar; identify the shared
> knowledge that will diverge.

> Do not produce architecture essays. Tie every finding to code and a practical next step.

**Override exactly two of its rules:**

1. Its scope rule says *"Review only the provided diff, changed files, or explicitly requested
   code"* and *"Treat surrounding code as context, not as a mandate to audit the whole
   repository."* **This review is a whole-module audit** — that is the explicit request.
   Override the scope rule. Keep everything else.
2. It says *"Do not flag formatting, naming, performance, security, or test coverage unless
   they directly intersect one of the three axes."* **Performance is lifted** into a
   first-class axis (section 4). Security findings are in scope where they arise from
   divergence (see section 7's sanitization note). **Formatting and naming stay excluded** —
   they are also settled policy, per section 5.

The rubric's three axes do not cover correctness or efficiency, so section 4 extends it from
three axes to five rather than replacing it.

### 4. The five axes, each with an explicit drop rule

An axis that cannot reject a finding produces slop. Each axis below states what earns a finding
**and what gets one dropped.** Choose one primary axis per finding even when it overlaps
several, per the rubric.

**Axis 1 — Duplication / utility consolidation (primary axis, weight it highest).**
For every candidate, first check whether a canonical helper already exists (section 6 lists the
known ones; search before asserting none exists). Then classify:

- `adoption-gap` — a designated shared helper exists and this code bypasses it. **State the
  canonical path.** Most common and most actionable class here.
- `divergent-copies` — N copies exist with **no designated owner**, and they have already
  drifted; one of them is correct (or least wrong). The fix is to **promote one and delete the
  rest**, not to write something new. State which copy is correct and why. This is the class
  that generates correctness findings — both anchor bugs in section 8 are instances of it, so
  expect it to be common.
- `true-duplication` — no canonical home exists and no copy is clearly the right one. **Name
  the cohesive destination module you propose**, and its single responsibility.
- `justified-divergence` — looks duplicated, genuinely is not. Say why, in one line, and drop
  it from the findings list.

*Drop rule:* a grep count is not a finding. Name the **shared knowledge that will diverge** and
the change-amplification cost. If you cannot say what future edit breaks because of this
duplication, drop it. Also drop anything where abstraction would create a worse dependency —
the rubric's own do-not-flag rule.

**Axis 2 — Encapsulation.** Codeslop's flag / do-not-flag lists as written. Callers reaching
into internals, wide APIs, feature envy, logic in the wrong layer, hidden dependencies.
*Drop rule:* explicit dependency injection, thin translation adapters, and public DTOs at a
boundary that intentionally uses them are **not** findings — that is the rubric's own
do-not-flag list. Also drop any encapsulation complaint you cannot tie to a caller that
actually reaches in; "this class is too public" without a caller is taste, not a finding.

**Axis 3 — Sequential coupling.** Codeslop's lists as written. Undocumented
`initialize → configure → call → cleanup` orders, objects valid only after hidden setup,
skippable cleanup/commit/close, state machines as scattered conditionals.
*Drop rule:* not every multi-step workflow is sequential coupling — the rubric is explicit on
this. Flag it only where the API **relies on caller memory instead of enforcing safe usage**.
If an invalid ordering is already impossible to express, or the sequence is enforced by a
context manager, drop it.

**Axis 4 — Correctness / latent bugs (extension).**
*Drop rule:* you must give a **concrete failure scenario — specific inputs or state → wrong
output or crash.** No scenario, no finding. The two anchor bugs in section 8 are the
calibration bar: that is the standard of concreteness expected.

**Axis 5 — Efficiency (extension; the reason the rubric's performance exclusion is lifted).**
Name the **actual cost driver** with `file:line`: N+1 query, sync I/O on an async path,
unbounded fetch or missing `LIMIT`, redundant re-tokenization, per-request model or client
re-instantiation, a full-table scan behind a paginated endpoint.
*Drop rule:* micro-optimizations, and anything where you cannot name the cost driver and say
roughly what it scales with.

**Poor patterns** do not get their own axis — they fold into the above. The list below bounds
what may be reported **on taste grounds alone**; it is **not** a cap on findings. Anything for
which you can meet an axis drop rule — a concrete failure scenario, or a named cost driver — is
reportable whether or not it appears here. Bounded taste list: god modules (138 files >2k LOC), the core→api
layering inversion (section 9), swallowed exceptions **beyond** the grandfathered list, raw SQL
outside `DB_Management/`, mutable default arguments, and missing module READMEs (10 of 100
`app/core/*/` dirs lack one).

### 5. Noise floor — what you must NOT report

Without this section a reviewer returns the project's known, deliberately-accepted lint
backlog as if it were discovery. All of the following are **settled and not findings**:

- **Ruff baseline 1,419 errors; MyPy baseline 10,191 errors.** Both run `continue-on-error: true`
  in `.github/workflows/ci.yml:230-240`. Known. The *blocking* mypy is changed-files-only
  (`.github/workflows/backend-required.yml:183`).
- **Coverage `--cov-fail-under=12`** while real exercised coverage is ~4.8%. The workflow
  comment itself says the gate mostly detects "the app stopped importing." Do not present the
  gap as a discovery; do not read 12% as a quality target.
- **The global ruff `ignore` list** — E501, E402, E741, B008, C901, N802/803/805/806/814/815/816/818/999,
  SIM102/105/108/112/115/116/117, TRY002/003/004/300/301/401. Flagging any of these is arguing
  against a decision already made and documented in `pyproject.toml`.
- **The 1,050-file `BLE001` grandfather list** in `[tool.ruff.lint.per-file-ignores]`, which
  starts at `pyproject.toml:756`. It is an existing ledger with a generation date and a ratchet
  keeping new files gated. Blind-except is reportable **only** in files not on that list, or
  where you can show a specific swallowed error causes a concrete failure. **Grep that block for
  the specific file** before reporting — the aggregate count cannot tell you whether *your* file
  is on it. Note the inverse is also useful: if a rule is **absent** from both the global
  `ignore` list and the per-file block, an inline `# noqa` for it is a per-line suppression, not
  sanctioned policy, and is worth looking at.
- **`ruff` and `mypy` both exclude `tests/` by config.** Tests are unlinted and untyped by
  design. Not a finding.
- **`Docs/Development/REFACTORING-PLAN-DDD.md`** proposes per-domain packages
  (`media/{endpoints,schemas,service,repository}.py`). It is **aspirational and unexecuted**.
  `Docs/Architecture.md` is the operative rule. Do not review against the DDD plan.
- **`ponytail:` comments** are not a convention in this repo (2 hits total). Do not look for a
  debt ledger there.
- **The ruff `target-version = "py39"` vs mypy `python_version = "3.11"` vs CI Python 3.12
  mismatch.** Known. Do not report it at all in a per-module review.

### 6. Verified seed clusters — confirmed starting points

A verification sweep already separated true redundancy from naming coincidence. These ten are
**confirmed**. Use them as starting scent, not as a ceiling: finding more is the job, and each
still needs its change-amplification cost articulated per the Axis 1 drop rule.

**If you are scoped to one module, first skim the Cluster column and skip the rows with no sites
in your module** — do not spend a verification pass discovering a cluster does not apply to you.
Rough module map: 1 → Workspaces/Notes_Graph/DB_Management/AuthNZ/endpoints; 2 → Embeddings,
Ingestion(OCR), LLM_Calls, Setup, MCP_unified; 3 → DB_Management, Sandbox, Setup, endpoints,
services; 4 and 5 → DB_Management (`PromptStudioDatabase.py`), RAG, Workflows, Embeddings,
http_client; 6 → RAG, Workflows, Research, Persona, LLM_Calls, endpoints; 7 → Ingestion(OCR),
Jobs, Audio_Studio, AuthNZ; 8 → LLM_Calls, Local_LLM, Chat, Embeddings, TTS, endpoints;
9 → endpoints only; 10 → MCP_unified only.

| # | Cluster | Confirmed evidence |
|---|---|---|
| 1 | **Base64 cursor/token padding idiom — verified broader than first scoped** | The idiom `"=" * (-len(x) % 4)` + `urlsafe_b64decode` is independently written at **22 sites**, spanning pagination cursors, API-key crypto, character import, sync, and signed tokens: `core/Workspaces/membership_models.py:198`; `core/Workspaces/file_inventory_models.py:165`; `core/Notes_Graph/suggestion_api.py:85`; `core/MCP_unified/modules/implementations/prompts_catalog.py:115`; `core/DB_Management/chacha/shared_workspace_chat_store.py:1238`; `core/DB_Management/media_db/runtime/email_search_cursor.py:35`; `core/Chat/chat_loop_approval.py:21`; `core/AuthNZ/api_key_crypto.py:119`; `core/Character_Chat/modules/character_io.py:256`; `core/Sync/v2/service.py:559`; `core/Local_LLM/llamacpp_snapshot_operations.py:122`; `core/Visual_Identities/source_context.py:105`; `api/v1/endpoints/chat.py:6479`; `api/v1/endpoints/notes.py:878,881`; `api/v1/endpoints/workflows.py:2597`; `api/v1/endpoints/mcp_unified_endpoint.py:138`; `api/v1/endpoints/character_messages.py:448`; `api/v1/endpoints/audio/audio_history.py:82`; `api/v1/endpoints/audio/audio_jobs.py:147`; `services/admin_data_ops_service.py:459`. The full encode/decode pair (with `json.dumps(..., separators=(",",":"), sort_keys=True)`, version check and max-bytes guard) is near-identical in the first six. Plus 3 **mutually incompatible** cursor schemes: `core/Slides/standalone_html_reconciler.py:135` (raw JSON, no base64), `core/DB_Management/Moderation_Review_DB.py:69` (bare int offset), `core/Audit/unified_audit_service.py:1966` (tuple keyset). **No shared helper exists.** Note the sites split into two trust classes — opaque pagination cursors vs **signed/crypto tokens** (`api_key_crypto.py`, `notes.py` signature segments) — so the destination must not flatten a security boundary; say so in the proposal. |
| 2 | **Scalar / env coercion — worst by volume** | ~270 re-implementations of 5-line functions: 89 int-coercers, 91 bool-coercers, 37 float, and 53 files with a private `_env_bool`/`_env_int`/`_env_flag`. Byte-identical: `core/Embeddings/jobs_adapter.py:17` ≡ `core/Embeddings/redis_pipeline.py:30`. Five sibling OCR backends each roll their own with **three different truthy sets**: `core/Ingestion_Media_Processing/OCR/backends/{chatllm_ocr.py:51, deepseek_ocr.py:42, nemotron_parse.py:62, dolphin_ocr.py:480, llamacpp_ocr.py:60}`. Silent semantic drift: `core/LLM_Calls/cache_intents.py:88` accepts `"enabled"`, `core/Setup/readiness_service.py:69` does not. Shared-but-ignored: `core/MCP_unified/environment.py:17` (`is_truthy`). |
| 3 | **`now()` / ISO datetime parsing** | ~60 private `_utc_now`/`_now_iso`, ~40 private `_parse_iso_datetime`, while `api/v1/utils/datetime_utils.py:21` exists and is bypassed. Verbatim `return datetime.now(timezone.utc).isoformat()` at `core/DB_Management/Orchestration_DB.py:179`, `core/Sandbox/store.py:53`, `core/Setup/readiness_store.py:29`, `api/v1/endpoints/media/reading_progress.py:27`, `core/DB_Management/ResearchSessionsDB.py:17`, `core/DB_Management/Moderation_Review_DB.py:28`. Parse variants diverge on tz-normalization (naive vs aware), NOT on `Z` handling — **an earlier draft of this prompt claimed `api/v1/endpoints/files.py:83` "fails on Z-suffixed inputs"; that was wrong and was corrected during validation.** `pyproject.toml:15` sets `requires-python = ">=3.11"`, and `datetime.fromisoformat` has accepted the `Z` suffix since 3.11 (verified on the repo's 3.12 interpreter). So `files.py:83` is the *simplest correct* copy — promote its shape, do not "fix" it — and the **119** `.replace("Z", "+00:00")` sites across `app/` are dead pre-3.11 compatibility code. Treat this row as a caution: verify a claimed failure mode against the project's actual Python floor before filing it. |
| 4 | **Inline exponential backoff** | **28 copies of the same 8-line loop in one file** (exact count verified) — `core/DB_Management/PromptStudioDatabase.py` at `:4165, :4220, :4278, :4313, :4401, :4490, :4528, :4582, :4605, :4751, :4830, :4918, :5032, :5066, :5159, :5219, :5507, :5647, :6027, :6159, :6263, :6267, :6329, :6452, :6675, :6772, :6886, :6946`, all `delay = base_delay * (2 ** attempt) * (0.5 + random.random())`. Plus 8 independent module-level implementations: `core/Embeddings/Embeddings_Server/Embeddings_Create.py:1160`, `core/RAG/rag_service/resilience.py:577`, `core/DB_Management/transaction_utils.py:71`, `core/DB_Management/Workflows_DB.py:1659`, `core/http_client.py:2341`, `core/Web_Scraping/extraction/pipeline.py:267` + `strategies/llm.py:193`, `core/Workflows/engine.py:845` + `:1361`. And a verbatim-duplicated retryable-exception `__init__` at `core/exceptions.py:1836,1846,1886`, `core/Embeddings/services/jobs_worker.py:88`, `core/Chatbooks/services/jobs_worker.py:73`. |
| 5 | **`PromptStudioDatabase.py` dual-backend triplication** | 7,426-line file. `_BackendPromptStudioDatabase` (`:722`–`:3848`) and `_SQLitePromptStudioDatabase` (`:3848`–`:7144`) implement **~40 identically-named methods twice** with parallel SQL; `PromptStudioDatabase` (`:7144`) is a third `*args/**kwargs` delegating facade. Triplicated names include `get_signature` (`:1624`, `:4501`, `:7215`), `retry_job_record` (`:3055`, `:6281`, `:7291`), `list_jobs`, `update_job_status`, `renew_job_lease`, `search_test_cases`. ~3,000 lines of near-duplicate business logic — the largest single-file redundancy in the repo. |
| 6 | **Truncation + token budget + tiktoken** | Verbatim `_truncate_content_by_tokens` at `core/RAG/rag_service/web_fallback.py:50` ≡ `core/Workflows/adapters/rag/search.py:162`. Verbatim `_truncate_text` at `core/Research/providers/local.py:34` ≡ `core/Research/providers/web.py:24`. `_truncate_to_budget` 3x with one **silently missing `.rstrip()`**: `core/Persona/exemplar_prompt_assembly.py:58`, `api/v1/endpoints/character_chat_sessions.py:5651`, `core/VN_Assets/prompts.py:180`. The tiktoken `encoding_for_model → cl100k_base` fallback is rewritten **8x** despite `core/LLM_Calls/tokenizer_resolver.py:929`. |
| 7 | **LLM-output JSON + DB JSON-blob coercion** | `core/Ingestion_Media_Processing/OCR/backends/hunyuan_ocr.py:353` ≡ `dolphin_ocr.py:496`, byte-identical apart from comments; `nemotron_parse.py:314` same name, weaker behavior. ~13 further "strip fences / find first `{`…`}` / `json.loads`" sites. DB-blob side, byte-identical pairs: `core/Jobs/operations/sqlite/lifecycle.py:502` ≡ `core/Jobs/operations/postgres/lifecycle.py:518`; `core/Audio_Studio/render.py:387` ≡ `core/Audio_Studio/migration.py:374` (which also duplicate `_json_dumps` and `_sha256_json`). Five sibling AuthNZ repos with five private copies: `core/AuthNZ/repos/{shared_workspace_repo.py:25, mcp_hub_repo.py:152,166, prototype_workspaces_repo.py:75,89, managed_secret_refs_repo.py:57, data_subject_requests_repo.py:45}`. |
| 8 | **Error → HTTP status mapping** (note: the exception classes and their default status codes live in `core/exceptions.py:796-879` — trace there to state what a misclassification actually returns) | `get_http_status_from_exception` in 4 places, 2 byte-identical: `core/LLM_Calls/error_utils.py:126`, `core/Local_LLM/http_utils.py:54`, `core/Chat/chat_orchestrator.py:247`, `core/Embeddings/Embeddings_Server/Embeddings_Create.py:174`. `_is_http_status_error` 4x: `core/LLM_Calls/error_utils.py:407`, `core/TTS/adapters/{openai_adapter.py:52, elevenlabs_adapter.py:54, qwen3_runtime_remote.py:130}`. Byte-identical `_membership_service_error_to_http` at `api/v1/endpoints/workspace_memberships.py:42` ≡ `api/v1/endpoints/workspaces.py:789`. Shared-but-bypassed: `api/v1/utils/http_errors.py:145`, `core/exceptions.py:1104`. |
| 9 | **Discord / Slack whole-module clones** | `api/v1/endpoints/discord_support.py` (677 lines) vs `slack_support.py` (678) are **~72% identical** after `s/discord/slack/; s/guild/team/`. Byte-identical `_error_response` and `_metric_labels` at the *same line numbers* in both (`:241`, `:248`); also `discord_support.py:534` ≡ `slack_support.py:530`. Same story for `discord.py` (504) vs `slack.py` (593), e.g. `discord.py:128` ≡ `slack.py:133`. Mechanical copy-paste-rename of whole modules. |
| 10 | **MCP `sanitize_input` overrides** | Base at `core/MCP_unified/modules/base.py:766`; four subclasses re-implement the identical depth-guard + strip-control-chars + recurse body with a different char predicate each: `modules/implementations/web_tool_base.py:51` (regex), `run_command_module.py:309` (loop), `sandbox_module.py:259` (genexp, **drops `\t`**), `filesystem_module.py:1465` plus a fifth variant at `:1480`. Each hardcodes `_depth > 20` separately except `web_tool_base`. The differing predicate is the whole point and should be one parameter. |

**Secondary clusters** — real, weaker evidence, worth confirming: SSE frame formatting (4
independent formatters, 2 byte-identical at `core/Meetings/stream_adapter.py:51` ≡
`api/v1/endpoints/mcp_hub_management.py:445`, while `core/Chat/streaming_utils.py` still has
~25 inline `f"data: {json.dumps(...)}\n\n"` despite `core/LLM_Calls/sse.py:51`); streaming file
SHA-256 hand-rolled 8x with **4 different chunk sizes**; SQLite pragma/WAL setup bypassing
`core/DB_Management/sqlite_policy.py:47` (and a `busy_timeout=0 → wal_checkpoint(TRUNCATE) →
restore` block copy-pasted 3x with **different restore defaults**, 5000 vs 10_000);
bounded-gather duplicated **twice in one file** at `core/RAG/rag_service/batch_utils.py:123`
and `:219`, differing only `func(item)` vs `func(index, item)`; `offset = (page - 1) * limit`
inline at 12 sites in `api/v1/endpoints/watchlists.py` alone; 56 private `_emit_*_counter`
shims over `log_counter`.

**A known live consequence of the dual-backend duplication pattern.** A separate audit dated
2026-09-21 found that cross-user isolation leaks in this codebase fail to converge for two
reasons: the **SQLite/PostgreSQL split is only tested on SQLite**, and isolation mechanisms
default to off. Cluster 5 (`PromptStudioDatabase` dual-backend) and cluster 7
(`Jobs/operations/sqlite/` ≡ `postgres/`) are instances of exactly that split. When you assess
those, treat "the two backends can silently diverge and only one is covered" as a **known-real
risk with precedent**, not a hypothetical — and check whether each duplicated pair has tests on
both backends or only one.

### 7. Do-not-seed list — four hypotheses already checked and disproved

These looked like duplication clusters and are not. Do not spend budget rediscovering them, and
do not report them as duplication.

- **httpx / aiohttp client construction.** Only ~14 raw non-test sites; most code already routes
  through `core/http_client.py`. The real problem is the **opposite of duplication**: three
  competing abstraction layers over one client — `core/LLM_Calls/http_helpers.py:55`
  (`_RetrySession`), `core/LLM_Calls/chat_calls.py:62` (`_SessionShim`), and
  `http_client.fetch`. Report this as **over-layering**, and note that `core/http_client.py` at
  6,600 LOC is itself a cohesion problem.
- **`count_tokens`.** The 4 definitions in `core/Chunking/strategies/tokens.py` are legitimate
  tokenizer-protocol implementations. Not a target. (The *tiktoken fallback* rewrite in cluster
  6 is a real and separate finding — do not conflate them.)
- **`*_to_dict` (~91 hits).** Per-table typed row mappers over genuinely distinct schemas, 40+
  in `core/DB_Management/Sync_DB.py:1578`+. Not shareable. `justified-divergence`.
- **`def sanitize` (62 hits).** Almost all domain-specific payload scrubbers. The genuine
  path-component cluster is only 3 wide and **mutually incompatible**:
  `core/Workflows/adapters/_common.py:71` (allowlist) vs
  `core/Image_Generation/reference_images.py:134` (denylist) vs `core/Utils/Utils.py:680` (a
  third charset). File that as a **security-consistency** finding — three different answers to
  "what is a safe path component" — not as a DRY finding.

**Temp-file lifecycle** is mostly fine (~22 sites, nearly all already
`tempfile.TemporaryDirectory`). The one live sub-cluster worth reporting is 7 sibling sandbox
runners doing the same bare `mkdtemp` + manual-cleanup dance:
`core/Sandbox/runners/{worktree_runner.py:229,503,529, docker_runner.py:956,
firecracker_runner.py:297, seatbelt_runner.py:311, vz_linux_runner.py:448}`.

### 8. Two anchor bugs — your calibration bar

Both were found incidentally by the duplication sweep, which is the point: **divergence between
copies is a bug generator.** Include both in your report, and use them to calibrate what "a
concrete failure scenario" means under the Axis 4 drop rule.

1. **Dead regex from a double-escape.** `core/LLM_Calls/error_utils.py:145` and
   `core/Chat/chat_orchestrator.py:268` use `r"HTTP\\s+(\\d{3})"` — double-escaped inside a raw
   string, so the pattern looks for a literal backslash and **never matches**. The sibling copy
   at `core/Local_LLM/http_utils.py:72` has the correct `r"HTTP\s+(\d{3})"`. Failure scenario:
   an upstream provider error carrying `HTTP 429` in its message text fails status extraction
   and is misclassified, in 2 of 3 copies.
2. **Two sibling services, same helper name, divergent correctness.** Both
   `app/services/workflows_webhook_dlq_service.py` and
   `app/services/meetings_webhook_dlq_service.py` define a private `_now_iso()`. They do not
   agree:
   - `workflows_webhook_dlq_service.py:47` → `_dt.datetime.utcnow().isoformat()` — **tz-naive**.
   - `meetings_webhook_dlq_service.py:41` → `datetime.now(timezone.utc).replace(microsecond=0).isoformat()`
     — tz-aware, and truncated to whole seconds.

   So the same-named helper in two sibling files differs on **both** timezone-awareness and
   precision. Failure scenario: workflow DLQ timestamps serialize with no UTC offset and compare
   incorrectly against the tz-aware values the rest of the system emits — silently, with the
   error scaling by the host's UTC offset — while the two services' DLQ records are not directly
   comparable to each other at all.

   This is the cleanest available illustration of the review's whole thesis: **copies drift, and
   the drift is invisible because both look fine in isolation.** Note also that the broader class
   is large — **239 `datetime.utcnow()` sites** across `app/` — and that `datetime.utcnow()` is
   deprecated as of Python 3.12, which is what CI runs. Treat the class as a real finding; treat
   the specific pair as the calibration example.

### 9. Scoping the layering finding correctly

There are **161** `core/` → `app/api/` imports across **107** files. Do not report this as one
undifferentiated blob, and do not recommend a mass refactor. Separate two shapes:

- **Mild — schema-only imports** from `api/v1/schemas/*`. The common case. The honest reading is
  that the *schemas are in the wrong package*, not that core is wrong to need them. Example:
  `core/Research/service.py:13`.
- **True inversion — core importing `api/v1/endpoints/*` or `api/v1/API_Deps/*`.** Example, and
  the worst class: `core/Embeddings/services/jobs_worker.py:42` imports from
  `api/v1/endpoints/media_embeddings` (also deferred imports at `:656`, `:990`). Also
  `core/Embeddings/audit_adapter.py:23` → `api/v1/API_Deps/Audit_DB_Deps`.

The operative rule, `Docs/Architecture.md`:

> Clients → FastAPI endpoints → Core domain services → Databases / Vector stores / External providers

> The goal is to keep endpoints thin, push logic into core modules, and keep storage access
> centralized via `core/DB_Management/` and the vector store adapters.

and on `DB_Management/`: **"no raw SQL in endpoints."**

**Precedent for the correct fix already exists:**
`tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py` is an AST-based import
ratchet with a ban list and a required re-export set. The proportionate recommendation is a
**sibling ratchet test seeded at the current 107 files** so the number can only go down — not a
mass refactor. Recommend that shape.

### 10. Constraints your recommendations must satisfy

A recommendation that cannot be merged is not a recommendation. Check each against these:

- **Owner-only paths.** Per `CONTRIBUTING.md`, non-owner PRs may not modify
  `tldw_Server_API/app/api/v1/**`, `tldw_Server_API/app/main.py`, `admin-ui/**`,
  `apps/tldw-frontend/**`, `apps/extension/**`, `apps/packages/ui/**`, or root licensing files.
  Since `api/v1/endpoints/` is in scope, **explicitly label any recommendation touching the v1
  API boundary as owner-only work.**
- **Design-first.** Per `AGENTS.md` / `CLAUDE.md`, non-trivial refactors require a design doc at
  `Docs/Design/YYYY-MM-DD-<slug>-design.md`, an ADR entry if it is a decision, a Backlog task
  linking both, and `IMPLEMENTATION_PLAN_<slug>.md` with 3–5 staged goals. Your migration plan
  should say which findings need that treatment versus which are small enough not to.
- **Backlog is the ledger of record** (~3,110 task files) and is managed via its MCP/CLI.
  **Never hand-edit a task file.** You are read-only, so propose tasks in your report; do not
  create them.
- **Base branch: `CONTRIBUTING.md:86,121` says PRs go against `dev`; `origin/HEAD` currently
  resolves to `main`, and both branches exist.** The repo is ambiguous here — say which you
  assumed if it matters to a recommendation, rather than asserting one.
- **Bandit runs on touched scope** (`Docs/ADR/005-bandit-touched-scope-security-gate.md`);
  security findings must clear HIGH/CRITICAL.
- **Six required CI gates** are contractual by name, per
  `Docs/Development/CI_REQUIRED_GATES.md`: `backend-required`, `security-required`,
  `coverage-required`, `frontend-required`, `e2e-required`, `container-build-check`. Any new gate
  you propose must fit that contract.

### 11. Orchestration protocol

Fan out, then synthesize. Do not attempt this in one linear pass — the scope is ~800k LOC.

**Phase A — orient (you, not subagents).** Read the prior art in section 2. Confirm which
modules already have ledgers. Do not skip this; it determines what the subagents are told.

**Phase B — fan out, one subagent per module.** Give each subagent: its module and LOC, the
seed clusters from section 6 **that touch its module**, the do-not-seed list from section 7, the
noise floor from section 5, the five axes with drop rules from section 4, and the import-grep
rule for locating tests.

Every finding — whether produced by a subagent, or by you directly if you are running this
scoped to a single module without fan-out — uses **this schema, which is the report format, not
merely an inter-agent wire format**:

```
FINDING <module>-<n>
  axis:        duplication | encapsulation | sequential-coupling | correctness | efficiency
  class:       adoption-gap | true-duplication | justified-divergence | n/a
  severity:    High | Medium | Low
  sites:       <file:symbol (line-range)>, ... (every site, not "and others")
  canonical:   <existing helper path, or NONE>
  destination: <proposed cohesive module + its single responsibility, or n/a>
  knowledge:   <the shared knowledge that will diverge>
  scenario:    <concrete failure scenario — REQUIRED for correctness axis>
  impact:      <why this severity, for any axis — the consequence that makes it High vs Low.
                Optional but strongly encouraged on duplication/encapsulation findings, where
                severity is otherwise unassessable.>
  cost-driver: <REQUIRED for efficiency axis, with what it scales with>
  tests:       <test files covering these sites, found by import-grep>
  effort:      <cheap | moderate | expensive, and why — this is a ranking input, so it needs
                its own field. Gate it on coverage: well-covered sites are cheap to migrate.>
  owner-only:  yes | no
  confidence:  confirmed | probable-risk | assumption. A finding may split — e.g.
               "confirmed (the duplication); probable-risk (the downstream consequence)".
               Write both rather than averaging them down to one.
```

**On `tests:` and what "coverage" means here.** Unless you actually execute the suite, you are
reporting **import-grep reachability**, not measured coverage. That is acceptable and is the
expected default — but **label it as such** and do not call it coverage. Do not run the full
suite for a review; it is large.

**Prioritize within a module by size × churn**, the way the RAG review's stage1 churn baseline
did. The table below is the **repo-wide** hot set over the last 12 months — most in-scope modules
have no entry in it. If your module is absent, **derive its own churn** rather than falling back
to size alone, which is a worse heuristic:

```
git log --since='12 months ago' --name-only --pretty=format: -- <your module path> \
  | grep '\.py$' | sort | uniq -c | sort -rn | head -20
```

Repo-wide hot set:

| File | LOC | commits/12mo |
|---|---|---|
| `core/DB_Management/ChaChaNotes_DB.py` | 45,292 | **409** |
| `api/v1/endpoints/chat.py` | 8,077 | 204 |
| `core/config.py` | — | 173 |
| `core/Chat/chat_service.py` | 7,285 | 148 |
| `core/Jobs/manager.py` | 12,455 | 135 |
| `api/v1/endpoints/persona.py` | 11,124 | 132 |
| `core/Sync/v2/service.py` | 11,361 | 127 |
| `core/RAG/rag_service/unified_pipeline.py` | 9,586 | 105 |

`ChaChaNotes_DB.py` is simultaneously the **largest file in the repo and the most-changed** —
highest blast radius by a wide margin. Do not let a subagent spend its budget on a quiet
500-line module while that file goes unread.

**There is already a successful precedent for decomposing a god module in this repo.**
`core/DB_Management/Media_DB_v2.py` was a monolith with 121 commits of churn; it no longer
exists as a file and has been split into the `core/DB_Management/media_db/` package (`api.py`,
`constants.py`, `errors.py`, `legacy_content_queries.py`, `runtime/`, and so on). When you
recommend anything for `ChaChaNotes_DB.py`, **point at that migration as the template** rather
than inventing a shape — it is the same team, the same layer, and it already shipped. Read the
`media_db/` package's boundaries before proposing ChaChaNotes' ones.

**Phase C — cross-module dedupe.** The same helper reimplemented across six modules is **ONE
finding with six sites**, not six findings. Merge before ranking. This is where a fan-out review
usually inflates its own numbers; do not let it.

**Phase D — rank and write.** Per section 12.

### 12. Ranking and output contract

**Ranking:** severity × blast-radius × effort, using the numeric-weight-plus-High/Medium
vocabulary from `audits/2026-07-04-test-suite-audit-round2.md` — a priority table of
`| ID | finding | weight | class |` rows, sorted by weight descending. Blast radius is
site count × churn of the files involved. Effort is gated by test coverage: a consolidation
across well-covered sites is cheaper than one across thin coverage, so **state the coverage**.

**This output section yields to the invoker.** If whoever handed you this prompt specified a
different output target — return findings inline, write elsewhere, write nothing — that
instruction **overrides everything below**, even though what follows is long and specific and
this paragraph is short. Observed failure: in a 10-module validation run every agent was told
"do NOT write output files, return findings inline," and all ten wrote full ledgers anyway,
because the detail below outweighed the brief override in their attention. If you are about to
create a file and your invoker told you not to, stop and return the findings instead.

**Output paths** — documentation only, nothing else:

1. **Per-module ledger**, one directory each: `Docs/superpowers/reviews/<module-slug>/`
   - `README.md` — stage order, canonical links, and these four rules quoted verbatim from
     `Docs/superpowers/reviews/rag/README.md`:
     - Write findings before suggested actions in every stage file.
     - Label uncertain items as probable risks or assumptions instead of confirmed defects.
     - Keep later-stage summaries pointed back to the stage files; do not replace the per-stage
       record with a rolling summary.
     - Keep the stage files as the durable review ledger for this audit.
   - Stage files named `YYYY-MM-DD-stageN-<slug>.md`, following the RAG arc: stage1 architecture
     survey + inventory → stage2 core orchestration → stage3 API/schema boundaries → stage4
     data-source boundaries → stage5 composition → stage6 test gaps + synthesis. Collapse stages
     that genuinely do not apply to a module rather than padding them.
   - Per-stage section order: `## Scope` → `## Code Paths Reviewed` (cited as
     `file:symbol (line-range)`) → `## Tests Reviewed` (each test file, what it protects, and
     whether it downgrades the risk) → `## Validation Commands` → findings → suggested actions.
   - **`## Validation Commands` must record the command *and its observed output*** — e.g.
     `71 passed, 508 warnings` — per the RAG ledger's practice. A command without its output is
     not validation.
   - Machine-generated inventories go in `.txt` sidecars, not in the prose files, mirroring
     RAG's `stage1-source-inventory.txt`, `stage1-hotspot-sizes.txt`,
     `stage1-churn-baseline.txt`, `stage1-test-inventory.txt`.
   - For `rag/` and `db-management/`, **add** dated stage files to the existing directory and
     state explicitly which prior findings you confirm as still-live versus already-addressed.

2. **Cross-module synthesis:**
   `Docs/superpowers/reviews/2026-09-21-core-module-duplication-synthesis.md`
   - Header block matching the audits house style: `**Date:** / **Scope:** / **Method:** /
     **Inclusion rule:** / **Relationship to prior reviews:**`.
   - The ranked priority table.
   - Every finding as `### N.M Title (ID)`.
   - A **`## core/Utils Migration Plan`** section: for each accepted consolidation, the proposed
     cohesive destination module and its single responsibility, the sites to migrate, the
     canonical helper if one already exists, sequencing (cheapest well-covered wins first),
     which items need the design-doc treatment, and which are owner-only.
   - A **`## Dead Helpers`** section: helpers with zero importers, recommended for deletion
     rather than migration. Deletion is the cheapest win available and should be called out
     separately from consolidation.

**Useful commands.** Local CI is `make ci-local-full` (ruff all + full suite `-n auto`);
`make lint-changed` runs ruff on changed files; `make test-triage` runs
`Helper_Scripts/ci/test_quality_triage.py`. Pytest has ~60 strict markers (`--strict-markers`),
`asyncio_mode = "auto"`, and `testpaths = ["tldw_Server_API/tests",
"tldw_Server_API/app/core/MCP_unified/tests"]`. Note the documented gotcha in `pyproject.toml`:
the `plugins` key under `[tool.pytest.ini_options]` is **not** a real pytest key and is silently
ignored — plugins load via `pytest_plugins` in the root `conftest.py`.

### 13. Final quality bar

Before you write anything, apply these to every finding:

1. **Evidence or drop.** Every finding cites `file:symbol (line-range)`. No finding rests on a
   filename, a module name, or a grep count alone.
2. **Cite all sites.** "and others" is not acceptable — enumerate, or narrow the claim.
   Enumerating is what turns "there's some duplication here" into a taxonomy with named
   variants, so do not skip it. For clusters **over ~20 sites**, per-file grouping with an
   exact count per file plus full line lists for the in-scope files is acceptable; a bare
   total is not.
3. **No new abstraction where a helper already exists.** Search first. An `adoption-gap` finding
   is more valuable and cheaper to fix than a `true-duplication` one; do not upgrade the former
   into the latter by failing to look.
4. **No recommendation that grows `Utils.py` or `http_client.py`.**
5. **Mark confidence honestly.** `confirmed` / `probable-risk` / `assumption`, per the RAG
   ledger's rule. An unverified inference labelled as a defect is worse than no finding.
6. **Check against `Docs/ADR/` and the Backlog before asserting.** A finding contradicting a
   binding ADR is invalid. A finding whose *recommended fix* would undo something a closed
   Backlog task deliberately relied on is worse than invalid — it is actively harmful. Run
   `backlog search` on the symbol before recommending a deletion.
7. **Prefer a few defensible findings over exhaustive commentary** — the rubric's Core Rule. If
   you produce 400 findings, you have produced zero.
8. **State what you did not cover.** Scope you skipped, and why, in the synthesis. A review that
   silently leaves gaps is worse than one that names them.
