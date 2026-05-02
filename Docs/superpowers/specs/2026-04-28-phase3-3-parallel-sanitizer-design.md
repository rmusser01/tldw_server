# Phase 3.3 Parallel Sanitizer Design

## Goal

Complete the remaining Phase 3.3 conservative-plus error-handler adoption work by parallelizing only safe, covered sanitizer tranches across the app. The work should reduce raw exception/path/token exposure in fallback logs and generic failure payloads without changing validation-facing errors, not-found/conflict behavior, or public diagnostic contracts already covered by tests.

## Current Context

The active worktree is `.claude/worktrees/phase3.3-error-handler-adoption` on branch `worktree-phase3.3-error-handler-adoption`. The branch is clean and locally ahead of the remote with recent Phase 3.3 commits covering RAG, vector stores, storage, skills, TTS, File Artifacts, Sharing, setup, AuthNZ, Prompt Studio, and related sanitizer tranches.

The design baseline is commit `b045c6978f7b3cb3ebf29b58b1f5d3d743739c5d`. Before dispatching workers, the parent must record the current `git rev-parse HEAD` in the candidate matrix. Workers must refresh line numbers in their shard worktree before writing tests or patches because scan line numbers drift as commits are applied.

The remaining raw-error scan still contains a mix of small candidates and giant files. The strategy is not to edit every grep hit. It is to classify candidates first, then dispatch only independent, covered work.

Use this scan as the reproducible starting point:

```bash
rg -n "exc_info=|str\(e\)|str\(exc\)|str\(error\)|error=str|detail=str|detail=f\".*\{e\}|detail=f\".*\{exc\}\" tldw_Server_API/app -g '*.py'
```

The parent may add narrower follow-up scans by directory. The candidate matrix must include: baseline commit, source file, refreshed line number, function/branch, pattern type, exposed surface, existing tests reviewed, proposed safe label/helper, owned test file, red-test strategy, worker shard, and skip/patch decision.

## Execution Model

Use a two-layer workflow:

1. The parent coordinator builds a candidate matrix from the current raw-error scan and assigns disjoint worker shards.
2. Implementation workers run in isolated shard worktrees or produce patch bundles that the parent applies serially. They must not edit the shared Phase 3.3 checkout directly unless the parent explicitly assigns a single active writer.
3. Worker agents each own a narrow source/test slice, add red/green tests, patch only covered fallback behavior, and return without staging in the parent worktree.
4. The parent reviews diffs, applies or cherry-picks worker changes one shard at a time, runs combined verification, updates the Phase 3.3 plan, and commits in small logical batches.

Workers must skip rather than patch when behavior is ambiguous, public-facing by contract, validation-facing, or not testable without broad integration setup.

## Worker Matrix

| Shard | Ownership | Scope |
| --- | --- | --- |
| RAG tail | Exact files and owned tests from the candidate matrix, initially `document_grader.py`, `guardrails.py`, and specific `research_agent.py` sites if isolated | Direct unit-testable metadata/log fallbacks only. Skip `unified_pipeline.py` unless separately approved. |
| API deps health/cache | `DB_Deps.py`, `ChaCha_Notes_DB_Deps.py`, `kanban_deps.py` | Public health `last_error` and cache/close fallback logs only where existing or cheap tests exist. |
| Small endpoint fallbacks | `skills.py`, `llamacpp.py`, `chunking.py`, `vector_stores_openai.py` | Generic `500` or fallback logs only. Preserve `400/404/409/422`. |
| Web/search/core IO | `WebSearch_APIs.py`, `Web_Search.py`, and exact scraping helper files from the candidate matrix | Network/backend failure logs and metadata covered by monkeypatchable tests. |
| Ingestion helpers | PDF/book/audio/plaintext helpers | Fallback logs exposed by existing ingestion unit tests. No ingestion pipeline rewrites. |
| Chat/TTS/core services | `chat_orchestrator.py`, `tts_service_v2.py`, and exact service files from the candidate matrix | Existing sanitizer tests preferred. Avoid streaming giants unless a focused test already exists. |
| MCP small-core | `MCP_unified/protocol.py`, small module implementations | Direct unit-testable fallback returns/logs. Avoid `mcp_hub_management.py`. |
| Test inventory | No source edits | Find existing focused tests and report safe candidates or skip reasons. |

The parent may start with the test-inventory shard plus a few obvious low-risk shards. Each worker must treat its source and test ownership as exclusive and must not edit plan files, stage, commit, or push.

Public health fields such as `last_error` are eligible only when the response schema remains unchanged and tests prove the value is a fixed safe label or sanitized type label. If existing tests assert raw diagnostic details, the candidate is skipped unless the user separately approves changing that public contract.

## Sanitizer Contract

Approved replacement patterns are:

- Fixed labels for public payloads, for example `search_failed`, `grading_error`, or `Error details unavailable`.
- Exception type labels in internal debug logs, for example `error_type={type(exc).__name__}`.
- Existing project sanitizer helpers when already present in the touched module or adjacent tests.
- Generic log messages with no exception object, no traceback extra, and no user-controlled identifiers.

Sanitized outputs must not include:

- Raw exception `str(...)` or `repr(...)`.
- Filesystem paths, DSNs, database names, URLs with credentials, request bodies, prompts, transcripts, queries, document IDs, or user IDs unless the existing public contract explicitly requires them.
- Tokens, API keys, bearer values, passwords, cookies, or auth headers.
- Loguru `exc_info=True` or `exc_info=<exception>` on covered fallback logs.
- Other traceback-bearing paths on covered fallback logs, including `logger.exception(...)`, `logger.opt(exception=True)`, `traceback.format_exc()`, traceback strings in `extra`, or traceback-bearing metadata.

Sanitized outputs may retain:

- Safe operation names.
- Safe exception type names.
- Stable status codes and response schema.
- Existing validation, not-found, conflict, and rate-limit details that are not part of generic fallback sanitization.

## Verification Contract

Each worker must report:

- Files changed.
- Red test failure summary.
- Green focused test result.
- Full touched test-file result.
- Source-scope Bandit result path.
- Confirmation that Bandit `results`, `errors`, and `skipped` entries for touched files were reviewed.
- Confirmation that there are no new Bandit findings. If a touched file has pre-existing unrelated findings, the worker must document the baseline finding and prove the patch did not introduce it.
- `git diff --check` result.
- Skipped candidates and the reason for each skip.

Before each commit, the parent must run:

- Manual diff review.
- Combined touched pytest selection for the batch.
- Bandit over all touched source files in the batch.
- `git diff --check`.
- `git diff --cached --check`.
- `git status --short --branch`.

The parent must update `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md` with `**Recent Update**` lines before staging each implementation batch.

For log-only changes, acceptable red/green proof includes a focused test using a Loguru sink, logger stub, `caplog`, monkeypatched logger, or existing sanitizer regression test. The worker must run the focused test against the unpatched shard baseline and show it failing for a concrete leak or traceback extra, then run the same focused test after the source patch and show it passing while preserving fallback behavior. The worker must include enough output for the parent to verify the red and green failure mode.

## Commit Batching

Use small logical commits:

1. RAG tail plus small core service tranches.
2. API deps and small endpoint tranches.
3. Ingestion, web/search, and core IO tranches.
4. MCP small-core tranches, if any are safely coverable.

Do not push unless explicitly requested.

## Skip Rules

Skip these by default:

- Validation-facing `400/422` details.
- Not-found `404` and conflict `409` details.
- Public diagnostic payloads already asserted by tests.
- Giant files without isolated focused tests, including `ChaChaNotes_DB.py`, `mcp_hub_management.py`, `characters_endpoint.py`, `sync.py`, `chat.py`, and `unified_pipeline.py`.
- Any candidate requiring broad integration setup or behavior inference.

Skipped items should be recorded in the parent summary so they can become a separate, explicitly approved tranche later.

Before changing any public payload or response metadata, workers must search and read existing tests for that endpoint/module. The candidate matrix must record the exact test file and assertion that allows the change, or record that no public-contract test exists and explain why the new red/green test is sufficient. If existing tests assert raw diagnostic details, the candidate is skipped unless the user separately approves the contract change.

## Success Criteria

- Every source change has a red/green test proving the sanitized behavior.
- No worker writes outside its ownership.
- Parent-side combined verification passes before every commit.
- Bandit source-scope scans introduce no new findings for touched source files.
- If pre-existing Bandit findings exist in a touched file, the parent confirms no new findings were introduced and records the baseline finding in the batch summary.
- The branch remains clean after each commit.
- The work advances Phase 3.3 without expanding into Phase 4/5 architectural changes.
