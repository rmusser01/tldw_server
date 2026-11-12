# Native assigned dictionary repair implementation plan

> **For agentic workers:** Use the existing executing-plans workflow inline. Root owns independent review and actual runtime acceptance.

**Goal:** Let an explicitly assigned dictionary transform the provider copy of the current native Character user input while retaining authored storage and selected history.

**Architecture:** Materialize dictionary definitions in the existing protected `roleplayBehaviorV1` settings authority during owner-validated assignment. Reuse the existing dictionary entry and iterative processing semantics on detached data; native preparation reads only the digest/version-bound authority, never a live dictionary catalog.

**Tech Stack:** Existing FastAPI/Pydantic, ChaCha storage, pytest/httpx, and dictionary regex/tokenizer utilities. No new dependency or runner.

**Spec / task:** Root's approved B03 design and actual disclosed `unsupported_history_context_saved_settings` outcome are recorded before edits in TASK-13260.278.18.83.45. The parent owns that Backlog record.

## ADR check

**ADR required:** No new ADR. [ADR-049](../ADR/049-chat-history-selection-ownership.md) governs selected history and owner authority. The accepted [immutable Character behavior contract](../superpowers/plans/2026-08-27-character-conversation-behavior-snapshot-contract.md) governs materialized, snapshot-only context. This repair extends its existing protected authority without changing the H1 wire protocol, storage ownership, or credential boundary. Dictionary edits affect an existing accepted native chat only after explicit reassignment; this frozen assignment behavior was assessed and approved by root.

The approved compatibility write occurs before the final H1 capture, under the existing owner transaction/lock and settings CAS. It only adds missing dictionary materialization to otherwise valid owner-assigned legacy state. Unsupported behavior, invalid protected authority, a mismatched source/scope/view, and concurrent assignment refuse without overwriting state. The normalized authority is then immutable for that captured selection; native preparation performs no live catalog reads. This reuses the governing writer and frozen-context decisions rather than introducing a mutable context resolver.

## Global constraints

- Current API 75653 remains loaded old source; root alone owns builds/runtime and live acceptance.
- Preserve all unrelated WIP and the preceding worldbook/required-variable changes.
- Preserve authenticated owner/scope, selection/version/digest, provider binding, and unsupported-effect guards.
- Store authored user text. Transform only current user provider content; leave ancestry, instructions, assistant/tool content, and images intact.
- No global-active dictionaries, mutable catalog reads after capture, new credentials, schema migration, model/network actions, frontend edits, Git/index actions, or parent task edits.
- Use supported Python 3.12 activated environment, normal configured pytest plugins, 300-second case / 600-second outer bounds; maximum three attempts per issue.

## Stage 1: Causal integration regression

**Goal:** Reproduce assignment through real create/entry/PUT settings/reload/history capture/native preparation and injected provider transport.
**Success Criteria:** The positive fails at the actual saved-settings guard on unchanged production; unassigned control proves the harness reaches the real provider payload.
**Tests:** New `tests/Chat_NEW/integration/test_native_assigned_dictionary.py`; exact token output in provider copy and original authored storage, accepted Character instructions and lore preserved.
**Status:** Complete

- [x] Write real owner-scoped integration fixture with task-private official storage and transport injection.
- [x] Run positive/control naturally and record causal red before production edits.

**Evidence:** Initial command had two setup errors because the generic credentialed fixture selected the default AuthNZ path; this is a harness issue, not application evidence. Replaced it with the existing real private `single_user_cookie_client`, retaining cookie/CSRF/auth handlers and ordinary configured plugins. Corrected two-case run: unassigned PASS, assigned FAIL. Failed-positive-only diagnostic rerun: natural exit 1, pytest 9.86s / outer 13.867s; real native stack raises HTTP 409 `unsupported_history_capability` / `unsupported_history_context_saved_settings` before append. No production edit preceded this red.

## Stage 2: Protected materialization and detached processing

**Goal:** Capture assigned definition/include/entry/default-budget data in the existing materialized authority and reuse real processing semantics before admitting a user.
**Success Criteria:** Positive succeeds; invalid/unbound context refuses before any append; old controls remain intact.
**Tests:** New integration controls for unassigned/global spillover, owner/scope, stale settings, malformed authority, captured catalog edit/delete, reassignment, include/order/disabled/case/regex/probability/budget/size; detached processing equivalence with existing service.
**Status:** Complete

**Files:** Existing `core/Character_Chat/chat_dictionary.py` owns capture and shared processing; a focused pure frozen-dictionary module validates/uses detached plans. Existing `character_conversation_factory.py` classifies assignment and materializes plans. Existing `DB_Management/chacha/conversation_resume_store.py` accepts the added closed materialized value. Released `Chat/history_context.py` projects validated assignment and `Chat/chat_service.py` applies it only to the current provider copy before owner append. Existing protected settings reservation remains authoritative.

**Compatibility assessment:** Existing assignments carry only public IDs. Root approved narrowly owner-locked, CAS-protected first-use materialization before history capture after validating other saved behavior, source/owner/view/scope, existing protected authority and active-Sync refusal. This adds one-time legacy settings normalization to the capture endpoint; it does not append messages. Malformed/stale/foreign/unsupported authority is never overwritten. ADR-049's captured digest binds the normalized immutable authority. Fresh settings writes reuse the existing protected carrier and existing classified-behavior writer without endpoint changes. Add the capture endpoint to the scoped files; update its read-only documentation accurately.

- [x] Add only assignment keys plus validated materialized dictionary plan support.
- [x] Extract the existing replacement loop into a shared detached function without changing its semantics or service usage bookkeeping.
- [x] Validate plan and compute provider transformation before owner input admission; preserve exact authored input and selected ancestry.
- [x] Run new module and genuinely affected existing suites.

**Bounds:** At most 64 assigned roots/sources and 10,000 entries, plus the existing 1 MiB authority limit. A conservative UTF-8 estimate bounds entry expansion at the database boundary, including replacement/pattern/timed-effects/group content. Default budgets and partial-result behavior use the existing processor; budget exceedance is its warning/early-stop contract, not a newly introduced hard output limit. Native detached processing does not write mutable dictionary usage/activity data.

**Attempt ledger:** The corrected application red above was followed by two expected legacy compatibility failures before implementation. Initial green: 31 cases; corrected invalid-view/CAS coverage green: 33 cases (pytest 72.70s / outer 79.445s, natural exit 0). The invalid legacy cursor regression first failed because refused capture advanced settings version; prior owner/view resolution now precedes normalization. Additional self-review red found bool/float pass IDs accepted as integers and seven malformed stored catalog rows defaulted or raised generic conversion errors; the focused corrected run passed 19 cases. Two malformed numeric-string cases then demonstrated uncaught `ValueError`; conversion now returns the same explicit `InputError` refusal. A wrong test response field (`stats.token_budget_exceeded` rather than the public top-level field) was a test-only error and was corrected separately.

**Affected checkpoint:** 381 passed / 79 configured skips / 2,970 warnings, pytest 256.66s / outer 262.653s, natural exit 0. Includes both new modules, saved-context/native frozen lore/native Character transport, dictionary endpoint/error-mapping/service/settings-merge suites, and H1 transaction/branch/settings-read suites. Preserve these passes; final capture/API/factory plus updated new-module qualification is running separately. Postgres runtime coverage is not claimed from configured skips.

**Final configured qualification:** Final new-module/capture/factory run: 220 passed / 14 failed / 6,949 warnings, pytest 448.72s / outer 454.678s, natural exit 1. All 63 new cases and the full existing Character behavior API suite passed. The 14 existing completion-related H1 cases failed before context preparation because the Topic Monitoring fixture tried to open its unwritable default worktree database. No production or test edit was used to address that environment failure. Reran only those 14 cases with supported `MONITORING_ALERTS_DB` bound to the task-private path: 14 passed / 452 warnings, pytest 32.05s / outer 36.448s, natural exit 0. Retained all other passing cases. Across these final affected commands there are 557 distinct passing cases (including 63 new cases) and 79 configured skips; source/type/coverage claims do not extend to live model quality or unavailable runtime backends.

## Stage 3: Qualification, independent review, and freeze

**Goal:** Qualify current source without claiming it is loaded or live accepted.
**Success Criteria:** Affected tests and compile checks pass; Ruff/Bandit baseline/current show no new findings; root receives stable exact scope/hashes and reviewable bodies.
**Tests:** Configured affected pytest suites, static compile/Ruff, project-venv AST-only Bandit baseline/current, whitespace and full self-review.
**Status:** In Progress

**Static checkpoint:** Seven production paths and both new modules compile under supported Python 3.12. Configured Ruff baseline/current each have the same ten inherited findings; new code/tests have none. Baseline is evaluated with each original filename so project per-file rules remain identical. Project primary-venv Bandit performs AST-only analysis (no application imports): production baseline/current retain the same three inherited findings (probability RNG B311 and two budget-statistic B105 false positives); new tests have zero findings after documenting two `None` budget false positives. Own-path whitespace is clean. Production freeze recorded at 2026-10-08 23:29:05 UTC; final test qualification and independent review remain pending before runtime load.

- [x] Record counts, natural exits, limitations, and source hashes.
- [x] Send stable source to root for independent review before runtime load.
- [ ] Root performs actual B03 UAT. Remove only this plan when all stages are complete.

**Freeze:** Seven production paths and both new test modules remain unchanged since the 23:29:05 UTC production freeze. Full self-review covered all changed hunks and complete new bodies, the existing settings classification/materialization/owner CAS, capture handshake and active-Sync refusal, detached service semantics, saved-context projection, provider-copy handoff and authored persistence. Root owns independent review and actual-runtime acceptance. Current API 75653 was not restarted or otherwise operated by this worker.
