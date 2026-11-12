# PR3084 privilege registry snapshot consistency — 2026-10-04

Associated task: TASK-13260.278.18.83.45. Published source: ea397bd3d82f58353fb92ac86c8e57002b0a9c0b on dev e70e0abb129ea8010f8096d1080b8ea23570f442.

ADR required: no. This unit regenerates an existing test contract through its supported helper; it changes no architecture or security policy. Task editing follows ADR-059.

## Stage 1: Identify the complete discrepancy
**Goal**: Reproduce the actual new privilege snapshot assertion unchanged and identify every registry difference.
**Success Criteria**: The original target fails naturally on the existing proper stack; the full generated-data delta is attributable to current production route metadata without changing assertions.
**Tests**: One unchanged snapshot target, outer 600 seconds and existing per-case 300-second timeout, no retry. Read complete route, dependency and helper source.
**Status**: Complete

Actual CI job111390062187/run37185438773 is bound to ea397. Setup passed; the snapshot assertion failed. Shard: 1 failed, 2,921 passed, 14 skipped, 5,935 warnings, 2,810.85 seconds. The actual job log was read once in memory; no log/proof file was saved. The failed assertion names the supported update_privilege_registry_snapshot.py helper. All seven required contexts are successful, but this additional full-suite failure remains open.

## Stage 2: Regenerate the existing contract
**Goal**: Refresh only the stale JSON registry through the supported helper when the complete delta is justified.
**Success Criteria**: Every added or changed field matches current source; privilege scopes, dependencies and security behavior remain intact. No manual snapshot substitution, assertion weakening, production/CI/config/environment change or model download is used.
**Tests**: One supported generation bounded at 600 seconds; inspect the whole JSON comparison and source bindings. Stop on unexpected data, environment or lifecycle behavior.
**Status**: Complete

Causal red: the unchanged snapshot target failed naturally (1 failed / 4 warnings / 8.75s pytest / 12.722s outer). Supported generation exited naturally0 in 9.908s. The entire comparison preserves 83 scopes and 282 entries: exactly four existing usage_quota_deps._check dependency objects are added to RAG search (any/rag.search) and Text2SQL query (any/text2sql.query). Removing only those additions restores the complete old JSON, including ordering. All other fields, route/scopes and dependencies are identical; the current route decorators and existing dependency factory justify the metadata. No manual snapshot edit or Python change.

## Stage 3: Verify and publish the reviewed correction
**Goal**: Qualify the generated contract and publish the exact independently reviewed patch through normal hooks.
**Success Criteria**: Original and affected introspection/helper/catalog checks pass naturally, scoped security assessment is recorded, independent actual-source/data review is clear, and live dev/remote-source guards pass before exact-lease publication. Human summary remains verbatim.
**Tests**: Declared affected group only, outer 600 seconds/300-second per-case guard; no stopped lifetime/retention/GC/native controls or unrelated full-suite replay. JSON/prose-only Bandit N/A unless Python changes.
**Status**: In Progress

Affected verification: the full original introspection and catalog-loader files passed naturally (15 passed / 28 warnings / 7.55s pytest / 11.064s outer), with no skips reported. Canonical JSON format/parse and diffcheck pass. Bandit N/A: only generated JSON and prose changed; Python production/tests/helper bytes remain unchanged. Independent full generated-data/source/prose review is CLEAR with no findings; reviewer ran no tests/imports/generator or model controls. Normal exact-patch commit/publication remains pending, with immediate dev/remote guards required. Hosted final-head checks and all broader native/UAT gates remain open.

## Retained holds

The separate ingestion failure remains open, with three diagnostic attempts exhausted. Its local cc549 tracking remains on the original branch and is excluded here. Unexpected partial models/Whisper output is retained untracked/unstaged, without cleanup, reuse, copying or hashing. No new ingestion case/shard or log replay. First-full-import ownership, bound workflow and one uninstrumented whole native macOS Prompt natural-exit gates remain open; retained red native WIP is untouched. The API replacement approval remains unanswered; no API/Node/launcher/frontend/browser action is allowed under this unit. All live UAT criteria remain open; seven hosted required successes do not permit merge.
