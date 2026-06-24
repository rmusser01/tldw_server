# Moderation PolicyCompiler Refactor Design

Task: TASK-2430
Date: 2026-06-24
Status: Draft for user review

## Purpose

Refactor the Moderation module in a compiler-first slice that improves long-term stability without changing public behavior. The first implementation should extract deterministic policy assembly from `ModerationService` into a dedicated `PolicyCompiler`, while preserving `ModerationService` as the public facade and keeping the existing `ModerationPolicy` and `PatternRule` types as the compatibility contract.

This slice intentionally prepares the module for a later mostly pure `PolicyEvaluator`, but does not move evaluation logic yet.

## Current Problem

`moderation_service.py` currently mixes unrelated responsibilities:

- config and environment fallback resolution
- path anchoring and file reads/writes
- runtime and user override persistence
- blocklist parsing and regex compilation
- PII rule inclusion
- effective policy construction
- text evaluation and redaction

That coupling makes precedence hard to reason about and creates risk when changing rule behavior. The most pragmatic first step is to make effective policy construction explicit and testable while leaving the public service and evaluation path intact.

## Goals

- Preserve existing behavior and public method signatures.
- Keep `ModerationService` as the public facade used by chat, endpoints, workflows, and tests.
- Keep returning `ModerationPolicy` from `get_effective_policy()`.
- Extract deterministic policy assembly into a `PolicyCompiler`.
- Make precedence, invalid-rule handling, and category behavior explicit.
- Produce sanitized internal compilation diagnostics without creating a new public endpoint contract.
- Keep file I/O, locking, logging, and persistence in `ModerationService`.

## Non-Goals

- Do not introduce a public `CompiledModerationPolicy` type in this slice.
- Do not freeze or replace `ModerationPolicy` or `PatternRule`.
- Do not extract text evaluation, redaction, match ranking, or result building yet.
- Do not fold guardian/supervised policy overlays into the compiler.
- Do not change moderation endpoint responses unless required to preserve current behavior.
- Do not make load-time invalid rule handling stricter than it is today.

## Proposed Structure

Add `tldw_Server_API/app/core/Moderation/policy_compiler.py`.

The module should contain:

- `PolicyCompiler`: pure policy normalization and rule compilation.
- `PolicyCompilationInput`: a dataclass containing already-loaded inputs.
- `PolicyCompilationResult`: a dataclass containing the compatible `ModerationPolicy` plus `PolicyCompilationReport`.
- `PolicyCompilationReport`: sanitized internal diagnostics for skipped or normalized inputs.

The compiler should return existing `ModerationPolicy` objects. The optional report is for service logging and tests, not a public API contract.

## Responsibilities

`ModerationService` keeps:

- config loading from `load_and_log_configs()` and fallback parser loading
- environment fallback lookup
- relative path anchoring
- blocklist, runtime override, and user override file reads
- all writes and file mutation locks
- warning and error logging
- reload behavior
- public APIs such as `get_effective_policy()`, `effective_policy_snapshot()`, `update_settings()`, `check_text()`, `evaluate_text()`, and `redact_text()`

`PolicyCompiler` owns:

- boolean and category normalization
- runtime override application
- blocklist line parsing into `PatternRule`
- regex flag parsing and safety validation through one shared owner
- PII rule inclusion when effective PII is enabled
- user override field application
- per-user quick-rule compilation
- effective `ModerationPolicy` construction
- sanitized report entries for skipped rules and normalization issues

## Inputs

The compiler receives already-loaded data, not file paths:

- base moderation config mapping
- environment-derived values that the service has already resolved
- runtime override mapping
- blocklist lines
- optional user override mapping
- built-in PII rules supplied by `ModerationService` through a provider callback or explicit list

The compiler must not call `open()`, inspect filesystem paths, create directories, or persist anything.

## Precedence

The first implementation should preserve the current effective order:

1. Base moderation config and service-resolved defaults.
2. Runtime overrides for `pii_enabled` and `categories_enabled`.
3. Blocklist lines compiled into `PatternRule`.
4. Built-in PII rules appended only when effective PII is enabled.
5. Global `ModerationPolicy` construction.
6. User override field application when per-user overrides are enabled and a user override exists.
7. User category override resolution.
8. User quick-rule compilation and append.

The service should still cache/store the current global policy as it does today. Recompile points remain reload, runtime setting changes, blocklist mutations, and user override changes.

## Compatibility Details

`get_effective_policy(user_id)` must continue returning `ModerationPolicy`.

Existing callers that construct `ModerationPolicy` directly should continue working. Existing supervised policy overlay behavior should continue composing by receiving and returning `ModerationPolicy`.

The design must preserve the current category distinction where `categories_enabled=None` means no configured filter and user override `categories_enabled=""` normalizes to `set()`. Evaluation currently treats falsy filters as allow-all, but policy construction should not collapse those states earlier than today.

Per-user quick-list rules should remain category-agnostic by using the existing wildcard behavior so they still apply when category filters are enabled.

## Invalid Input Handling

The compiler should continue forgiving load-time behavior:

- invalid blocklist lines are skipped
- invalid actions are skipped
- dangerous regexes are skipped
- invalid regexes are skipped
- malformed loaded user rules are skipped
- invalid loaded user override fields are dropped or ignored consistently with current sanitization

Strict API validation for user override writes remains separate from forgiving load-time sanitization. The refactor should not merge these paths into one stricter compiler path.

`PolicyCompilationReport` should record sanitized diagnostics only. Report entries should use reason codes such as `invalid_action`, `invalid_regex`, `dangerous_regex`, `invalid_phase`, or `invalid_is_regex`, plus source kind and line/rule index when useful. They should not store raw dangerous regex text, sensitive matched content, or full filesystem paths.

`ModerationService` may convert report entries into the same style of warning logs emitted today.

## Regex And Parser Ownership

Regex parsing and regex safety checks should have one owner after the refactor. The preferred first-slice shape is for `PolicyCompiler` to own blocklist parsing and rule compilation helpers, while existing service lint and validation methods delegate to the shared compiler/parser helpers where behavior overlaps.

The implementation should avoid copying the same regex flag parsing and dangerous-regex heuristics into multiple modules.

## PII Rule Boundary

Built-in PII rule loading should stay explicit. The compiler should not import PII detector dependencies implicitly. `ModerationService` should provide PII `PatternRule` values through a small provider callback or by calling an existing service method and passing the resulting rules into the compiler.

This keeps compiler tests deterministic and makes dependency failures visible at the service boundary.

## Supervised Policy Boundary

Guardian and supervised policy behavior remains outside this compiler slice. `supervised_policy.py` can continue to overlay guardian rules onto a `ModerationPolicy` returned by `ModerationService`.

This avoids mixing DB-backed guardian schedule/chat-type filtering with local moderation policy assembly. A later design can decide whether supervised overlays should have their own compiler, but this slice should not expand that scope.

## Testing Plan

Add focused compiler unit tests for:

- base config defaults and boolean normalization
- runtime override precedence
- category parsing for lists, strings, empty strings, and invalid types
- blocklist literal and regex parsing
- invalid action, invalid regex, invalid flags, and dangerous regex reports
- effective PII enablement and appended PII rules
- user override field precedence
- user category override empty-string behavior
- user quick-rule compilation and wildcard categories
- sanitized report contents

Keep service compatibility tests around:

- `get_effective_policy()`
- `effective_policy_snapshot()`
- `update_settings()`
- `reload()`
- blocklist mutation-triggered recompilation
- user override mutation-triggered recompilation
- existing evaluation methods consuming compiler-produced `ModerationPolicy`

Keep regression coverage for supervised policy overlays to prove they still consume and return `ModerationPolicy` without behavioral drift.

## Verification Plan

For the implementation branch, run:

- targeted Moderation unit and Guardian tests touched by the refactor
- Python compile checks for touched Moderation modules and tests
- `git diff --check`
- Bandit on `tldw_Server_API/app/core/Moderation`

For this design-only branch, verify:

- the spec has no draft markers or unfinished sections
- Backlog task acceptance criteria are aligned with this document
- `git diff --check` passes

## Future Refactor Path

After this compiler slice lands, the next practical slice is a mostly pure `PolicyEvaluator` behind `ModerationService`. That evaluator can consume `ModerationPolicy` produced by the compiler and own rule matching, category/phase gating, redaction decisions, and result construction.

Keeping the compiler slice first reduces the risk of moving evaluation logic while policy construction remains implicit and stateful.
