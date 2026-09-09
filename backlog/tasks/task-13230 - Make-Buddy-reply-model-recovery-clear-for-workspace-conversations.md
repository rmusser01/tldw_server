---
id: TASK-13230
title: Make Buddy reply model recovery clear for workspace conversations
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:18'
updated_date: '2026-09-09 07:14'
labels: []
dependencies: []
references:
  - TASK-13227
documentation:
  - Docs/Reviews/2026-09-09-buddy-v1-qualification.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-13227, a newly created workspace conversation generated successfully using the UI default model but its Buddy reply returned Choose a Chat provider and model before sending. Entering provider/model in collapsed Reply model settings recovered the reply. Clarify or repair the handoff of conversation model settings and preflight missing settings before Send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Buddy can reuse valid model/provider settings from a newly generated workspace conversation without requiring redundant manual identifiers.
- [x] #2 If no usable settings exist, required recovery controls and explanatory text are visible before sending and the user's draft is retained.
- [x] #3 Provider overrides remain explicit and cannot silently redirect the target conversation or cross server/account boundaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md. Reason: repair existing Chat settings persistence and Buddy reply preflight; no new schema, credential store or provider authority. Trace actual Research Workspace generation through authoritative Chat creation/update and preserve the selected provider/model in the conversation's existing settings. Buddy preflight must use settings from the exact authorized conversation, show required model/provider recovery visibly when absent, retain drafts, and keep overrides explicit. Add failure-first contract regressions for persistence and scoped recovery, including target changes and unchanged account/server fences.
Expose an authenticated, attached-target-only reply-settings projection because the ordinary public Chat settings response intentionally omits roleplay resume state. Share its effective-completion resolver with Buddy turn acceptance; return only nullable provider/model, private/no-store. Keep the existing ADR-005 authority boundary.

PR2934 Qodo findings 1-4 follow-up: verify the current route/core split and document the existing public contracts. Move requested-save, explicit-provider/model and target eligibility into the existing workspace Chat persistence helper, preserving atomic settings merge and all Buddy/principal/Persona guards; keep the route responsible for parsed inputs and delegation. Add precise docstrings and concrete test-fixture annotations, retain real HTTP/SQLite handoff and preflight coverage, and run targeted tests plus scoped Ruff/Black/Bandit. After runtime and endpoint-docstring freeze, regenerate the canonical OpenAPI fingerprint and frontend API types; review the generated delta for the intended Buddy reply-settings route/schema and nullable conversation timestamp, and run the exact drift/client checks. ADR required: no; existing backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md governs the unchanged authority and persistence contracts. Preserve all published evidence and append a separately attributed review receipt.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Saved explicit provider/model from owned neutral workspaceChat with atomic merge; excludes Buddy overrides, omitted choices, global and tracked roleplay behavior. Added current-attachment-only reply-settings projection sharing acceptance resolver. UI shows effective settings, opens required fields beforeSend, keeps drafts and rejects unchecked/stale targets. Review fixed transport config race using pinned factory with deferred regression. Final backend58passed1Postgresunavailable skip; Persona compatibility12passed test-onlyCHAT_FORCE_MOCK; UI94passed; productionBandit0findings; focusedentrypointtypecheck0errors. Live explicit recovery completed exactturn156c528764bb46ffaa23d5a7f4e7c1cf; canonicalhandoff covered by realHTTP/SQLitetests, subsequentbrowserhandoff limited by disposableconnectionrecovery. ExistingADR-005,user/API docs,sourceboundevidence updated. PR2934.

PR2934 review follow-up started after verifying Qodo findings against source. Official MCP task reads were unresponsive and terminated; using the explicitly documented CLI fallback. Historical completion and qualification receipts remain attributed to their original source.
PR2934 generated-contract repair: after the Buddy endpoint/schema docstring freeze, regenerated the canonical OpenAPI fingerprint and ignored frontend artifacts. The committed fingerprint is now 2086 paths, 3163 schemas, sha256 f5792a491dd56cc1480f13532c30c98c15e90fce3bfeaab0358e861ea9a6f316. Generated schema.d.ts exposes the attached-target reply-settings GET, required nullable provider/model, and optional nullable Buddy conversation created_at. Exact drift check passed; extension verification found all 344 ClientPath entries and all 49 media fallback fields, with the existing 10 reviewed OSS exceptions. Local `bun run generate:api-types` first failed because the shared project venv lacks an editable tldw_profile_core install; rerunning the same canonical exporter with this checkout plus packages/tldw_profile_core/src on PYTHONPATH, followed by the script's documented openapi-typescript command, succeeded without dependency installation or global environment changes.
PR2934 Qodo backend findings 1-4 follow-up completed against HEAD 3b73d70f0f5dde12e4831c3441dccf656f64f2e6. The existing Chat persistence helper now owns save_to_db, explicit-selection and target eligibility; the route only delegates parsed inputs. Atomic settings merge, owned-neutral-workspace restriction, Persona/global exclusions and temporary Buddy overrides are unchanged. Added full Args/Returns/Raises contracts to persistence and Buddy reply-settings APIs, plus concrete TestClient/CharactersRAGDB/MagicMock fixture types and purpose/parameter docs in both new Buddy regression modules. Before: 19 focused tests passed. After: 36 Buddy/adjacent tests passed (server-pr2934-backend-after.log), and 12 Persona compatibility cases passed with test-only CHAT_FORCE_MOCK=1 (server-pr2934-persona-compatibility.log). All twelve functions in the three touched test modules have documented, annotated boundaries. Nine-file static review adds zero Ruff findings; inherited debt remains chat.py 3 I001, llm_providers.py 1 and provider_config_resolution.py 17 UP045, explicitly compared with HEAD rather than called clean. Black passes for four complete files and changed ranges in five files; compile and diff whitespace checks pass. Production Bandit reports zero findings/errors; test scan reports only 40 ordinary B101 assertions. Evidence: /private/tmp/server-pr2934-backend-static-summary.log, server-pr2934-ruff-comparison.json, server-pr2934-black.log, server-pr2934-contract-and-bandit-summary.log, server-pr2934-bandit-production.json and server-pr2934-bandit-tests.json. Endpoint docstrings frozen and schema owner notified before OpenAPI regeneration. Existing AC outcomes and published receipts remain unchanged; leave In Progress for root reconciliation of the separate OpenAPI/frontend follow-up. ADR-005 still governs; no commit, push, live provider or full suite.

Root reconciliation: all five backend Qodo findings are addressed and independent review found no runtime regression. Removed an unreachable HTTP 409 statement from the GET docstring and regenerated API artifacts; executable AST is unchanged by that final wording correction. 52 distinct scoped backend cases passed (36 Buddy/adjacent, 12 Persona, 3 catalog unit-marker, 1 configuration), with zero production Bandit findings, zero new Ruff findings versus 21 existing issues, and formatting/compilation checks passing. Combined frontend gate passed 99 tests and the final Chrome build/manifest/ZIP checks passed. Review receipts: Docs/Reviews/artifacts/buddy-pr2934-qodo. ADR-005 applies; no database migration, global provider guess or normal-profile changes. Native/audio/upgrade qualification remains open under TASK13227.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Owned neutral workspace Chat preserves explicit provider/model defaults through the core persistence service. Buddy preflight and turn acceptance share effective-setting resolution; missing settings are visible before Send, drafts survive recovery, and temporary overrides preserve conversation defaults. All reviewed backend contracts, test types/markers and generated API fingerprint are reconciled. 52 scoped backend cases and 99 combined UI tests passed; final Chrome build and API drift verification pass. Static baseline and remaining native qualification limits are recorded in the separate Qodo evidence bundle. ADR-005 remains applicable.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
