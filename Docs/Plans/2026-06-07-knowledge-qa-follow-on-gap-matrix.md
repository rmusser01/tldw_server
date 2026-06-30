# Knowledge QA Follow-On Gap Matrix

## Scope Boundary

`/knowledge` is a Knowledge QA workflow for searching a personal library and reviewing grounded answers with citations. Flashcards, decks, spaced repetition, and study-set behavior are out of scope.

This matrix starts after `TASK-528`. It does not reopen the readiness, first-run, no-source, source-control, settings, evidence-view, export, and baseline UAT work that already landed there. It maps the remaining live QA findings to follow-on owner tasks and release gates.

## Current Baseline

`TASK-528` and child tasks already provide these baseline improvements:

| Baseline area | Existing owner | Covered state |
| --- | --- | --- |
| Deterministic state fixtures and route-state coverage | `TASK-528.1` | Backend offline, setup required, empty/no-source, ready search, cited result, no result, settings, export, WebUI/extension route states |
| WebUI readiness timeout and recovery | `TASK-528.2` | `/knowledge` no longer stays blank after health timeout; retry, diagnostics, and settings recovery are visible |
| Extension setup diagnostics | `TASK-528.3` | Missing URL/API key, host permission, allowlist, and backend reachability are surfaced in setup state |
| Beginner empty and no-source recovery | `TASK-528.4` | No indexed source, no selected source, backend unavailable, web-only, and ready states are separated |
| Ready search controls and saved profiles | `TASK-528.5` | Source category selection, exact document/note counts, saved profiles, presets, web fallback, and answer model/provider controls |
| Results, evidence, no-results, and export guardrails | `TASK-528.6` | Citation mappings in export, Sources/Details telemetry when available, no-results recovery, failed search, and route-state coverage |
| Power-user settings and cross-surface parity | `TASK-528.7` | Basic/Expert settings, compact source controls, provider/model recovery, WebUI/extension parity notes |
| UAT and regression guardrails | `TASK-528.8` | Repeatable UAT checklist, user guide, focused shared UI and WebUI regression commands |
| PR review hardening | `TASK-528.9` | Deterministic fixtures and no-results nearest-match evidence behavior |

Known baseline limitation carried forward: extension runtime E2E/UAT remained blocked by the WXT production build stall before browser launch. Follow-on work must treat this as a release risk, not a soft skip.

## Matrix

| Live finding | Existing `TASK-528` coverage | Remaining gap | Owner task | Release gate |
| --- | --- | --- | --- | --- |
| WebUI search returned five sources but zero citations | `TASK-528.6` added visible evidence/export guardrails when citation data exists | Normal answer can still appear without valid citations or a durable trust state | `TASK-2279.2`, `TASK-2279.4` | Normal successful answer requires valid citations that map to returned inspectable evidence |
| Scoped query produced a general uncited answer while web fallback was disabled | `TASK-528.5` hardened source scope controls and saved profiles; `TASK-528.6` improved no-results recovery | Empty or uncitable local evidence can still fall through to a general answer instead of abstention/degraded state | `TASK-2279.4`, `TASK-2279.6` | Empty retrieval and disabled web fallback cannot produce normal success; scope exceptions must be explicit |
| Evidence source preview could show only `Full source content is unavailable` | `TASK-528.6` improved evidence/details display when telemetry is available | Source rows do not yet guarantee an excerpt, chunk, quote, open target, or specific unavailable reason | `TASK-2279.3` | Every source row has inspectable evidence or a specific unavailable reason |
| Source rows reported zero percent match while still supporting a generated answer | `TASK-528.6` surfaced retrieval/details diagnostics when present | Weak or near-zero relevance is not yet part of the trust contract and can still be styled as normal support | `TASK-2279.3`, `TASK-2279.4` | Weak evidence produces abstention or degraded answer with surfaced reason codes, not `cited_answer` |
| Extension showed setup/offline state changes before recovering | `TASK-528.3` added setup diagnostics and reachability checks | Extension still needs explicit runtime state taxonomy for setup missing, invalid setup, unreachable backend, auth failure, and allowlist failure | `TASK-2279.5` | Extension setup, reachability, auth, and allowlist failures have visible distinct recovery states |
| Extension search completed while thread sync failed with messaging timeout | `TASK-528.1` and `TASK-528.3` added route-state and setup coverage; `TASK-528.8` documented extension blocker | Search success and persistence failure are not yet separated in durable UI state | `TASK-2279.2`, `TASK-2279.5`, `TASK-2279.7` | Search success with sync failure becomes an actionable `unsynced_local_result` and is not exported/history-restored as grounded success |
| Export and history can preserve degraded answers without durable trust status | `TASK-528.6` added export content and citation/context fields when available | Trust state, evidence origin, source status, and unsupported-draft labels are not yet persisted across history/export surfaces | `TASK-2279.2`, `TASK-2279.7` | History, recent sessions, restore, and export preserve cited, degraded, unknown, no-results, failed, and unsynced states |
| WXT runtime E2E blocked before browser launch | `TASK-528.8` recorded the WXT production build stall and avoided claiming extension browser behavior was verified | Follow-on release needs an explicit harness health gate before extension signoff | `TASK-2279.5`, `TASK-2279.8` | Extension runtime E2E must launch `options.html#/knowledge`; if blocked, release signoff records command, timeout, owner, and failure artifact |

## Test Inventory

Observed current coverage:

- Shared UI Knowledge QA Vitest coverage under `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/`, including answer states, evidence rail, export, history, source controls, settings, provider/model controls, empty recovery, no-results recovery, and route fixtures.
- WebUI Knowledge QA E2E coverage:
  - `apps/tldw-frontend/e2e/workflows/knowledge-qa.spec.ts`
  - `apps/tldw-frontend/e2e/ux-audit/knowledge-empty-recovery.spec.ts`
  - `apps/tldw-frontend/e2e/ux-audit/knowledge-qa-states.spec.ts`
  - `apps/tldw-frontend/e2e/ux-audit/knowledge-readiness-recovery.spec.ts`
- Extension Knowledge QA E2E coverage:
  - `apps/extension/tests/e2e/knowledge-qa-setup-diagnostics.spec.ts`
  - `apps/extension/tests/e2e/knowledge-empty-recovery.spec.ts`
  - `apps/extension/tests/e2e/knowledge-qa-states.spec.ts`
  - `apps/extension/tests/e2e/knowledge-rag-ux.spec.ts`

## Follow-On Owner Map

- `TASK-2279.2`: trust taxonomy and safe response handling
- `TASK-2279.3`: evidence materialization
- `TASK-2279.4`: citation validity, weak-evidence policy, abstention, and web-fallback origin labeling
- `TASK-2279.5`: extension runtime and sync reliability
- `TASK-2279.6`: scoped-search request/result/profile round trip
- `TASK-2279.7`: export/history trust-state propagation
- `TASK-2279.8`: deterministic live-backend UAT gates
- `TASK-2279.9`: non-blocking evidence workflow improvements

## Stage 0 Result

Stage 0 does not change runtime behavior. It establishes that the follow-on implementation should start from the `TASK-528` baseline and focus on trust state, evidence materialization, citation enforcement, extension runtime reliability, scoped-search auditing, export/history durability, and live UAT gates.
