# User-Customizable Service Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose an allowlisted set of internal content-generation prompts for safe per-user editing in the shared settings UI and browser extension without making control, evaluator, or hidden enforcement prompts editable.

**Architecture:** Implement the approved design as six dependency-ordered review units. A checked-in inventory controls eligibility and stable IDs; a typed registry and resolver provide one resolution path; per-user revisions live in the existing Prompts DB; Context Integrity signs approvals; the Jobs DB holds authenticated execution snapshots and bindings; the shared React settings page consumes a typed API. Domain migrations start only after the inventory matrix is approved.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, SQLite/PostgreSQL Jobs backends, existing Context Integrity HMAC support, pytest/Hypothesis, React/TypeScript, Ant Design, Vitest, Playwright.

---

## Governing artifacts

- Approved specification: `Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md`
- Tracking task: `TASK-12956`
- Parent design task: `TASK-12955`
- Related context-integrity design: `Docs/superpowers/specs/2026-06-25-context-integrity-skills-prompts-design.md`
- Existing context-integrity foundation plan: `Docs/superpowers/plans/2026-06-25-context-integrity-foundation-implementation-plan.md`

## Fixed implementation decisions

1. `service_prompt_id` and `part_id` are code-defined allowlist identifiers. Users cannot create definitions or change contracts.
2. Only user-visible content-generation instructions are eligible. Routing, moderation, evaluators/judges, auth, tool policy, output-schema enforcement, and locked hidden parts are excluded.
3. Resolution is per part: request override → authenticated job pin → approved user revision → deployment file override → packaged default.
4. `TLDW_SERVICE_PROMPTS_MODE` accepts `enabled`, `read_only`, or `bypass_stored_overrides`; unknown values fail startup. The initial default is `read_only`.
5. A part is limited to 64 KiB UTF-8 and a complete definition to 256 KiB. The parser accepts only declared `{variable}` placeholders, supports `{{` and `}}` as escaped literal braces, and rejects other template syntax.
6. Save creates one pending bundle revision. A newer save explicitly supersedes the prior pending revision. Activation requires a signed Context Integrity manifest update even in single-user mode.
7. Store immutable state events, per-definition generations, an O(1) per-user catalog generation, and client-mutation receipts. Keep active and pending revisions plus the newest 50 historical revisions and 200 state events per definition; retain mutation receipts for 24 hours up to 1,000 per account; never prune active or pending rows.
8. Protected prompt components and pin sets are stored beside Jobs data, not in user content databases. Every component manifest, pin-set envelope, and binding has an externally keyed authenticator; Jobs payload encryption remains optional and independent.
9. Prompt-bearing jobs are created in `held`, bound to an authenticated full-bundle pin set, then changed to `queued` by one compare-and-swap transaction. Workers never execute unbound or unverifiable jobs.
10. Unbound held jobs and pin sets expire after one hour. Terminal bindings are retained for the greater of 30 days or the configured Jobs retention. Verification keys remain available for at least the maximum retained-job lifetime. Unreferenced snapshots are garbage-collected; protected prompt data is capped at 256 MiB per user and enqueue fails closed when the cap cannot be met without deleting live data.
11. Private account archives include user-authored revisions and non-operative state-history provenance only. Import creates `unapproved_import` history rows/events, leaves active and pending pointers null, and deterministically orders imported history by archive timestamp plus archive row ID.
12. Generic route availability is detected from OpenAPI (`hasServicePrompts`). The authenticated feature endpoint owns `mode`, `availability`, `contract_version`, and `can_approve_pending`; user authorization is never cached in unauthenticated docs-info.

## Dependency order

| Order | Plan | Delivers | Depends on |
|---|---|---|---|
| 1 | [Inventory and rollout matrix](2026-07-12-service-prompts-01-inventory.md) | Complete eligible/excluded prompt matrix, stable IDs, domain migration worklist | Approved spec |
| 2 | [Context Integrity approval infrastructure](2026-07-12-service-prompts-02-context-integrity-approval.md) | Signed-manifest mutation, anti-rollback anchoring, and review primitives | Inventory identity rules |
| 3 | [Registry and resolver](2026-07-12-service-prompts-03-registry-resolver.md) | Typed contracts, strict parser, resolution/provenance, strict deployment overrides | Inventory + signed-manifest verifier interface |
| 4 | [Persistence, approval API, and private backup](2026-07-12-service-prompts-04-persistence-api-backup.md) | Per-user revisions/state, Context Integrity review decisions, user API, deterministic archive behavior | Registry + manifest infrastructure |
| 5 | [Protected Jobs pinning](2026-07-12-service-prompts-05-protected-job-pinning.md) | Authenticated snapshots, held/bind/queue lifecycle, worker guard | Resolver + Jobs |
| 6 | [Shared settings UI](2026-07-12-service-prompts-06-shared-settings-ui.md) | WebUI/extension page, capability gating, sidepanel deep link, E2E | API complete; real-server E2E waits for the separately planned canary domain |

The inventory plan writes the authoritative migration worklist at `Docs/Design/service-prompt-inventory.md`. Each domain migration must then receive its own Backlog task and execution plan naming exact call sites from that matrix. Do not batch all domains into one code review.

## Mandatory per-commit gate

Every commit named in every child plan occurs only after its focused red→green tests pass and the existing-suite gate is green:

- Always run `source .venv/bin/activate && python -m pytest -v` from the repository root.
- For a commit touching `apps/`, also run `cd apps/tldw-frontend && bun run test:run && bunx vitest run -c vitest.extension.config.ts`, then `cd ../packages/ui && bun run test`. The extension command is explicitly non-watch.
- Run the touched-scope formatter/linter/type/build checks named by the child plan before committing.
- Do not commit around an unrelated or environment failure. Record it in the Backlog task, diagnose it under the three-attempt rule, and stop if the full gate cannot be made green.

Focused commands in child plans are the fast TDD loop; they supplement this gate and never replace it.

## Release slices

- Slice A: plans 1–3, with `read_only` catalog/preview capability only.
- Slice B: plan 4, enabling pending saves, approvals, history, reset, restore, and private backup semantics.
- Slice C: plan 5, then the separately planned first async domain migration from the approved inventory; no async domain may migrate earlier.
- Slice D: plan 6, then opt-in `enabled` deployments.
- Slice E: remaining domain plans in this order: summarization/media/audio; documents/web; RAG generation; reports/digests/watchlists/outputs; eligible extraction/chunking.

The route remains `availability: experimental` until at least one end-to-end domain exists. It may report `general` only when every inventory-approved broad content-facing domain has migrated, all remaining candidates are explicitly locked/deferred, and no-override provider-message goldens are byte-equivalent.

## Cross-plan verification gate

- [ ] Activate the project environment before Python tooling: `source .venv/bin/activate`.
- [ ] Run every focused command listed in the six child plans.
- [ ] Run the affected backend suites together: `python -m pytest -q tldw_Server_API/tests/Context_Integrity tldw_Server_API/tests/Service_Prompts tldw_Server_API/tests/Jobs/test_service_prompt_pinning_sqlite.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py tldw_Server_API/tests/Chatbooks/test_chatbooks_service_prompts_private_backup.py`.
- [ ] Run PostgreSQL prompt-pinning coverage through the standard AuthNZ/Jobs fixture: `python -m pytest -q tldw_Server_API/tests/Jobs/test_service_prompt_pinning_postgres.py`.
- [ ] Run frontend focused tests from `apps/tldw-frontend`: `bunx vitest run ../packages/ui/src/services/__tests__/service-prompts.test.ts ../packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx ../packages/ui/src/routes/__tests__/service-prompts-route.test.tsx ../packages/ui/src/components/Sidepanel/Settings/__tests__/service-prompts-link.test.tsx`.
- [ ] Run the real settings workflow: `bunx playwright test e2e/workflows/service-prompts-settings.spec.ts --reporter=line`.
- [ ] Run formatting/lint checks on touched code and `git diff --check`.
- [ ] Run Bandit on every touched Python scope: `python -m bandit -r tldw_Server_API/app/core/Context_Integrity tldw_Server_API/app/core/Service_Prompts tldw_Server_API/app/core/Jobs/models.py tldw_Server_API/app/core/Jobs/migrations.py tldw_Server_API/app/core/Jobs/pg_migrations.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/Jobs/worker_sdk.py tldw_Server_API/app/core/Jobs/service_prompt_store.py tldw_Server_API/app/core/DB_Management/Prompts_DB.py tldw_Server_API/app/core/DB_Management/Service_Prompts_DB.py tldw_Server_API/app/core/Chatbooks tldw_Server_API/app/core/AuthNZ/permissions.py tldw_Server_API/app/services/startup_context_integrity.py tldw_Server_API/app/services/startup_service_prompts.py tldw_Server_API/app/main.py tldw_Server_API/app/api/v1/API_Deps/context_integrity_deps.py tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py tldw_Server_API/app/api/v1/endpoints/service_prompts.py tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py tldw_Server_API/app/api/v1/endpoints/admin/__init__.py tldw_Server_API/app/api/v1/schemas/service_prompt_schemas.py tldw_Server_API/app/api/v1/schemas/admin_schemas.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/app/api/v1/router_groups/minimal.py -f json -o /tmp/bandit_task_12113.json`.
- [ ] Review the Bandit JSON and fix every new finding in changed code.
- [ ] Update the relevant Backlog task with tests, Bandit result, files, and final summary before each commit.

## Stop conditions

- Stop if Context Integrity has no configured external MAC key; report capability as unavailable and do not accept saves, approvals, or prompt-bearing job enqueue.
- Stop a domain migration if the inventory cannot prove that its output is user-visible content or if changing it can bypass a policy/schema guarantee.
- Stop after three failed approaches to the same issue and record the attempts in the active Backlog task before reassessing.
- Do not remove legacy `load_prompt` behavior globally; strict failure applies only to registry-managed definitions.
