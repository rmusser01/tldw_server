# Persona/Buddy Stage 0 Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-backed current-state audit for the Persona/Buddy assistant runtime before any reliability/UX implementation work begins.

**Architecture:** This is a docs/reporting slice, not a runtime-code slice. The worker will inspect existing backend, frontend, tests, docs, and GitHub tracker state; record contract ownership and flow reliability in one audit report; then recommend Stage 1 reliability/UX issues that are justified by evidence.

**Tech Stack:** Markdown, GitHub CLI, `rg`, FastAPI/Python source inspection, React/TypeScript source inspection, existing pytest/Vitest/Playwright tests when practical.

---

## Scope

This plan implements Stage 0 from
`Docs/superpowers/specs/2026-05-10-persona-buddy-assistant-maturity-roadmap-design.md`.

Allowed changes:

- Create `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`
- Add screenshot or JSON evidence under `Docs/Reviews/assets/2026-05-10-persona-buddy-audit/` only if browser or API probes are actually run
- Update the Backlog task for the audit

Out of scope:

- Runtime code changes
- New tests or test rewrites
- New persona capabilities
- New MCP runtime triggers
- New renderer behavior
- Closing or rewriting `#635` before the audit preserves its useful references

## Files To Inspect

Backend and contracts:

- `tldw_Server_API/app/api/v1/endpoints/persona.py`
- `tldw_Server_API/app/api/v1/schemas/persona.py`
- `tldw_Server_API/app/core/Persona/README.md`
- `tldw_Server_API/app/core/Persona/session_manager.py`
- `tldw_Server_API/app/core/Persona/memory_integration.py`
- `tldw_Server_API/app/core/Persona/exemplar_prompt_assembly.py`
- `tldw_Server_API/app/core/Persona/buddy.py`
- `tldw_Server_API/app/core/Persona/visuals.py`
- `tldw_Server_API/app/core/Persona/visual_service.py`
- `tldw_Server_API/app/core/Persona/visual_library_service.py`
- `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py`
- `tldw_Server_API/Config_Files/mcp_modules.yaml`
- `tldw_Server_API/Config_Files/persona_archetypes/*.yaml`

Frontend Persona/Buddy surfaces:

- `apps/tldw-frontend/pages/persona.tsx`
- `apps/tldw-frontend/extension/routes/sidepanel-persona.tsx`
- `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- `apps/packages/ui/src/components/PersonaGarden/`
- `apps/packages/ui/src/components/Common/PersonaBuddy/`
- `apps/packages/ui/src/hooks/personaWakeDetector.ts`
- `apps/packages/ui/src/hooks/chat/personaServerChat.ts`
- `apps/packages/ui/src/services/persona-stream.ts`
- `apps/packages/ui/src/services/persona-visuals.ts`
- `apps/packages/ui/src/services/tldw/persona-setup-analytics.ts`
- `apps/packages/ui/src/store/persona-buddy-shell.ts`
- `apps/packages/ui/src/store/persona-visual-runtime.ts`
- `apps/packages/ui/src/types/persona-buddy.ts`
- `apps/packages/ui/src/types/persona-visuals.ts`

Existing tests and E2E:

- `tldw_Server_API/tests/Persona/`
- `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py`
- `tldw_Server_API/tests/Chat/test_persona_prompt_assembly.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_chat_persona_exemplars_integration.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_chat_persona_selector_prompt_assembly_perf.py`
- `tldw_Server_API/tests/VoiceAssistant/test_persona_voice_command_persistence.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visual_library_db.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py`
- `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`
- `apps/packages/ui/src/routes/__tests__/sidepanel-persona.command-handoff.test.tsx`
- `apps/packages/ui/src/components/PersonaGarden/__tests__/`
- `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/`
- `apps/packages/ui/src/hooks/__tests__/personaWakeDetector.test.ts`
- `apps/packages/ui/src/services/__tests__/persona-stream.test.ts`
- `apps/packages/ui/src/store/__tests__/persona-buddy-shell.test.ts`
- `apps/packages/ui/src/store/__tests__/persona-visual-runtime.test.ts`
- `apps/tldw-frontend/e2e/workflows/persona.spec.ts`
- `apps/tldw-frontend/e2e/workflows/persona-live.spec.ts`
- `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`

Docs and prior design context:

- `Docs/Design/Personas.md`
- `Docs/Design/Character_Chat.md`
- `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- `Docs/Code_Documentation/Persona_Visual_Packs.md`
- `Docs/Code_Documentation/Character_Chat.md`
- `Docs/API/Voice_Assistant.md`
- `Docs/User_Guides/WebUI_Extension/Persona_Live_Wake_Phrases.md`
- `Docs/Product/WebUI/Character_Chat_Terminology_Taxonomy_2026_05_09.md`
- `Docs/Design/2026-05-10-persona-visual-renderer-provider-adapter-evaluation.md`

## Report Template

The audit report must contain these sections:

1. Summary verdict
2. Tracker state checked on `YYYY-MM-DD`
3. Contract inventory table
4. Evidence table
5. Known-good flow checklist
6. Flow-by-flow findings
7. `#635` migration recommendation
8. Stage 1 issue recommendations
9. Explicit non-goals and VN/CYOA boundary
10. Verification commands and skipped checks

Contract inventory columns:

| Flow | Server contract | Client owner | Persisted state | Session/runtime state | MCP/tool surface | Tests | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |

Evidence table columns:

| Flow | Journey | Evidence files | API/runtime contracts | Existing tests | Issue links | Observed or inferred gap | Severity | Stage 1 recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Severity rubric:

- P0: Persona/Buddy flow is broken or blocks setup/live controls
- P1: Existing feature works inconsistently or has misleading diagnostics
- P2: Existing feature works but lacks clear copy, test coverage, or recovery affordance
- P3: Future enhancement or cleanup, not Stage 1 reliability work

## Task 1: Create Audit Report Skeleton And Tracker Snapshot

**Files:**

- Create: `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`
- Optional create: `Docs/Reviews/assets/2026-05-10-persona-buddy-audit/tracker-state.json`
- Update: Backlog task for the audit

- [ ] Create the audit report with the required sections and empty tables.
- [ ] Re-verify GitHub tracker state before making recommendations.

Run:

```bash
gh issue view 635 --repo rmusser01/tldw_server --json number,title,state,url,body,comments,updatedAt
gh issue view 1388 --repo rmusser01/tldw_server --json number,title,state,url,updatedAt
gh issue view 1389 --repo rmusser01/tldw_server --json number,title,state,url,updatedAt
gh issue view 1428 --repo rmusser01/tldw_server --json number,title,state,url,updatedAt
gh issue view 1449 --repo rmusser01/tldw_server --json number,title,state,url,updatedAt
gh issue view 1497 --repo rmusser01/tldw_server --json number,title,state,url,updatedAt
gh issue view 1391 --repo rmusser01/tldw_server --json number,title,state,url,updatedAt
```

Expected:

- `#635` state and useful body/comment links are recorded.
- Closed visual/runtime tracker issues are confirmed or drift is called out.
- No GitHub issue is edited in this task.

- [ ] Extract useful `#635` links/comments into the report before recommending any rewrite or close action.
- [ ] Record whether the audit is working from live GitHub state or from an unavailable/offline snapshot.
- [ ] Commit only after the full audit is complete, not after the skeleton.

## Task 2: Inventory Backend Contracts

**Files:**

- Read: backend and contract files listed above
- Modify: `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`

- [ ] Identify Persona profile, state docs, session, voice/wake, visual-pack, and buddy summary persistence boundaries.
- [ ] Identify `/api/v1/persona` REST endpoints relevant to setup, profiles, sessions, visuals, buddy, analytics, command tests, and import/export.
- [ ] Identify `/api/v1/persona/stream` websocket frame types, including `user_message`, live notices, tool-plan/tool-call/tool-result frames, `wake_activation`, and `wake_deactivation`.
- [ ] Identify server-owned versus client-derived live state.
- [ ] Identify MCP persona tool contracts for `persona_visuals.capabilities`, `persona_visuals.library_items`, `persona_visuals.trigger_state`, `persona_visuals.create_draft_pack`, `persona_visuals.update_manifest`, `persona_visuals.use_library_item`, and `persona_visuals.enqueue_generation`.
- [ ] Populate the contract inventory table for backend-owned flows.
- [ ] Add evidence rows for any missing diagnostics, ambiguous ownership, or brittle persistence/runtime seams.

Useful commands:

```bash
rg -n "wake_activation|wake_deactivation|WAKE_ACTIVATION|voice_chat_trigger_phrases|wake_behavior|persona_visuals|websocket|WebSocket|notice|tool_plan|tool_call|tool_result" tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/core/Persona tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py
rg -n "CREATE TABLE|persona_visual|persona_profile|persona_session|wake_behavior|voice_chat_trigger_phrases" tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py
```

Expected:

- The report says which contracts are persisted, which are runtime-only, and which are MCP-triggered.
- Any Stage 1 candidate from backend inspection is limited to diagnostics/recovery/copy/test hardening.

## Task 3: Inventory Frontend Surfaces

**Files:**

- Read: frontend files listed above
- Modify: `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`

- [ ] Map Persona Garden setup/profile/defaults/policies/visuals/live-session surfaces.
- [ ] Map Persona Live route ownership in `sidepanel-persona.tsx`, including connection state, setup detours, handoff cards, and wake controls.
- [ ] Map Persona Buddy shell state resolution, selected persona precedence, active-pack loading, visual fallback, popover actions, and dormant/no-buddy states.
- [ ] Map wake detector behavior and browser/extension limitations.
- [ ] Map Persona Chat entry points and persona-backed chat hooks without turning character-chat or VN/CYOA flows into the Persona/Buddy runtime.
- [ ] Populate frontend rows in the contract inventory and evidence tables.
- [ ] Add known-good flow checklist rows for setup, chat, live voice, wake, Buddy display, visual fallback, MCP runtime trigger, and recovery.

Useful commands:

```bash
rg -n "wake|Wake|personaId|selectedPersona|Buddy|visual|Visual|live|Live|setup|handoff|connection|runtime|stream" apps/packages/ui/src/routes/sidepanel-persona.tsx apps/packages/ui/src/components/PersonaGarden apps/packages/ui/src/components/Common/PersonaBuddy apps/packages/ui/src/hooks apps/packages/ui/src/services apps/packages/ui/src/store
rg -n "persona|Persona|character|Character|buddy|Buddy|wake|Wake" apps/tldw-frontend/pages/persona.tsx apps/tldw-frontend/extension/routes/sidepanel-persona.tsx apps/tldw-frontend/e2e/workflows
```

Expected:

- The report distinguishes first-time setup, returning user, live-session, and recovery journeys.
- The report cites concrete files for each frontend claim.

## Task 4: Inventory Tests, Docs, And Existing Coverage

**Files:**

- Read: test and docs files listed above
- Modify: `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`

- [ ] Map existing backend tests to each audited flow.
- [ ] Map existing frontend unit tests and E2E workflows to each audited flow.
- [ ] Identify flows with tests but weak product copy or diagnostics.
- [ ] Identify flows with UI affordances but no clear backend/runtime contract.
- [ ] Identify flows that appear documented but are not protected by tests.
- [ ] Populate smoke/E2E candidate list for Stage 1 reliability coverage.

Optional verification commands if the local environment is ready:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_ws.py tldw_Server_API/tests/Persona/test_persona_buddy_api.py tldw_Server_API/tests/Persona/test_persona_voice_commands_api.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q
```

```bash
cd apps/packages/ui
bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx src/hooks/__tests__/personaWakeDetector.test.ts src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/services/__tests__/persona-stream.test.ts
```

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/persona.spec.ts e2e/workflows/persona-live.spec.ts --reporter=line
```

Expected:

- If commands are run, exact pass/fail/skipped results are recorded.
- If commands are not run, the report explains the environment blocker.
- Test failures are recorded as evidence; do not fix them in this task.

## Task 5: Synthesize Stage 1 Recommendations

**Files:**

- Modify: `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`

- [ ] Create a concise summary verdict that separates existing capability from reliability/UX gaps.
- [ ] Write `#635` migration recommendation that preserves useful links/comments and explains whether to keep, rewrite, or supersede the issue.
- [ ] Recommend Stage 1 child issues only where the evidence table supports reliability diagnostics, recovery, copy, or existing-flow test coverage.
- [ ] Mark non-Stage-1 items as Stage 2, Stage 3, Stage 4, or out of scope.
- [ ] Keep `#1391` VN/CYOA compatibility-only unless the audit finds a concrete cross-surface boundary issue.

Recommended Stage 1 issue format:

```markdown
### Candidate Issue: <title>

Evidence:
- <file/test/doc/issue reference>

Problem:
- <specific observed gap>

Reliability/UX scope:
- <diagnostic/recovery/copy/test-only fix>

Out of scope:
- <new capability/runtime expansion excluded>
```

Expected:

- Recommendations are narrow enough for focused PRs.
- No candidate issue depends on unverified assumptions.

## Task 6: Verification And Closeout

**Files:**

- Modify: `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`
- Update: Backlog task for the audit

- [ ] Run `git diff --check`.
- [ ] Run targeted `rg` checks to ensure required sections exist.

Run:

```bash
git diff --check
rg -n "Contract inventory|Evidence table|Known-good flow checklist|#635 migration|Stage 1 issue recommendations|VN/CYOA" Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md
```

- [ ] Document Bandit as not applicable if the audit remains docs/task-only.
- [ ] Update the Backlog task with completed acceptance criteria, verification, skips, and final summary.
- [ ] Commit the audit report and Backlog task.

Commit message:

```bash
git add Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md backlog/tasks/<task-file>
git commit -m "docs: audit persona buddy current state"
```

## Review Note

The writing-plans skill normally requests a plan-document-reviewer subagent.
This session may only spawn subagents when the user explicitly asks for
delegation. If the user asks for review before execution, dispatch a focused
plan-document-reviewer with this plan path and the roadmap spec path only.
