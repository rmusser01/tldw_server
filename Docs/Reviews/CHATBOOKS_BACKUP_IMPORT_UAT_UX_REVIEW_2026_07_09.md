# Chatbooks Backup And Import UAT UX Review

Date: 2026-07-09
Backlog task: TASK-12095
Design spec: `Docs/superpowers/specs/2026-07-09-chatbooks-backup-import-uat-ux-design.md`
Audience lens: Senior UX/HCI review using cognitive walkthrough, task analysis,
NN/g heuristic evaluation, severity scoring, and lightweight executable
verification.

## Executive Verdict

Backup and import are technically present, but the current WebUI and browser
extension flows are not straightforward or easy for a normal acceptance test.

Verdict by surface:

| Surface | Possible | Straightforward | Easy | Verdict |
|---|---:|---:|---:|---|
| Main Chatbooks page, selective export | Yes | Mostly | No | Works when the user already knows to select or include IDs. |
| Main Chatbooks page, backup everything | Partly | No | No | UI blocks the documented empty-selection backup path. |
| Main Chatbooks page, archive restore | Partly | No | No | Default `Import media` sends an unsupported option and is rejected. |
| Settings Chatbooks shortcut | Partly | No | No | Export is conversation-ID-only; import has the same unsupported media default. |
| Browser extension `/chatbooks` | Partly | No | No | Reuses the same main Chatbooks component, so the same risks apply. |
| Browser extension `/settings/chatbooks` | Partly | No | No | Reuses the same Settings shortcut. |
| OpenWebUI JSON/DB import | Yes | Mostly | No | Preview/user-selection guardrails exist, but hydration is a separate expert step. |
| OpenWebUI attachment hydration | Yes | No | No | Requires server-local paths, allowed-root setup, and remembered tldw conversation IDs. |

Short answer: **possible, not straightforward, not easy** for backup/restore.
OpenWebUI import is closer to acceptance-ready, but attachment hydration remains
high-cognitive-load.

## Method

The review used:

- Cognitive walkthrough: can users find the next action, understand it, and see
  the result?
- NN/g heuristic evaluation: 10 usability heuristics scored 0-4.
- Severity scoring: frequency, impact, persistence, and data-trust risk.
- Static code/docs inspection first, per the approved low-cost review path.
- Targeted executable verification for the OpenWebUI import/hydration guardrails.

References:

- NN/g 10 usability heuristics:
  https://www.nngroup.com/articles/ten-usability-heuristics/
- NN/g heuristic evaluation:
  https://www.nngroup.com/articles/how-to-conduct-a-heuristic-evaluation/
- NN/g severity ratings:
  https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/
- NN/g cognitive walkthroughs:
  https://www.nngroup.com/articles/cognitive-walkthroughs/

## Environment And Evidence

Evidence sources:

- Main Chatbooks UI:
  `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- Settings Chatbooks shortcut:
  `apps/packages/ui/src/components/Option/Settings/chatbooks.tsx`
- Extension routing:
  `apps/tldw-frontend/extension/routes/option-chatbooks-playground.tsx`
  and `apps/tldw-frontend/extension/routes/route-registry.tsx`
- API/schema/service:
  `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`,
  `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`,
  `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- User/API docs:
  `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`,
  `Docs/API-related/Chatbook_API_Documentation.md`
- Existing tests:
  `apps/extension/tests/e2e/chatbooks-export-download.spec.ts`,
  `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts`,
  `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx`

Executable verification:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
```

Result: **7 tests passed**. The first attempt with `--reporter=line` failed
before tests started because this Vitest version tried to load `line` as a
custom reporter, so the test was rerun with the default reporter.

Live browser UAT was not started because local API/frontend servers were not
running (`127.0.0.1:8000` health check failed, ports 3000 and 8080 were not
listening), and the primary P0 failures are statically provable from UI,
schema, endpoint, service, and docs. Starting full live UAT would add cost
without changing the acceptance verdict for those blockers.

## UAT Matrix

| Task | Entry Point | Result | Evidence | Acceptance |
|---|---|---|---|---|
| Quick backup everything using the user guide | Main Chatbooks | Fail | Guide says leave selections empty, but UI refuses empty selections and backend exports only selected types. | Not acceptable |
| Backup everything by toggling include-all per type | Main Chatbooks | Partial pass | Include-all controls exist, but all are initially false and require per-type action. | Possible, not easy |
| Selective export | Main Chatbooks | Pass with caveats | UI collects per-type IDs and submits export jobs. Existing extension E2E validates one export/download path. | Acceptable for power users |
| Download completed backup | Main Chatbooks / Extension | Pass | Job table exposes Download for completed jobs; extension E2E waits for a completed job and download. | Acceptable |
| Import a normal `.chatbook` archive with defaults | Main Chatbooks | Fail | UI defaults `importMedia=true`; endpoint rejects `import_media=true`. | Not acceptable |
| Import a normal `.chatbook` archive after disabling media | Main Chatbooks | Likely pass | Service imports all manifest items when selections are omitted. | Possible, not straightforward |
| Export from Settings | Settings Chatbooks | Narrow pass | Requires manual conversation IDs and exports only `conversation`. | Not a backup flow |
| Import from Settings with defaults | Settings Chatbooks | Fail | Same unsupported `import_media=true` default. | Not acceptable |
| Import from Settings after disabling media | Settings Chatbooks | Partial pass | Upload accepts `.zip` only, not `.chatbook`, and has no preview. | Possible, not straightforward |
| OpenWebUI JSON import | Main Chatbooks | Pass | Source selector, preview path, archive-only controls hidden, unit test coverage passes. | Mostly acceptable |
| OpenWebUI DB import | Main Chatbooks | Pass with caveats | Preview requires a selected source user and shows destination namespace. | Mostly acceptable |
| OpenWebUI attachment hydration | Main Chatbooks | Partial pass | Preview gate exists, but user must supply server-local root and imported conversation IDs. | Possible, not straightforward |
| Browser extension backup/import | Extension options | Same as WebUI | `/chatbooks` reuses main component; `/settings/chatbooks` reuses Settings shortcut. | Same issues as WebUI |

## Heuristic Scores

Scores cover the backup/import acceptance journey, not the entire Chatbooks
module.

| # | Heuristic | Score | Key Issue |
|---|---|---:|---|
| 1 | Visibility of System Status | 3 | Job tracker, progress, preview, and download affordances exist. |
| 2 | Match Between System And Real World | 1 | "Backup everything" does not match the implemented export-selection model. |
| 3 | User Control And Freedom | 2 | Jobs can be cancelled/removed and conflicts selected, but restore defaults are unsafe. |
| 4 | Consistency And Standards | 1 | Docs, Settings, main UI, and API defaults contradict each other. |
| 5 | Error Prevention | 1 | Default import options trigger backend rejection. |
| 6 | Recognition Rather Than Recall | 1 | Backup requires knowing include-all per type; hydration requires remembered IDs. |
| 7 | Flexibility And Efficiency | 2 | Power-user controls exist, but they complicate first-use backup/restore. |
| 8 | Aesthetic And Minimalist Design | 2 | Many controls appear at once; workflow lacks a primary "Backup all" path. |
| 9 | Error Recovery | 2 | Errors surface through notifications/alerts, but fixes are not made obvious. |
| 10 | Help And Documentation | 1 | Docs contain stale/nonfunctional backup instructions. |
| **Total** |  | **16/40** | **Poor for backup/import acceptance.** |

## Cognitive Load

Backup/restore flow: **5 failed checklist items out of 8**, high cognitive load.

Failures:

- Single focus: export page mixes naming, metadata, media, embeddings, generated
  content, and per-type selection before the primary backup task is clear.
- One thing at a time: users must configure backup scope and content-type
  mechanics simultaneously.
- Minimal choices: export/import expose many switches, selectors, tabs, and
  content cards before a first backup/restore path is established.
- Working memory: hydration requires conversation IDs from an earlier import.
- Progressive disclosure: OpenWebUI details are better gated, but normal
  backup/restore still exposes advanced settings too early.

What works:

- Import preview is automatic after file selection.
- OpenWebUI DB import forces source-user selection.
- Hydration requires a fresh preview before job creation.

## Findings

### P0 / Severity 4: Documented Backup-All Flow Does Not Work

Evidence:

- The user guide instructs users to leave content selections empty to export
  everything: `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md:62-71`.
- The API documentation gives a pre-migration backup example with
  `"content_selections": {}`:
  `Docs/API-related/Chatbook_API_Documentation.md:1014-1020`.
- The main UI initializes every export include-all switch to false:
  `ChatbooksPlaygroundPage.tsx:1076-1081`.
- The main UI rejects empty export selections:
  `ChatbooksPlaygroundPage.tsx:1479-1491`.
- The endpoint forwards only the provided `content_selections` entries:
  `chatbooks.py:433-447`.
- The service collects content only for content types present in
  `content_selections`: `chatbook_service.py:1736-1807`.

Why it matters:

A user following the "Complete Backup" instructions either cannot submit the
backup in the UI or produces a semantically empty API export path. This is a
direct acceptance failure for backup.

NN/g heuristic impact:

- Match between system and real world.
- Consistency and standards.
- Error prevention.
- Help and documentation.

Recommended fix:

Make "Backup all supported data" a first-class action. Use one consistent
contract everywhere:

- Preferred UX: a primary **Backup all** mode that visibly includes all
  supported content types and explains unsupported media/binary limitations.
- API/docs: either make empty/null selections genuinely mean all supported
  content, or stop documenting that behavior and show the exact all-selection
  contract.
- Main UI and extension: default the first-run backup path to the all-backup
  mode, with advanced selective export behind a secondary path.

### P0 / Severity 4: Archive Restore Fails By Default Because Import Media Is Unsupported

Evidence:

- Main Chatbooks import defaults `importMedia` to true:
  `ChatbooksPlaygroundPage.tsx:1095`.
- Main Chatbooks sends `import_media` for archive imports:
  `ChatbooksPlaygroundPage.tsx:1654-1655`.
- Main Chatbooks shows the import media switch for non-OpenWebUI imports:
  `ChatbooksPlaygroundPage.tsx:2574-2584`.
- Settings import also defaults `chatbookImportMedia` to true:
  `chatbooks.tsx:31`.
- Settings sends `import_media: chatbookImportMedia`:
  `chatbooks.tsx:163-168`.
- The schema default and example say `import_media=false`:
  `chatbook_schemas.py:163-177`.
- The endpoint rejects media/embedding imports:
  `chatbooks.py:717-722`.
- The user guide FAQ tells users to keep `import_media=false`:
  `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md:505-506`.

Why it matters:

The default restore path sends an unsupported option. Users do not need to make
a mistake to fail. That violates error prevention and destroys trust in a
backup/restore workflow.

NN/g heuristic impact:

- Error prevention.
- Recognition rather than recall.
- Consistency and standards.
- Error recovery.

Recommended fix:

- Change both main and Settings import defaults to `false`.
- Disable or hide unsupported media/embedding import toggles until implemented.
- If visible, label them as unavailable with short copy: "Media import is not
  supported yet."
- Add a unit/E2E check that default archive import submits
  `import_media=false` and `import_embeddings=false`.

### P1 / Severity 3: Settings Chatbooks Looks Like Backup/Restore But Only Handles Manual Conversation IDs

Evidence:

- Settings export requires at least one manual conversation ID:
  `chatbooks.tsx:80-90`.
- Settings export builds `content_selections` with only `conversation`:
  `chatbooks.tsx:95-107`.
- Settings import accepts `.zip` only:
  `chatbooks.tsx:450-454`.
- Settings still exposes import media and embeddings toggles that the backend
  rejects: `chatbooks.tsx:414-429`.

Why it matters:

Settings is a likely first stop for backup/restore. It currently behaves like a
thin developer shortcut, not a user-safe backup surface. A user trying to back
up "my data" has to know internal conversation IDs and cannot see that notes,
characters, prompts, media metadata, and other content are excluded.

NN/g heuristic impact:

- Match between system and real world.
- Recognition rather than recall.
- Aesthetic and minimalist design.
- Help and documentation.

Recommended fix:

- Rename this panel to **Conversation chatbook shortcut** or replace it with a
  link/card to the full Backup & Import page.
- If kept, state plainly: "Exports selected conversations only."
- Accept `.chatbook,.zip` for archive restore if the backend accepts ZIP-format
  archives.
- Remove unsupported toggles or default them off and disabled.

### P1 / Severity 3: OpenWebUI Hydration Requires A Memory Bridge

Evidence:

- The hydration payload is built from manually entered conversation IDs:
  `ChatbooksPlaygroundPage.tsx:1197-1211`.
- The UI asks users to paste imported tldw conversation IDs:
  `ChatbooksPlaygroundPage.tsx:2292-2311`.
- The user guide instructs users to enter imported conversation IDs after import:
  `Chatbook_User_Guide.md:214-218`.
- Unit tests verify the preview/run guardrails but still provide IDs manually:
  `ChatbooksPlaygroundPage.openwebui-import.test.tsx:415-438`.

Why it matters:

OpenWebUI hydration is a follow-up recovery step after migration. Asking users
to remember or discover imported tldw conversation IDs creates a working-memory
failure at the exact moment they are trying to verify data completeness.

NN/g heuristic impact:

- Recognition rather than recall.
- Visibility of system status.
- User control and freedom.
- Error recovery.

Recommended fix:

- Carry imported conversation IDs from the import response/job result into the
  hydration panel.
- Add a **Use last OpenWebUI import** action when a recent import job is present.
- Show a compact import result summary with copied scope: conversations,
  source user, attachment references, and whether hydration is needed.
- Keep the current fresh-preview gate. It is good error prevention.

### P1 / Severity 3: Product Language Still Says "Playground" For A Data-Safety Workflow

Evidence:

- The main page heading defaults to "Chatbooks Playground":
  `ChatbooksPlaygroundPage.tsx:2873`.
- The page object and extension E2E expect "Chatbooks Playground":
  `ChatbooksPage.ts:23`, `chatbooks-export-download.spec.ts:88-90`.
- The feature is described in docs as backup/restore and migration, not a
  playground.

Why it matters:

"Playground" is acceptable for experiments. It is poor information scent for
backup, restore, migration, and data-safety operations. This weakens trust and
makes users less confident that the page is the right place for a serious
backup.

NN/g heuristic impact:

- Match between system and real world.
- Help and documentation.
- Aesthetic and minimalist design.

Recommended fix:

Rename the visible page to **Chatbooks Backup & Import** or **Backup & Import**.
The route/component name can stay internal if changing it creates churn.

### P2 / Severity 2: Existing Tests Prove Mechanics, Not Acceptance

Evidence:

- Extension E2E validates one selective export/download path by toggling
  include-all for Prompts and downloading the resulting ZIP:
  `chatbooks-export-download.spec.ts:100-169`.
- WebUI tier-2 test treats "export may require content selection" as acceptable
  if the API call does not fire:
  `chatbooks.spec.ts:134-146`.
- OpenWebUI import unit tests are stronger and cover preview, selected user,
  stale preview invalidation, hydration preview-before-run, and unavailable
  capability states.

Why it matters:

The automated suite can pass while the user-facing backup-all and default
restore flows fail. This leaves the highest-value UAT scenarios unprotected.

Recommended fix:

Add focused UAT tests:

- Main WebUI: Backup all supported data creates an export job with non-empty
  content selections or a documented all-selection contract.
- Main WebUI: Default archive import sends unsupported flags as false.
- Settings: Default import sends unsupported flags as false.
- Settings: Label/behavior makes conversation-only export explicit.
- Extension: `/chatbooks` backup-all and archive import mirror WebUI behavior.
- OpenWebUI: After import, hydration can use imported conversation IDs without
  manual copy/paste.

## Minimal Remediation Plan

1. **Fix import defaults first.**
   Set archive import media and embeddings to false in both main Chatbooks and
   Settings. Disable or hide unsupported toggles. This is the smallest change
   that unblocks restore.

2. **Create one honest backup-all path.**
   Add a primary **Backup all** action or mode to the main Chatbooks page and
   extension route. It should preselect all supported content types, show
   unsupported limitations, and create a job without requiring users to toggle
   each content card.

3. **Align docs/API/UI semantics.**
   Decide whether empty/null selections mean all. Make the implementation,
   OpenAPI examples, user guide, and UI copy all say the same thing.

4. **Demote or clarify Settings.**
   Either make Settings a small entry point to the full Backup & Import page or
   label it as a conversation-only shortcut. Do not let it masquerade as a full
   backup tool.

5. **Reduce OpenWebUI hydration recall.**
   Auto-fill hydration scope from the last import job or expose a "Use imported
   conversations from this job" action. Keep preview-before-run.

6. **Rename the visible page.**
   Replace "Chatbooks Playground" with "Chatbooks Backup & Import" or similar
   user-facing copy. Avoid a route rename unless needed.

7. **Add acceptance tests for the exact failures above.**
   Keep them narrow. The goal is to prevent backup-all and default restore from
   regressing, not to build a broad new harness.

## What Not To Do

- Do not build a new backup product surface before fixing the defaults and
  selection semantics.
- Do not add more explanatory text around a broken default.
- Do not make users learn IDs for routine backup/restore.
- Do not rely on the API being technically capable when the UI/docs path is
  contradictory.

## Final Acceptance Answer

Current state:

- **Backup from WebUI/extension:** possible for selective exports, not
  straightforward for complete backup, not easy.
- **Import/restore from WebUI/extension:** possible only if users disable media
  import, not straightforward, not easy.
- **OpenWebUI import:** possible and comparatively well guarded.
- **OpenWebUI hydration:** possible, not straightforward, not easy.

After the minimal remediation plan, the likely target state is:

- Backup/restore becomes straightforward for common cases.
- OpenWebUI migration remains advanced but can become acceptable if hydration
  scope is carried forward from import results.
