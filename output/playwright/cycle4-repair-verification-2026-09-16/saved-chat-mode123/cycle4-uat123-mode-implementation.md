# UAT123 ordinary saved-chat workflow repair

Task13260.63. Production: only apps/packages/ui/src/components/Option/Playground/Playground.tsx. Tests: existing coordinator integration and Notes backlink-label suite. No route parsing feature, preference rewrite, backend/transport, runtime/browser, staging or commit edits. Root owns plan/tracker and integration.

## Cause and bounded correction

A previous Character route stores the new-chat workflow preference as Character. Even when History or Note navigation loaded canonical ordinary conversation metadata and cleared selected assistant, Playground ORed that preference into the active workflow, displaying Character controls and applying its send blocker.

Once the session scope is ready and an existing serverChatId has resolved metadata, Playground now derives workflow from canonical assistant kind/character ID. Ordinary and persona chats use ordinary workflow; tracked Character chats remain Character. A fresh Character route or a different still-pending route target retains explicit route intent. Pending scope/metadata and actual unsaved chats keep the prior behavior. This is render-time derivation with no asynchronous mutation and no preference writes; existing owned/cancellable metadata loader remains the authority. The stored preference is retained for future new chats.

The earlier bare ordinary chatId URL observation was a harness-created unsupported route, not an application-produced link. No parsing support was added. Final tracker number123 is the stale workflow defect; earlier provisional123/124 mapping is superseded.

## Permanent RED and validation

Coordinator targeted RED:3 failures(history load, cold canonical ordinary restore, persona) with5 passing pending/fresh/unsaved/knownCharacter controls. /private/tmp/cycle4-uat123-mode-red.log. Initial fresh-entry control incorrectly expected no clearing for an intentional new entry; corrected to assert unsaved preservation only for the genuine unsaved control before final RED.

Actual Notes menu→real useSelectServerChat→mounted Playground RED:1 expected failure/24 unselected. /private/tmp/cycle4-uat123-note-red.log. It reads real canonical responses via mocked API boundary and verifies resulting workflow and Character send-blocker props; it does not stub the workflow calculation. Message rendering and composer presentation are lightweight doubles. The canonical history path uses the actual selected-assistant WebUI storage, real selection consumer and real canonical loader. Test assertions also verify the persisted Character preference remains unchanged.

Final full suites:49 coordinator +25 Notes =74 passing. /private/tmp/cycle4-uat123-mode-green.log. Includes existing delayed owner/picker/replacement cancellation, actual persisted Character route restoration, fresh entry, Notes text/image drafts, delayed reads, streaming and account guards. New controls cover metadata and session-scope pending, actual history, cold saved ordinary, knownCharacter, persona and unsaved draft. These are bounded mounted integration controls, not a claim of full application or native acceptance.

Exact command from apps/packages/ui:

    ./node_modules/.bin/vitest run src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage26.backlink-labels.test.tsx --maxWorkers=1 --no-file-parallelism

Static checks: scoped ESLint0 errors/16 exact unchanged warnings across3 files. Script /private/tmp/cycle4-uat123-mode-static-check.mjs compares against HEAD, outputs baseline/current/comparison JSON with same prefix. Scoped git diff --check clean. Bandit not applicable to this TypeScript-only slice. Parent will run combined compiler against its existing baseline after all concurrent slices freeze; no isolated compiler-pass claim.

## Freeze and limits

Exact paths/hashes including official task note are in /private/tmp/cycle4-uat123-mode-owned-manifest.json. No more source/test/task writes planned after this freeze. Independent review and targeted native Character→ordinary History/Note confirmation remain required. Earlier native111/121/117 passes are preserved separately and do not substitute for this repair's native verification.
