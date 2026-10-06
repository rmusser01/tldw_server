# /notes + /chat UX remediation: integration log (2026-10-04)

Thirteen open PRs from the /notes and /chat UX review (tracking #3101) were
merged together onto `origin/dev` on the branch
`integration/ux-remediation-2026-10`. The branch is a rehearsal: it is not
pushed and is not meant to be merged. This log records every conflict and how
it was resolved, every reproduction marker that had to change, and what else
the combination needed, so that each step can be replayed when the PRs are
merged into `dev` one at a time.

- Base: `origin/dev` at `75ab224081`. Later, `origin/dev` was merged in twice:
  `27ce976387` (merge 15) and `94854ca3db` (merge 16), so the branch reflects
  `dev` on 2026-10-06. None of the 13 PRs had landed on `dev` by then.
- Branch tips are the remote tips on 2026-10-04, except
  `feat/chat-idempotent-create-p3`, which gained #3198 afterwards and was merged
  a second time (merge 14).
- A resolution between two PRs applies to **whichever of the two merges
  second**. Section 2 lists which pairs conflict, so the log still works if the
  queue order changes.

## 0. Per-PR summary

One line per original PR, with the PRs stacked into it. Review ids are from
the 2026-10-02 /notes and /chat UX review (tracking #3101).

**Stage 0: harnesses and red-first reproductions**

- **#3137** Client/server contract reproductions for /notes and /chat (NL-01, NL-02, NL-03, CS-N2, CC-01).
  - **#3168** Loaded server chats keep their server message ids, so Save to Notes works again (XP-02).
  - **#3169** The Prompt picker lists the server prompt library (CC-05).
  - **#3173** Wikilinks accept `[[Title]]` and `[[id:UUID]]`: render, follow, backlinks, autocomplete (NE-02).
  - **#3190** After a rename, offer to update `[[Old title]]` links in other notes, with Undo (#3110).
- **#3139** Playground integration harness, with red-first chat reproductions (CS-01, CS-03, CS-05, CC-01, CC-04, CM-01, CM-N1).
  - **#3166** New chat and Clear reset the history selection, so the next send starts clean (CS-01, CS-05).
  - **#3167** A slow first token waits for the 120 s startup timeout (CM-N1).
- **#3136** Browser UX-regression harness (live backend, extension side panel), with notes reproductions (NL-01, NS-01).
  - **#3140** Browser reproductions for NS-N1, NE-04 and CS-02.
  - **#3141** Accessibility and request-budget ratchets for /notes and /chat (AX-01, AX-07, AX-13, AX-16, XP-16).
  - **#3146** Side-panel reproductions for XS-01, XS-07 and XP-08.

**Notes**

- **#3133** Hide the WYSIWYG input mode until NE-01 is fixed (D5).
  - **#3158** WYSIWYG typing reads forward, Heading and List work, and the toggle is back (NE-01).
- **#3148** The notes list pages with limit/offset, shows real totals and sorts on the server (NL-01).
  - **#3151** "Export matching notes" pages with limit/offset and can be cancelled (NL-02).
  - **#3156** Bulk "Add tags" merges into existing tags, with Undo and a themed picker (NL-03).
- **#3145** "Reload notes" after a save conflict loads the server's content together with its version (NS-N1).
  - **#3164** One save state machine: flush on leave, retry by status, a single conflict panel, one status pill (NS-01, NS-02, NS-03, NS-05, NS-06, NS-N2).

**Chat (WebUI and extension)**

- **#3147** Persistence labels report only saves the server acknowledged (CS-03 labels, XS-05).
  - **#3163** Interrupted and stopped replies are kept and marked; no empty server chats (CS-04, CS-N3).
- **#3152** The side panel opens a past chat in its own tab; Delete and Rename act on the real chat (XS-01, XS-07).
  - **#3161** Side-panel tabs refresh when their chat changed elsewhere; recent chats are listed without a search (XP-08, XS-06).
- **#3178** Chat history search matches message content and shows a snippet (CS-02).

**Chat (server)**

- **#3154** Moving a chat to Trash keeps its messages, so Restore brings them back (CS-N2).
- **#3171** Refuse stale "extend latest" admissions with `history_branch_changed` (D7 P1).
- **#3174** Idempotent chat create with a client-supplied id (D7 P3; ChaChaNotes schema SQLite v75, PostgreSQL v79).
  - **#3183** Native history and client chat ids for Sync v2 owners (D7 Q4).
  - **#3189** Deleting a chat row Sync v2 never saw no longer blocks the dataset (#3181, part of #3182).
  - **#3198** Clear the retired sync-v2 placeholder persona; device tombstones of unknown messages count as applied (#3182; schema SQLite v76, PostgreSQL v80).
- **#3175** Server-saved replies record generation metadata and keep partial replies (D7 P2, CM-04).
  - **#3186** `POST /chats/import` saves a local chat to the server losslessly (D7 P8).

Not in any PR, added on this branch: the conflict resolutions and follow-up
commits in sections 3 to 5 (S1 to S8, F1 to F5, B1, B2).

## 1. What was merged, in order

| # | Branch | PRs | Tip | Merge commit | Needed by hand |
|---|---|---|---|---|---|
| 1 | `test/ux-contract-reproductions` | #3137, #3173, #3190 | `358fb60d51` | `de55426237` | fingerprint (G1) |
| 2 | `test/playground-integration-harness` | #3139, #3166, #3167 | `afd1917b80` | `06e35b95ae` | nothing |
| 3 | `feat/ux-regression-harness` | #3136 | `2966f05be5` | `ad6fa9a3c7` | nothing |
| 4 | `fix/notes-hide-wysiwyg-d5` | #3133, #3158 | `4bfa40809a` | `2876a0a939` | R1 |
| 5 | `fix/notes-list-contract-nl01` | #3148 | `1591ed4b75` | `ad0042fd46` | fingerprint (G1) |
| 6 | `fix/notes-conflict-reload-nsn1` | #3145, #3164 | `1c6ee43ace` | `ed7c3c0f11` | R2, R3, R4, R5, S1 |
| 7 | `fix/chat-honest-persistence-labels` | #3147, #3163 | `79afbf2148` | `86cbd764ef` | R6 |
| 8 | `fix/sidepanel-tab-integrity` | #3152, #3161 | `217a16165b` | `531c0977d1` | nothing |
| 9 | `fix/chat-restore-trash-messages-csn2` | #3154 | `d5e6441535` | `7513791236` | fingerprint (G1) |
| 10 | `feat/chat-admission-leaf-check-p1` | #3171 | `25a636cfa5` | `bdcb8712ae` | fingerprint (G1) |
| 11 | `feat/chat-idempotent-create-p3` | #3174, #3183, #3189 | `13391f5ace` | `1e1d084ab9` | R7, R8, S3, fingerprint (G1) |
| 12 | `feat/chat-server-settled-metadata-p2` | #3175, #3186 | `793c6a02de` | `2557d903f1` | fingerprint (G1) |
| 13 | `fix/chat-history-content-search-cs02` | #3178 | `20546807e5` | `373aa74955` | R9, fingerprint (G1) |
| 14 | `feat/chat-idempotent-create-p3` again | #3198 | `f7e29b60e9` | `c141da468d` | nothing (schema check, section 6) |
| 15 | `origin/dev` | #3179, #3100, #3165, #3193, #3195 and others | `27ce976387` | `908ebb88db` | nothing textual; S6, S7, F5 |
| 16 | `origin/dev` | #3196, #3194, #3192, #3199, #3200, #3201, #3188 and others | `94854ca3db` | `0524591aeb` | R10, R11, fingerprint (G1) |

Commits after the merges:

| Commit | What |
|---|---|
| `13ce343b07` | S3: new test for the P1 leaf check on the Sync v2 path |
| `4f2e9a7904` | G1: OpenAPI fingerprint regenerated once |
| `d9359f9b08` | G2: privilege snapshot regenerated (dev's own drift, see G2) |
| `22b271b2e6` | F1: Vitest NL-01/NL-02/NL-03 flips |
| `2a72e7e5f1` | F2: pytest CS-N2 xfail removed |
| `3319f8af83` | F4: NE-01 spec uses the shared `warmBackendOnce` |
| `14ade63200` | F3: Playwright flips, S2 and the README table |
| `96cf49d714` | S4: Sync no-profile delete test follows CS-N2 |
| `3dd0f8a83f` | S5: side-panel search helpers accept CS-02's snippet |
| `d390f1ce92` | B1: notes-editor a11y baseline lowered to 4 (one early measurement; corrected by `5a9b4dd54b`) |
| `c9a494dc11` | S6: harness fake server returns the chat scope |
| `e1688a5017` | S7: five Playground tests follow a fresh chat to its server owner (#3195) |
| `9db2a72044` | F5: CS-03 `it.fails` becomes `it` (#3195) |
| `5a9b4dd54b` | B2: a11y spec waits for the open note to finish loading; baseline 8 |
| `43a74bcd44` | S8: two P3 Postgres tests that fail on #3174's own tip |
| `f871694cde` | G1: fingerprint regenerated after merge 16 |

## 2. Which PRs conflict with which

Textual conflicts only, computed pairwise with `git merge-tree`. The
fingerprint is left out: every pair of API-changing PRs conflicts on it (G1).

| Pair | File | Resolution |
|---|---|---|
| #3133 (NE-01) vs `dev` | `useNotesEditorState.tsx` | R1 |
| #3145 (NS) vs #3133 (NE-01) | `NotesEditorPane.tsx` | R2 |
| #3145 (NS) vs #3148 (NL-03) | `NotesManagerPage.tsx` | R3 |
| #3145 (NS) vs #3133 (NE-01) | `useNotesEditorState.tsx`, 3 hunks | R4 |
| #3145 (NS) vs #3137 (carries #3190) | `useNotesEditorState.tsx`, 6 hunks | R5 |
| #3147 (labels) vs #3139 (harness) | `harness/fake-tldw-server.ts`, `harness/playground-harness.tsx` | R6 |
| #3174 (P3) vs #3154 (CS-N2) | `.github/workflows/ci.yml`, 5 copies | R7 |
| #3174 (P3) vs #3171 (P1) | `character_messages.py` | R8 |
| #3178 (CS-02) vs #3174 (P3) | `chacha/conversation_store.py` | R9 |
| `dev` #3196 vs #3164 (NS-01) | `useNotesEditorState.tsx`, 3 hunks | R10 |
| `dev` #3188/#3201 vs #3136 (harness) | `tests/CI/test_license_first_workflow_contracts.py` | R11 |

Against `dev` alone, only two branches conflict: #3137 on the fingerprint and
#3133 on R1. R1 is a conflict with `dev` itself, so it also shows up between
#3133 and any branch whose base already has dev's pending-selection guard.

Conflicts in behaviour, with no textual conflict (section 4):

| Pair | What | Resolution |
|---|---|---|
| #3145/#3164 (NS) vs #3133 (NE-01) | NE-01 test clicks the removed "Reload" notice | S1 |
| #3164 (NS) vs #3136 (harness) | NS-N1 browser test waits for the removed toast | S2 |
| #3171 (P1) vs #3174/#3183 (P3) | leaf check on the Sync path needs a test | S3 |
| #3154 (CS-N2) vs #3189 (in P3) | no-profile delete test expects the old cascade | S4 |
| #3178 (CS-02) vs #3136 (harness) | side-panel helpers match the result name exactly | S5 |
| `dev` #3195 vs #3139 and #3147 (harness) | fake server omits the chat scope | S6 |
| `dev` #3195 vs #3139/#3166 and #3163 | tests assume a fresh chat is local | S7 |
| `dev` #3195 vs #3139 | CS-03 reproduction now holds | F5 |
| #3174 (P3) alone | two Postgres tests fail on its own tip | S8 |

## 3. Manual resolutions

### R1. #3133 (NE-01) vs `dev`: `useNotesEditorState.tsx`, `loadDetail`

File: `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`.

- **dev** (`2a7d1e64b8`, pending-selection guard) widened the `finally` block
  so a finished load clears `pendingSelectionEpochRef`.
- **#3133** added `replaceWysiwygHtml` to the callback's dependency array,
  because the uncontrolled WYSIWYG editor is rewritten through it.

Resolution: dev's `finally` block and #3133's dependency array.

```ts
    } finally {
      if (pendingSelectionEpochRef.current === noteEpoch) pendingSelectionEpochRef.current = null
      if (isCurrent()) setLoadingDetail(false)
    }
  }, [applyOfflineDraftToEditor, authorityScope, clearAssistUndoState, clearTaskState, isOnline, message, refreshTaskStateForNote, rememberRecentNote, replaceWysiwygHtml, setEditorKeywords, setIsDirty, setLoadingDetail, setSaveIndicator, setMonitoringNotice])
```

R4 changes this array again once #3145 is present.

### R2. #3145 (NS) vs #3133 (NE-01): `NotesEditorPane.tsx`, component body

File: `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`, just after
`unavailableLabel`.

- **#3145** replaced the status-line ref (`saveStatusRef`, keyed on
  `saveIndicatorText`) with the conflict panel's ref (`saveIssueRef`, keyed on
  `saveIssue`). `saveIndicatorText` no longer exists.
- **#3133** added, right below the old ref, the layout effect that writes the
  WYSIWYG document into the editor only on mount or on a new `wysiwygRevision`.

Resolution: #3145's two lines, then #3133's effect unchanged.

```tsx
  const saveIssueRef = React.useRef<HTMLDivElement | null>(null)
  const saveStatusDescriptionId = saveIssue ? NOTES_SAVE_STATUS_MESSAGE_ID : null

  // NE-01: React never owns the WYSIWYG editor's children (no
  // dangerouslySetInnerHTML), so re-renders caused by typing leave the DOM and
  // the caret alone. The document is written only into a newly mounted editor
  // node or when the hook publishes an external revision. No dependency array:
  // the editor can mount or remount on any render (mode, layout, loading).
  const appliedWysiwygRef = React.useRef<{ node: HTMLDivElement; revision: number } | null>(null)
  React.useLayoutEffect(() => {
    const node = richEditorRef.current
    if (!node) return
    const applied = appliedWysiwygRef.current
    if (applied && applied.node === node && applied.revision === wysiwygRevision) return
    replaceEditableHtml(node, wysiwygHtml)
    appliedWysiwygRef.current = { node, revision: wysiwygRevision }
  })
  const contentDescribedBy = joinAriaIds(NOTES_EDITOR_CONTENT_HELP_ID, saveStatusDescriptionId)
  const saveIssueKind = saveIssue?.kind ?? null
```

### R3. #3145 (NS) vs #3148 (NL-03): `NotesManagerPage.tsx`, above the component

File: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`.

- **#3148** added the bulk "Add tags" helpers above the component
  (`BulkTagUndoSnapshot`, `notePath`, `fetchNote`, `patchNoteTags`,
  `toBulkNoteLabel`, `formatBulkNoteLabels`, `formatNoteCount`,
  `formatTagSummary`, `hasInlineKeywords`).
- **#3145** replaced the component's inline props type with
  `NotesManagerPageProps`, which adds the `LeaveGuard` prop.

Resolution: keep all of #3148's helpers, then #3145's props type and
signature. Only the old one-line signature is dropped.

```tsx
const hasInlineKeywords = (note: unknown) => {
  const record = (note ?? {}) as { keywords?: unknown; metadata?: { keywords?: unknown } }
  return Array.isArray(record.keywords) || Array.isArray(record.metadata?.keywords)
}

type NotesManagerPageProps = {
  sourceNoteId?: string | null
  /**
   * Router-aware guard that holds in-app navigation while unsaved edits are
   * flushed (NS-01). The route supplies it (RouteLeaveGuard) so the page stays
   * independent of the router; without it only the unmount backstop applies.
   */
  LeaveGuard?: React.ComponentType<NotesLeaveGuardState>
}

const NotesManagerPage: React.FC<NotesManagerPageProps> = ({ sourceNoteId = null, LeaveGuard }) => {
```

### R4. #3145 (NS) vs #3133 (NE-01): `useNotesEditorState.tsx`, three dependency arrays

- **#3145** rewrote the three callbacks around the save machine, so their
  arrays now list `assignSelectedVersion`, `dispatchSave` and `setDirtyFlag`
  instead of `setIsDirty` and `setSaveIndicator`.
- **#3133** added `replaceWysiwygHtml` to the same three arrays.

Resolution: #3145's arrays with `replaceWysiwygHtml` added. The bodies merged
on their own: each still calls `replaceWysiwygHtml(...)`.

`applyOfflineDraftToEditor`:

```ts
  }, [assignSelectedVersion, dispatchSave, replaceWysiwygHtml, setDirtyFlag, setEditorKeywords, setMonitoringNotice])
```

`loadDetail` (dev's `finally` block from R1 is kept above it):

```ts
  }, [applyOfflineDraftToEditor, assignSelectedVersion, authorityScope, clearAssistUndoState, clearTaskState, dispatchSave, isOnline, message, refreshTaskStateForNote, rememberRecentNote, replaceWysiwygHtml, setDirtyFlag, setEditorKeywords, setLoadingDetail, setMonitoringNotice])
```

`resetEditor` (#3145's new `describeSaveFailure` callback follows it and is
taken whole):

```ts
  }, [assignSelectedVersion, clearAssistUndoState, clearTaskState, dispatchSave, replaceWysiwygHtml, setDirtyFlag, setEditorKeywords, setMonitoringNotice])
```

### R5. #3145 (NS) vs #3137 (carries #3190, the rename offer): `useNotesEditorState.tsx`, six hunks

What each side wanted:

- **#3190** tracks the open note's last saved title in `savedTitleRef` and
  calls `announceSavedTitle` when a save changes it, so the page can offer to
  update `[[Old title]]` links. It added `toNoteTitle`, set `savedTitleRef` in
  `loadDetail`, `resetEditor` and the create path, and wrapped the update
  request so the callback fires as soon as the server answers.
- **#3145** made the save read identity, version and text from refs taken at
  the start of the save (`noteId`, `baseVersion`, `snapshot`) instead of from
  the render's `selectedId` and `title`, keeps `selectedIdRef` in step by hand,
  and reordered the create path so the acknowledgement comes first.

Resolution: keep both, and make the rename offer use the save's own `noteId`
and `snapshot.title`. Using the render's `selectedId` and `title` there would
bring back the stale-render problem #3145 removed (a leave flush chained after
another save).

1. Module helpers after `noteResourcePath`: both blocks, #3190's `toNoteTitle`
   first, then #3145's `offlineDraftKeyFor`, `NOTES_OFFLINE_QUEUE_RETRY_MS`,
   `browserReportsOffline`, `ownerError`, `NotesSaveIssue`,
   `NotesLeaveGuardState` and `writeClipboardText`.

2. `loadDetail`, after `loadedTitle`:

   ```ts
         savedTitleRef.current = { noteId: String(id), title: String(d?.title || '') }
         selectedIdRef.current = id
         setSelectedId(id)
   ```

3. `resetEditor`, after `clearAssistUndoState()`:

   ```ts
       savedTitleRef.current = null
       selectedIdRef.current = null
       setSelectedId(null)
   ```

4. `saveNote`, create path. #3145's reordered block is taken, and #3190's
   line goes into it with `snapshot.title` in place of `title`. The old block
   further down (`setIsDirty(hasNewerEdits())` ... `setSelectedVersion`) is
   dropped: `acknowledge()` does that work now.

   ```ts
             if (created?.id != null) {
               savedTitleRef.current = { noteId: String(created.id), title: toNoteTitle(created) ?? snapshot.title }
               selectedIdRef.current = created.id
               setSelectedId(created.id)
               acknowledgeOfflineDraft(created.id, createdVersion)
             }
             if (createdVersion != null) assignSelectedVersion(createdVersion)
             if (createdLastSaved) setSelectedLastSavedAt(createdLastSaved)
             acknowledge()
             ...
             result = true
             // The page is going away: the acknowledgement is all the user needs.
             if (trigger === 'leave') return true
             await refetch()
   ```

5. `saveNote`, update path. #3190's two-argument `request(...)` with #3145's
   `noteId`, and `snapshot.title` in the callback:

   ```ts
             const previousSavedTitle =
               savedTitleRef.current?.noteId === String(noteId) ? savedTitleRef.current.title : null
             const updated = await request(
               {
                 path: noteResourcePath(noteId) as any,
                 method: 'PUT' as any,
                 headers: {
                   'Content-Type': 'application/json',
                   'expected-version': String(expectedVersion)
                 },
                 body: payload
               },
               // A rename is saved once the server answers, whatever the editor shows by then.
               // A blank title is not sent, so the server keeps the previous one.
               (saved) =>
                 announceSavedTitle(
                   noteId,
                   previousSavedTitle,
                   toNoteTitle(saved) ?? (snapshot.title.trim() || previousSavedTitle),
                   requestAuthorityScope
                 )
             )
   ```

   The callback lines sit just below the conflict hunk and merged without a
   marker, still reading `selectedId` and `title`. Change them by hand.

6. `saveNote` dependency array: both new entries.

   ```ts
       [
         announceSavedTitle,
         assignSelectedVersion,
         authorityScope,
   ```

The rest of #3190 merged on its own and was checked: the `onResponse`
parameter of `request`, the announcement in `syncOfflineDraft`, and the
`onNoteRenamed` wiring.

### R6. #3147 (labels, with #3163) vs #3139 (Playground harness): two harness files, add/add

Files under
`apps/packages/ui/src/components/Option/Playground/__tests__/harness/`:
`fake-tldw-server.ts` and `playground-harness.tsx`.

- **#3139** added the harness.
- **#3147** carries a cherry-pick of the same harness commit and #3163 extends
  it: streamed `chunks` with `pauseAfterChunks` / `resume` / `end`,
  `waitUnlessAborted`, `resetAppStores`, `simulatePageReload`,
  `keepBrowserState`, and resets for the turn registry and the save-status
  store.

Resolution: take #3147's version of both files. It is a strict superset:
`git diff 92b30045cc afd1917b80 -- harness/` is empty, so the harness branch
never changed these files after the commit that was cherry-picked. The other
harness files (`harness-i18n.tsx`, `memory-dexie.ts`,
`Playground.harness.integration.test.tsx`) are identical on both sides.

The harness branch's own tests (CS-01/CS-05 reset, CM-N1, CC-01, CC-04, CM-01,
CS-03) pass against the superset harness.

### R7. #3174 (P3) vs #3154 (CS-N2): `.github/workflows/ci.yml`, five shard copies

Both add one test file at the end of the `chat-character-integration-api`
path list, in each of the five copies of the shard matrix.

Resolution: both lines, in every copy.

```yaml
              tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_memory_endpoint.py
              tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_trash_restore.py
              tldw_Server_API/tests/Character_Chat_NEW/integration/test_chat_session_client_id_create.py
          - name: chat-character-integration-chat
```

`Helper_Scripts/ci/check_shard_coverage.py` reports `new_uncovered=0`.

### R8. #3174 (P3, with #3183) vs #3171 (P1): `character_messages.py`, `send_message`

File: `tldw_Server_API/app/api/v1/endpoints/character_messages.py`.

- **#3171** passed `history_branch=message_data.tldw_history_branch` to
  `db.append_selected_history_input` and added `**exc.details` to the 409, so
  `history_branch_changed` carries `leaf_ids`.
- **#3183** moved the admission and the settlement into a `history_write`
  closure, so the same write runs directly or inside
  `_publish_history_messages` for a Sync v2 owner, and added
  `except SyncStoreError`.

Resolution: #3183's structure, with #3171's argument inside the closure and
#3171's 409 body. One closure serves both paths, so the leaf check now applies
to Sync v2 owners too.

```python
                def history_write() -> tuple[str, dict[str, Any] | None]:
                    admission = db.append_selected_history_input(
                        chat_id, selection, data, owner_client_id=owner_id, owner_key=owner,
                        history_branch=message_data.tldw_history_branch)
                    return admission["input_message_id"], admission
```

```python
            try:
                if sync_service is None:
                    created_id, history_admission = await run_in_threadpool(history_write)
                else:
                    created_id, history_admission = await run_in_threadpool(
                        _publish_history_messages, sync_service, db,
                        owner_id=owner_id, message_ids=published_ids, write=history_write)
            except HistorySelectionError as exc:
                raise HTTPException(409, detail={"status": "stale_selection", "code": exc.code, **exc.details}) from exc
            except SyncStoreError as sync_exc:
                raise _message_sync_http_error(sync_exc) from sync_exc
```

The closure's `append_selected_history_input` call is above the conflict hunk
and merges without a marker and without `history_branch`. Add the argument by
hand. S3 is the test that catches it if it is missed.

`capture_applied_server_origin_write` re-raises an error from the write
unchanged, so `HistoryBranchChangedError` reaches the `except
HistorySelectionError` on the Sync path as well.

### R9. #3178 (CS-02) vs #3174 (P3): `chacha/conversation_store.py`, imports

Both add an import on the line after the `assistant_startup` import.

Resolution: both, in sorted order.

```python
from tldw_Server_API.app.core.Chat.assistant_startup import AssistantStartup, encode_assistant_startup
from tldw_Server_API.app.core.DB_Management.backends.base import UniqueConstraintError
from tldw_Server_API.app.core.DB_Management.chacha.conversation_search_snippets import (
    build_match_snippet,
    extract_search_terms,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
```

### R10. `dev` #3196 (knowledge provenance) vs #3164 (NS-01): `useNotesEditorState.tsx`, 3 hunks

- **dev** (`35396a21db` and siblings, #3196) wraps the content of a drafted or
  saved note in `retainKnowledgeNoteProvenance(content, originalMetadata)`, so
  a Knowledge QA note keeps its hidden provenance marker. It also sets
  `selectedIdRef.current = created.id` in the create path. Its other edits to
  this file (strip/read the marker on load and restore, the new-draft
  exception in the pending-selection guard, the provenance summary) merged
  without conflict.
- **#3164** builds drafts and payloads from `editRevisionRef`'s snapshot, not
  from the render's `content`, and reordered the create path.

Resolution: the snapshot, wrapped the way dev wraps it.

`buildCurrentOfflineDraft`:

```ts
        title: snapshot.title,
        content: retainKnowledgeNoteProvenance(snapshot.content, snapshot.originalMetadata),
        keywords: [...snapshot.editorKeywords],
```

`saveNote` payload:

```ts
          title: snapshot.title || undefined,
          content: retainKnowledgeNoteProvenance(snapshot.content, snapshot.originalMetadata),
```

Create path: #3164's block, which already contains
`selectedIdRef.current = created.id`.

One more change, without a marker: `isDraftKeptOnDevice` compared the queued
draft's content with the editor text. A queued draft now carries the marker
and the editor text does not, so for a Knowledge QA note it would always
report "not kept on device" after a failed save:

```ts
        queued.title === snapshot.title &&
        // A queued draft carries the note's knowledge provenance marker; the editor does not.
        stripKnowledgeNoteProvenance(queued.content) === snapshot.content
```

Checked: the hook, save-machine, conflict-reload, WYSIWYG, offline-draft and
provenance suites (107 tests), including dev's new "retains Knowledge QA origin
and qualifications after reopening" test, pass.

### R11. `dev` #3188/#3201 vs #3136: `test_license_first_workflow_contracts.py`

Both changed the same two pinned job lists. #3136 added the `ux-regression` job
to `frontend-e2e-tiers.yml`; dev's merge queue added `queue-tick` to
`frontend-required.yml`. Resolution: both.

```python
    "frontend-e2e-tiers.yml": ("critical", "features", "admin", "ux-regression"),
    "frontend-required.yml": ("changes", "frontend-unit-tests", "frontend-required", "queue-tick"),
```

`tests/CI` and `tests/Privileges` pass afterwards (797 passed).

## 4. Changes with no textual conflict

### S1. #3133's conflict-reload test vs #3145's conflict panel

`apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.wysiwyg-typing.test.tsx`,
test "replaces the WYSIWYG content with the server version on a conflict
reload". It clicked `notes-save-conflict-reload`, the old recovery notice's
button. #3164 replaced that notice with one conflict panel, so the test failed
with `Unable to find an element by: [data-testid="notes-save-conflict-reload"]`.

Before:

```ts
    fireEvent.click(await screen.findByTestId("notes-save-conflict-reload"))
```

After (in the merge commit `ed7c3c0f11`):

```ts
    // The single conflict panel (NS-03) replaced the "Reload" notice; "Use
    // their version" is the reload. jsdom has no clipboard, so the discard
    // confirm (mocked to accept) stands in for the copy.
    fireEvent.click(await screen.findByTestId("notes-conflict-take-theirs"))
```

The assertion is unchanged: the WYSIWYG editor shows "Server version body".
Whichever of #3133 and #3145 merges second needs this edit.

### S2. The NS-N1 browser reproduction vs #3164's conflict panel

`apps/tldw-frontend/e2e/ux-regression/notes-p0.spec.ts`, test "NS-N1". Its
precondition waited for the toast button "Reload notes". After #3164 a manual
save conflict shows the conflict panel instead, so removing
`expectKnownDefect` alone would leave the test failing at that precondition.
The driver now takes "Use their version" from the panel:

```ts
    const conflictPanel = pageA.getByTestId("notes-save-issue")
    await expect(conflictPanel).toHaveAttribute("data-kind", "conflict")
    const reloadAction = conflictPanel.getByTestId("notes-conflict-take-theirs")
    await expect(reloadAction).toBeVisible()
    ...
    await reloadAction.click()
    // The unsaved text is copied to the clipboard first. Where the browser
    // refuses the clipboard, the app asks before discarding the text.
    const discardConfirm = pageA
      .getByRole("dialog")
      .filter({ hasText: "Use their version?" })
      .getByRole("button", { name: "Use their version", exact: true })
    await expect
      .poll(async () => (await discardConfirm.isVisible()) || !(await conflictPanel.isVisible()), {
        message: "the reload should finish or ask before discarding the unsaved text",
      })
      .toBe(true)
    if (await discardConfirm.isVisible()) await discardConfirm.click()
```

The final assertion (editor and server both keep the other tab's change) is
unchanged. #3145's description only mentions swapping the wrapper; that was
written before #3164 was stacked on it.

### S3. A test for the P1 leaf check on the Sync v2 path (new, `13ce343b07`)

`tldw_Server_API/tests/Sync/test_sync_v2_native_history_capture.py`,
`test_stale_extend_latest_admission_is_refused_and_publishes_nothing`.

#3171 tests the leaf check without Sync; #3183 tests the Sync-published
admission without `tldw_history_branch`. Neither covers R8's result. The new
test sends from a stale tip as a Sync v2 owner and expects:

- `tldw_history_branch: false` gives 409 `history_branch_changed` with the
  tip, `leaf_ids` and `history_version`, no new message and no new envelope;
- `tldw_history_branch: true` gives 201 and one applied `chat.message` append.

It fails (201 instead of 409) when `history_branch` is left out of the
closure. Add it with whichever of #3171 and #3174 merges second.

### S4. #3189 (in P3) vs #3154 (CS-N2): no-profile Trash test (`96cf49d714`)

`tldw_Server_API/tests/Sync/test_sync_v2_unenrolled_delete.py::test_without_sync_v2_deletes_behave_as_before`
asserted the old cascade: trashing a chat without a Sync profile soft-deletes
its messages. CS-N2 removed that cascade on purpose, so the test failed on its
last line. The test's subject (direct deletes, Sync store never opened) is
unchanged; only the expected state of the trashed chat's message follows
CS-N2.

Before:

```python
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is None
    assert _is_deleted(chacha_db, LEGACY_ANSWER)
```

After:

```python
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is None
    # Without a profile, Trash flags only the conversation (CS-N2, #3154): its messages
    # keep their own deleted flag, so Restore brings the transcript back as it was.
    assert not _is_deleted(chacha_db, LEGACY_ANSWER)
```

Whichever of #3154 and #3174 merges second needs this edit. #3198 changed
other parts of the same file; the merge was clean.

### S5. #3178 (CS-02) vs #3136 (harness): side-panel search helpers (`3dd0f8a83f`)

`apps/tldw-frontend/e2e/utils/extension-sidepanel.ts`. CS-02 shows the matching
message text between a search result's title and its "Server" label, and that
text is part of the button's accessible name. The seeded chats' messages
contain the title's unique word, so every side-panel search result has a
snippet, and all four side-panel specs failed in `openServerChatFromSearch`
(seen in the browser run, section 9).

Before:

```ts
    return this.sidebar.getByRole("button", { name: new RegExp(`^${escapeRegExp(title)} (Server|Local)\\b`) })
    ...
    const result = this.sidebar.getByRole("button", { name: `${title} Server`, exact: true })
```

After:

```ts
    return this.sidebar.getByRole("button", { name: new RegExp(`^${escapeRegExp(title)}(?: .+)? (Server|Local)\\b`) })
    ...
    const result = this.sidebar.getByRole("button", { name: new RegExp(`^${escapeRegExp(title)}(?: .+)? Server$`) })
```

Both stay anchored. Without the `searchResult` change, XS-07 Delete's "no
match" check would pass even when the deleted chat is still listed.

### S6. `dev` #3195 vs #3139/#3147 (harness): fake chat summary has no scope (`c9a494dc11`)

`apps/packages/ui/src/components/Option/Playground/__tests__/harness/fake-tldw-server.ts`,
`chatSummary`. The real `ChatSessionResponse` always has `scope_type`
(`"global"` by default) and `workspace_id`; the fake left them out. Since
#3195, a fresh chat's first turn sets the server chat id, the server-chat
loader fetches the chat, and `validateCachedServerChatId` refuses a missing
scope. The loader then marks the owner unavailable, which replaces the turn's
history view. A stopped or dropped reply could no longer update the
transcript and kept its streaming cursor (`"Partial answer▋"`).

Traced with temporary logging (not committed): `canUpdateView()` was false at
the pipeline's catch, and the controller state was
`unsupported_history_capability` with an `unavailable` owner, after a
`GET /api/v1/chats/server-chat-1`. A real server returns the scope, so this
is a harness fidelity gap, not a product bug.

```ts
  const chatSummary = (chat: FakeChat) => ({
    id: chat.id,
    // The real ChatSessionResponse always carries its scope ("global" by
    // default); the server-chat loader refuses a chat whose scope is missing.
    scope_type: "global",
    workspace_id: null,
    title: chat.title,
```

This also made CC-04's `it.fails` pass for the wrong reason. With the scope
present, CC-04 reproduces again and keeps its marker. Needed in #3139 and in
#3147's copy of the harness, whichever reaches `dev` after #3195.

### S7. `dev` #3195 vs #3139/#3166 and #3163: a fresh chat is now a server chat (`e1688a5017`)

#3195 ("establish native ownership before fresh saved send", restoring
ADR-049) creates a server chat before the first plain send of a fresh chat on
a connected server. Five Playground tests assumed that chat stayed local and
read the local Dexie database. All seven harness failures also appear on
`dev` + `test/playground-integration-harness` alone (and none on `75ab224081`
+ the same branch), so this is between `dev` and those PRs, not caused by the
other PRs here. Each test keeps its subject and now reads the chat's real
owner (`server.chats` in the fake server):

| Test | File | Before | After |
|---|---|---|---|
| harness self-check | `Playground.harness.integration.test.tsx` | "keeps the local turn in the in-memory chat database": rows in Dexie under `historyId` | "keeps a fresh chat's turn with its owner, the server chat": exactly one server chat, holding both turns |
| CS-01 "not appended to the previous chat" | `Playground.conversation-reset.integration.test.tsx` | previous chat = local `historyId`, read from Dexie | previous chat = `serverChatId`, read from the server; also asserts the new send uses another chat |
| CS-01 "drops the stored reference" | same | precondition: reference `conversation_id` = local `historyId` | precondition: reference `conversation_id` = `serverChatId`; post-New-chat assertions unchanged |
| CS-04 reload in a fresh chat | `Playground.interrupted-replies.integration.test.tsx` | question and partial reply in local Dexie rows | server chat holds the question; persistence reads `"serverFailed"` (same as the existing server-chat reload test); transcript, notice and Retry assertions unchanged |
| CS-04/CS-N3 leave and return | same | no `POST /chats`, no server chat | exactly one `POST /chats` (the send's), and that chat holds the whole turn; renamed "...creates no extra server chat" |

Kept failing on purpose: "CS-04: a reply cut off by a dropped connection ...
Retry resends the question". See section 8: Retry duplicates the question in
any server-owned chat.

### S8. Two #3174 (P3) Postgres tests that fail on its own tip (`43a74bcd44`)

Not an interaction: both fail the same way on `origin/feat/chat-idempotent-create-p3`
(`f7e29b60e9`) alone, against a throwaway PostgreSQL 16.2. #3174 says its
Postgres variants had not been run locally. Fixed here so the branch is green;
the same edits belong in #3174.

- `Character_Chat_NEW/integration/test_chat_session_client_id_create.py::test_another_owner_reusing_an_id_never_sees_the_first_owners_chat[postgres]`:
  the `owners` fixture connected through `pg_database_config`, the superuser
  admin DSN. A superuser bypasses row-level security even under FORCE (see the
  `pg_restricted_backend` docstring), so the second owner saw the first
  owner's chat. CI's `tldw_user` is a superuser too. The fixture now uses
  `pg_restricted_backend`, as P8's import test does; with an RLS-bound role
  the assertion holds, so P3's isolation itself is fine.

  ```python
      backend = request.getfixturevalue("pg_restricted_backend") if request.param == "postgres" else None
  ```

- `DB_Management/test_chacha_postgres_migration_v61.py::test_postgres_v60_to_v61_constraints_forced_rls_and_head_rerun`
  lists every column later migrations add to `conversations`. P3's
  `create_request_fingerprint` (PostgreSQL v79) was missing:

  ```python
                      "native_bundle_json": None,
                      "create_request_fingerprint": None,
  ```

## 5. Marker flips

A reproduction becomes a plain assertion once its fix is on the same branch.
Assertions are unchanged in every case.

### F1. Vitest (`22b271b2e6`)

`apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.ux-contract.test.tsx`
(from #3137; fixed by #3148).

| Test | Before | After |
|---|---|---|
| NL-01 (#3103): browse request pages with the server's limit/offset params | `it.fails` | `it` |
| NL-01 (#3103): displayed total comes from pagination.total | `it.fails` | `it` |
| NL-01 (#3103): page 2 shows the next slice of notes, not page 1 again | `it.fails` | `it` |
| NL-02 (#3103): export pages with limit/offset and stops at the true total | `it.fails` | `it` |
| NL-03 (#3103): bulk Assign tags keeps each note's existing tags | `it.fails` | `it`, plus the mock below |

NL-03 needs a mock, because bulk "Add tags" now asks through
`NotesBulkAddTagsPrompt` instead of `promptModal`:

```ts
vi.mock("@/components/Notes/NotesBulkAddTagsPrompt", () => ({
  promptBulkAddTags: vi.fn(async () => ["c"])
}))
```

The file's header comment was reworded, since no `it.fails` is left in it.

### F2. pytest (`2a72e7e5f1`)

`tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py::TestChatSessionEndpoints::test_restore_chat_from_trash_restores_its_messages`
(from #3137; fixed by #3154).

Before:

```python
    @pytest.mark.integration
    @pytest.mark.xfail(strict=True, reason="CS-N2 (#3104): restoring a chat from trash leaves its messages soft-deleted")
    def test_restore_chat_from_trash_restores_its_messages(self, test_client, auth_headers):
```

After:

```python
    @pytest.mark.integration
    def test_restore_chat_from_trash_restores_its_messages(self, test_client, auth_headers):
```

### F3. Playwright, `apps/tldw-frontend/e2e/ux-regression/` (`14ade63200`)

Each `await expectKnownDefect(testInfo, {...}, <fn>)` is replaced by the body
of `<fn>`. The unused `testInfo` parameter and, where no wrapper is left, the
`expectKnownDefect` import are removed.

| Spec | Test | Fixed by | Other change |
|---|---|---|---|
| `notes-p0.spec.ts` | NL-01 | #3148 | none; the assertion is `expect(shownTotal).toBe(apiTotal)` |
| `notes-p0.spec.ts` | NS-01 | #3164 | none |
| `notes-p0.spec.ts` | NS-N1 | #3145 | driver change S2 |
| `chat-p0.spec.ts` | CS-02 | #3178 | none |
| `sidepanel-p0.spec.ts` | XS-01 | #3152 | none |
| `sidepanel-p0.spec.ts` | XS-07 Rename | #3152 | none |
| `sidepanel-p0.spec.ts` | XS-07 Delete | #3152 | precondition text `"cannot be undone"` becomes `"moves to Trash"`; confirm button `"Delete"` becomes `"Move to Trash"` |
| `sidepanel-p0.spec.ts` | XP-08 | #3161 | none |

`README.md`: the "Current reproductions" table lists NE-04 only. The fixed
ones moved to a second table, "Fixed, and now guarded by a plain assertion".

### F4. Shared helper (`3319f8af83`)

`apps/tldw-frontend/e2e/ux-regression/notes-wysiwyg.spec.ts` (#3133) dropped
its private copy of `warmBackendOnce` and imports the shared one from
`e2e/utils/seed-api.ts` (#3136): `await warmBackendOnce(createSeedApi(request))`.

### F5. CS-03 (`9db2a72044`)

`apps/packages/ui/src/components/Option/Playground/__tests__/Playground.server-save.integration.test.tsx`,
"CS-03 (#3104): sending in a fresh chat while connected creates a server
conversation holding the turn": `it.fails` becomes `it`.

The brief said to keep it, because server-by-default saving (D1) was not
built. `dev`'s #3195 has since built it for a fresh plain chat on a connected
server, and the `it.fails` reported "Expect test to fail" on `dev` + #3139
alone. Whichever of #3139 and #3195 lands second (#3195 already has) needs
this flip.

### Markers kept

Checked against the fixes on the branch; each defect is still open.

| Marker | Where | Why it stays |
|---|---|---|
| NE-04 `expectKnownDefect` | `notes-p0.spec.ts` | Print is not fixed |
| CC-01 `it.fails` | `Playground.model-startup.integration.test.tsx` | not fixed |
| CC-04 `it.fails` | `Playground.history-controls.integration.test.tsx` | not fixed; it only appeared to pass before S6 |
| CM-01 `it.fails` x2 | `Playground.message-actions.integration.test.tsx` | not fixed |

No other `it.fails`, `expectKnownDefect` or strict `xfail` names a review id.

## 6. Schema numbering

#3174 (P3) bumps the ChaChaNotes schema: SQLite v74 to v75, PostgreSQL v78 to
v79 (`conversations.create_request_fingerprint`). #3198, merged into the P3
branch afterwards, adds SQLite v75 to v76 and PostgreSQL v79 to v80 (clears
the retired `sync-v2` placeholder persona). `origin/dev` at `94854ca3db` is
still at v74 / v78, and no other branch in this set bumps the schema, so the
chain is linear (dev 74/78, P3 75/79, #3198 76/80) and **nothing was
renumbered**. Check again when #3174 reaches the front of the
queue: if `dev` has moved past v74 / v78 by then, renumber P3's migration to
follow it and update `test_conversation_create_fingerprint.py` and
`test_workspace_chat_startup_migration.py`.

## 7. Generated files

### G1. `apps/tldw-frontend/lib/api/openapi.fingerprint.json`

Every API-changing PR regenerates it, so merges 1, 5, 9, 10, 11, 12 and 13
conflicted on it. The integration side was kept each time and the file was
regenerated once at the end (`4f2e9a7904`): 2,112 paths, 3,265 schemas.

When replaying, regenerate it in each API-changing PR after updating the
branch with `dev`:

```bash
python Helper_Scripts/export_openapi_schema.py --out /tmp/x.json \
  --fingerprint apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

The same command on clean `origin/dev` reproduces dev's committed fingerprint
exactly, so a local export matches CI's.

### G2. `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json`

None of the 13 PRs changes this snapshot. Regenerating it on the integration
branch and on clean `origin/dev` gives byte-identical files. Both differ from
dev's committed snapshot by the same 20 lines: `usage_quota_deps._check` on
the unified RAG search and Text2SQL routes, added by `4b3ad3ced4` (per-user
RAG quotas, 2026-10-03) after the snapshot's last regeneration
(`937cf0fb09`). So `test_privilege_registry_snapshot_matches_live_app` is
stale on `dev` itself.

`d9359f9b08` committed the regenerated snapshot on its own. `dev` has since
landed the same refresh as #3179, byte for byte, so after merge 15 that commit
changes nothing and can be dropped on replay.

After merge 15 both files were regenerated again and neither changed: the
fingerprint stays at 2,112 paths and 3,265 schemas, and the snapshot equals
`dev`'s. After merge 16 the fingerprint moved to 2,112 paths and 3,266
schemas (`f871694cde`); the snapshot regenerated unchanged.

## 8. How the fixes interact

Behaviour that exists only when the PRs are combined.

- **Leaf check for Sync v2 owners (P1 + P3).** With R8, a Sync v2 owner's
  `POST /chats/{id}/messages` with `tldw_history_branch: false` from a stale
  tip is refused with 409 `history_branch_changed` and publishes no envelope.
  Neither PR does this alone. `/chat/completions` server-managed history is
  still refused for Sync v2 owners (`sync_owner_unsupported`) before the leaf
  check is reached.
- **Conflict reload in WYSIWYG mode (NS + NE-01).** "Use their version" and
  "Load latest version" go through `loadDetail`, which calls
  `replaceWysiwygHtml`, so the uncontrolled editor is rewritten with the
  server copy. The old "Reload" notice no longer exists (S1).
- **Rename offer after a flushed save (NS + #3190). Not fixed here.** The offer
  now fires from the save's own snapshot (R5), including a save the leave
  guard triggers. That exposes a race in `useNotesWikilinkRename`: the offer
  counts the linking notes with a request, and nothing stops it from opening
  after the notes page has unmounted. `closePrompts()` runs at unmount, before
  the prompt exists, and the prompt has `duration: 0`. So "rename a note, then
  navigate away inside the autosave debounce" can leave the "N notes link to
  ..." notice open on the page the user went to. Before #3164 that edit was
  lost on leave, so no save and no offer happened.
  - Shown at hook level with a throwaway test (not committed): start
    `handleNoteRenamed`, replace the hook's component with another page while
    the referrers request is pending, then answer it. The "Update links" button
    is in the document afterwards. Not checked in a browser.
  - Suggested fix, in #3190's hook: set a ref in the unmount cleanup and return
    from `offer` before `openPrompt` when it is set.
- **Trash under Sync v2 (CS-N2 + #3181).** Without Sync, trashing a chat now
  leaves its messages alone. With an active Sync v2 profile, `delete_chat_session`
  still deletes each message: published ones by tombstone, unpublished ones
  directly. The comment there says "as without a profile", which is no longer
  true of messages. Nothing breaks today, because Restore is refused for Sync
  owners, but a future Sync restore would bring back an empty chat.
- **Duplicate chat ids (P3 + P8).** P3 makes `add_conversation` raise
  `ConflictError(entity="conversations")` for a taken id on PostgreSQL too. P8's
  import store catches `UniqueConstraintError` around the same call; that
  branch is now unreachable, and the endpoint still answers 409
  `chat_id_conflict`. A P3 create with the id of an imported chat is a 409 (no
  create fingerprint), and an import over a P3-created chat is a 409 (no
  import marker).
- **Content search and Trash (CS-02 + CS-N2).** Trashed chats keep live
  messages after CS-N2. CS-02's content match checks the conversation's
  `deleted` flag and owner inside the message subquery, so they stay out of
  results.
- **Fresh chats are server chats (`dev` #3195 + #3147/#3163/#3139).** A fresh
  chat on a connected server is now created on the server at its first send,
  so the persistence label reads "Saving to server..." then "Saved on server"
  (#3147) instead of "Saved on this device", CS-03 holds (F5), and CS-04's
  interrupted-reply handling runs on the server-chat path for fresh chats too.
- **Retry duplicates the question in a server chat (#3163). Not fixed here.**
  After a reply is cut off by a dropped connection, Retry sends the question
  again as a new message. In a server-owned chat the question was already
  admitted, so the transcript and the server chat end with
  `user: Q, user: Q, assistant: <new reply>`. The completion request itself is
  right (the question once, no partial reply as context).
  - Reproduced with a throwaway harness test (not committed) on #3147's own tip
    (`79afbf2148`) in a saved server chat, so it predates the integration.
    #3163's server-chat drop test does not click Retry; its fresh-chat drop test
    does, and went red once #3195 made fresh chats server chats.
  - The fresh-chat test "CS-04: a reply cut off by a dropped connection ...
    Retry resends the question" is left failing on this branch on purpose.
  - Likely fix, in #3163: for a turn whose question was admitted, Retry should
    settle a new reply against the admitted question (or remove the kept
    question first) instead of appending it again.

## 9. Browser suite and ratchet baselines

`bun run e2e:ux-regression` was run on this branch:

| Run | At | Result |
|---|---|---|
| `integration` | `14ade63200` (before merges 14, 15) | 15 passed, 5 failed: 4 side-panel specs (fixed by S5) and the notes-editor a11y count |
| `integration2` | `9db2a72044` (after merge 15) | 19 passed, 1 failed: the notes-editor a11y count |
| `a11y3` | same, a11y specs only | 4 passed, 1 failed: the notes-editor a11y count |
| `a11y4`, `a11y5` | `5a9b4dd54b`, a11y specs only | 5 passed, 5 passed |
| `integration3` | `5a9b4dd54b`, full suite | 20 passed, 0 failed |
| `integration4` | `f871694cde` (after merge 16), full suite | 19 passed, 1 failed: NE-01 "typing continues at the caret after the new note's first autosave" |
| `ne01a`, `ne01b` | `f871694cde`, NE-01 specs only | 3 passed, 3 passed |

The `integration4` NE-01 failure: the space typed right after the first
autosave was lost ("Hello worldagain"). That spec passed in every other run,
before and after merge 16, including both reruns. It looks like a timing race
under heavy machine load. The likely window: the reload after the first save
checks for newer edits, then writes the WYSIWYG document in a later render; a
keystroke landing between the check and the write is overwritten. This is not
confirmed; see section 12.

`integration3` still exited 1 after the tests: the runner's post-run guard
found `Databases/` in the worktree (moved aside before `integration4`). Backend pytest runs started from the same
worktree had created it (gitignored runtime databases), not the browser
suite. Run the suite from a worktree where pytest has not run, or move that
directory aside first.

In every run, all reproductions flipped in F3 passed as plain assertions
(NL-01, NS-01, NS-N1, CS-02, XS-01, XS-07 x2, XP-08 once S5 was in), as did
the three NE-01 WYSIWYG specs and the harness check. NE-04 still reproduces.
Both request budgets matched their baselines in every run, as did every a11y
entry except the one below.

**B1/B2, the notes-editor AX-07 color-contrast count.** The baseline said 9.
Three runs measured 4, 8 and 6. The scan ran as soon as the textarea showed
the note, but the editor's toolbar stays disabled until `loadDetail` finishes
(the pending-selection guard keeps `loadingSelection` true through the
task-state fetch), and axe skips disabled controls. B1 had lowered the
baseline to the first measurement (4). B2 makes the spec wait for the
input-mode toggle to be enabled before scanning, and sets the baseline to 8:
the empty-library nodes with "Create study pack" in place of "Create note".
"Save" no longer fails, because it is primary only while there are unsaved
edits (#3164). Two runs with the wait both measured 8.

When replaying, B1 can be skipped: B2 replaces it.

## 10. Test results, head against dev

Final comparison: head `f871694cde` against dev `94854ca3db`, same commands
on both sides. Vitest groups ran with `--testTimeout=30000`; pytest with
`-n 4`. "Head-only failures" are tests that fail on head and pass on dev.

| Suite | dev | head | Head-only failures |
|---|---|---|---|
| Vitest `components/Notes` | 485 passed / 50 failed | 769 / 33 | none (head fixes 16 dev failures) |
| Vitest `Option/Playground` | 840 / 45 | 884 / 46 | 1: the CS-04 Retry duplicate, left failing on purpose (section 8) |
| Vitest Sidepanel + `routes/__tests__/sidepanel*` | 468 / 0 | 516 / 0 | none |
| Vitest `hooks/chat`, `hooks/chat-modes` | 639 / 1 | 651 / 1 | none |
| Vitest `models`, `services/tldw`, `db/dexie` | 1,080 / 0 | 1,109 / 0 | none |
| Vitest `components/Common`, `entries/shared` | 1,552 / 4 | 1,587 / 6 | 2 flaky (see below) |
| Vitest `routes/__tests__/option-notes*` | 2 / 0 | 2 / 0 | none |
| Web: live-tier, `e2e/utils`, `__tests__/e2e`, networking, navigation | 227 / 1 | 257 / 1 | none (the same 404-allowlist guard test fails on both) |
| pytest `tests/CI` | 747 passed | 754 passed | none |
| pytest `tests/Privileges` | 43 | 43 | none |
| pytest Notes, Notes_NEW, Notes_Tasks, Notes_Graph | 1,209 | 1,379 | 1 Hypothesis "input generation is slow" health check; passes alone |

- Typecheck (`bun run typecheck`): 0 errors.
- Ruff on the 83 Python files changed against dev: 37 findings on head, 37 on
  the same files on dev; none new.
- `Helper_Scripts/ci/check_shard_coverage.py`: OK, `new_uncovered=0`.
- The two Common failures (ThemeEditor "transient themes without ids",
  ConversationTab "summary window below an edited threshold") also fail on
  dev when run alone (3 runs each side: dev failed 1 of 3, head 2 of 3), and
  `components/Common/Settings` is identical on both sides. Timing flakes under
  load.

Merge 16 changed no chat, notes or sync backend code, so the other backend
groups were compared once, at `5a9b4dd54b` against dev `27ce976387`:

| Suite | dev | head | Head-only failures |
|---|---|---|---|
| ChaChaNotesDB | 1,094 passed | 1,190 | none |
| Character_Chat_NEW | 956 | 1,142 | 1 Hypothesis health check (dev had a different one) |
| Chat | 2,806 | 2,964 | none |
| Chat_NEW | 500 | 524 | 3 Hypothesis health checks; all 17 pass alone |
| Sync | 3,008 | 3,118 | none |
| DB_Management (note/history/conversation/message/chat/wikilink) | 686 | 713 | none |

An earlier run, before S4, had one real head-only failure: Sync
`test_without_sync_v2_deletes_behave_as_before` (S4).

## 11. PostgreSQL

Postgres variants of every changed or relevant test file (76 files on head, 68
on dev: ChaChaNotesDB, Character_Chat_NEW, Chat, Sync, Notes_NEW, Notes_Graph and
the chat/note/history/sync DB_Management files) ran with `-n 4` against a
throwaway PostgreSQL 16.2 from the `pgserver` wheel (installed into the
scratchpad only, `TEST_DATABASE_URL` pointing at it, `TLDW_TEST_NO_DOCKER=1`),
at `5a9b4dd54b` and at dev `27ce976387`.

| | Passed | Failed | Errors |
|---|---|---|---|
| dev | 1,944 | 1 | 3 |
| head | 2,448 | 3 | 3 |

- On both sides, environmental: the 3 errors (`test_prompt_studio_sync_log_backends`,
  the wheel has no `pgcrypto`) and
  `test_sync_v2_notes_link_postgres_contract::test_postgres_notes_link_same_edge_scope_is_tenant_isolated_by_rls`
  (asserts RLS through the superuser DSN).
- Head only: the two P3 tests in S8. Both fixed; 44 of 44 pass in their files
  afterwards, Postgres included.

Merge 16 changed no chat, notes or sync backend code, so the Postgres run
was not repeated after it.

## 12. Follow-ups

- **Retry duplicate in server chats** (#3163), section 8.
- **Rename offer after leaving the page** (#3190 + #3164), section 8.
- **Harness scope fields** (S6) belong in #3139 and in #3147's harness copy.
- **Possible NE-01 race** (section 9): a keystroke typed while the reload after
  a new note's first save is being applied may be overwritten. Seen once in
  six runs of that spec; not reproduced on demand.
- **P3 Postgres tests** (S8) belong in #3174.
- **Knowledge provenance marker in WYSIWYG** (dev #3196, not caused by these
  PRs): `loadDetail` and `applyOfflineDraftToEditor` strip the marker from the
  Markdown text, but still build the WYSIWYG document and
  `markdownBeforeWysiwygRef` from the raw content, marker included. That is the
  same on `dev`; it is noted because NE-01 makes WYSIWYG usable again.
- **`origin/DEV` on the remote.** The remote has a branch `DEV`
  (`d9c245ac14`) as well as `dev`. On macOS a plain `git fetch origin` writes
  `refs/remotes/origin/DEV` over `refs/remotes/origin/dev`, because the
  filesystem ignores case. `git fetch origin dev` puts it back. Fetch branches
  by name until `DEV` is deleted.
