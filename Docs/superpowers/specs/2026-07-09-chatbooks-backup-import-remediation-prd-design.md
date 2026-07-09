# Chatbooks Backup Import Remediation PRD

**Date:** 2026-07-09
**Backlog:** TASK-12098
**Milestone Tasks:** TASK-12098.1, TASK-12098.2, TASK-12098.3
**Source Review:** `Docs/Reviews/CHATBOOKS_BACKUP_IMPORT_UAT_UX_REVIEW_2026_07_09.md`
**Status:** Approved design

## Purpose

Make Chatbooks backup and import possible, straightforward, and trustworthy
from the WebUI and browser extension.

This PRD addresses every finding from the 2026-07-09 Chatbooks backup/import
UAT and UX review. It does not expand Chatbooks into a new backup product or
rewrite the existing architecture. The intent is to correct the current
contract, defaults, labels, and tests so the existing Chatbooks surfaces can be
accepted by users performing backup, restore, and OpenWebUI migration tasks.

## Problem Statement

The current Chatbooks feature is technically capable in several paths, but the
acceptance-critical backup/import flows fail or require expert workarounds:

- User and API docs describe a "backup everything" flow that the current UI
  blocks and the backend does not implement consistently.
- Archive import defaults send unsupported media import flags, so restore fails
  without user correction.
- Settings presents Chatbooks controls that look like backup/restore but only
  export manually-entered conversation IDs.
- OpenWebUI import has useful preview and safety controls, but attachment
  hydration asks users to remember or discover imported tldw conversation IDs.
- Existing tests cover mechanics, not the acceptance-critical failure cases.

## Goals

1. Define one consistent all-export contract across API, WebUI, extension, and
   docs.
2. Make default archive import valid and safe.
3. Add a visible, understandable Backup all path with a scope summary.
4. Clarify Settings so it cannot be mistaken for full backup/restore.
5. Reduce OpenWebUI hydration recall burden by reusing a visible import job
   scope.
6. Add focused regression coverage for the reviewed failures.

## Non-Goals

- No new backup product surface.
- No full Chatbooks architecture rewrite.
- No reconstruction of external source files that tldw does not store.
- No scheduled backups.
- No client-side encryption or password-protected archive design.
- No broad redesign of unrelated Chatbooks job infrastructure.

## Definitions

### Full Account Export Scope

"Full user-account export" means every tldw-owned record and stored artifact
associated with the account, including:

- account-owned metadata, settings, preferences, tags, categories, and
  relationships stored in per-user data stores
- conversations
- notes
- characters
- world books
- dictionaries
- prompts
- evaluations
- generated documents
- media records
- media metadata
- media transcripts, chunks, captions, summaries, tags, links, and processing
  results
- stored user/account file artifacts and attachments
- embeddings and vector-store records associated with exported account data
- any other user-owned records required to restore that account's state

Where tldw stores file bytes or attachment artifacts for the account, Backup all
must include those bytes. Where tldw only stores an external URL, local path, or
provenance pointer, Backup all must export that stored pointer and related
account data, not pretend that unavailable source-file bytes exist.

The scope summary must identify exported account-data categories and any stored
pointer-only sources. It must not label a pointer-only source as exported file
contents.

### Export Selection Semantics

The export contract must use these semantics:

- `content_selections` omitted: export the full user-account scope.
- `content_selections: {}`: export the full user-account scope.
- Non-empty `content_selections` object: explicit allowlist mode.
- Empty arrays inside a non-empty object mean export none for that type.
- A non-empty allowlist that resolves to zero total exportable items is invalid:
  normal UI flows must block it with a clear validation message, and the API
  must return a 4xx validation error instead of silently creating an empty
  backup.

Examples:

```json
{}
```

Means the full user-account scope if `content_selections` is omitted or set to
`{}`.

```json
{
  "content_selections": {}
}
```

Means the full user-account scope.

```json
{
  "content_selections": {
    "conversation": ["1", "2"],
    "note": []
  }
}
```

Means export conversations `1` and `2`, export no notes, and export no other
content types unless they are explicitly present.

```json
{
  "content_selections": {
    "conversation": []
  }
}
```

Means an invalid zero-item allowlist. It must not be treated as Backup all.

This is a behavior change for clients that previously sent `{}` expecting an
empty export. The PRD requires API docs and release notes to call it out.

## User Experience Requirements

### Backup All

The main Chatbooks page and browser extension `/chatbooks` route must expose a
primary Backup all path.

Backup all must:

- show a readable scope summary before export
- use the API all-export contract instead of fetching every ID client-side
- include all tldw-owned account data, including media records and associated
  stored/derived media data
- identify pointer-only sources before the job starts when known
- create an export job without requiring per-type switches

Selective export remains available for power users and continues to send
explicit IDs.

### Archive Import

Default archive import must be valid and restore all account data present in
the archive:

- media records and associated stored/derived media data import by default when
  present in the archive
- embeddings and vector-store records import by default when present in the
  archive
- any legacy flag for unavailable external/raw source-file import must be off,
  hidden, or clearly marked unavailable
- default archive import must not require users to know backend limitations

### Settings

Settings must not present itself as full backup/restore.

Default product direction:

- Settings Chatbooks becomes an entry point to the full Backup & Import page.
- Any conversation-only export shortcut is secondary and clearly labeled
  `Conversation export shortcut`.

If the shortcut remains, it must state that it exports selected conversations
only and does not perform a full backup.

### OpenWebUI Hydration

OpenWebUI hydration must support the normal path without manual conversation ID
paste.

Acceptable implementations:

- use the last OpenWebUI import job
- let the user select an OpenWebUI import job
- both

The hydration panel must show which job/scope is being used, including source
format, source user when available, conversation count, and attachment-reference
summary when available.

Manual conversation ID entry may remain as an advanced override.

### Naming

Visible product copy must not call the backup/import surface "Playground".

Preferred visible label:

- `Chatbooks Backup & Import`

Internal route and component names may remain unchanged if renaming them creates
unnecessary churn.

## Milestone Tasks

### TASK-12098.1: P0 Chatbooks Backup Restore Correctness Remediation

Objective:

Make backup-all and default archive restore acceptance-ready.

Ordered checkpoints:

1. Backend export contract.
2. Import defaults.
3. UI/extension Backup all affordance.
4. Docs and tests.

Acceptance criteria:

- API export treats omitted `content_selections` as the full user-account scope.
- API export treats `content_selections: {}` as the full user-account scope.
- The full user-account scope includes media records and associated
  stored/derived media data.
- The full user-account scope is not limited to the named content-selection
  types; it includes other account-owned records needed to restore account
  state.
- Non-empty `content_selections` remains explicit allowlist mode.
- Empty arrays in allowlist mode mean no items for that type.
- Zero-item allowlists are rejected with a 4xx validation error instead of
  creating empty backups.
- Main UI and extension Backup all use the API all-export contract and show a
  scope summary.
- Default archive import restores media records and associated stored/derived
  media data present in the archive.
- Default archive import restores embeddings/vector-store records present in the
  archive.
- Unsupported raw-source-file import toggles are not presented as normal enabled
  options.
- User guide, API docs, and examples match the runtime behavior.

### TASK-12098.2: P1 Chatbooks Backup Import UX Clarity Remediation

Objective:

Reduce ambiguity and recall burden in the existing surfaces.

Acceptance criteria:

- Visible heading/copy no longer says `Chatbooks Playground`.
- Settings points users to the full Backup & Import page by default.
- Any remaining Settings conversation export shortcut is clearly secondary and
  labeled as conversation-only.
- OpenWebUI hydration can reuse a last or selected import job scope.
- Reused hydration scope is visible before preview/job creation.
- Manual conversation ID paste is no longer required for the normal OpenWebUI
  hydration path.

### TASK-12098.3: P2 Chatbooks Backup Import Acceptance Coverage

Objective:

Add the smallest useful regression coverage for the reviewed failures.

Minimum required tests:

- API all-export contract: omitted selections, `{}`, full user-account scope,
  explicit allowlist, empty arrays inside allowlist mode, and zero-item allowlist
  rejection.
- Main UI default archive import restores all account data present in the
  archive, including media records and associated stored/derived media data.
- Settings import shortcut is removed/demoted to a safe entry point or uses the
  same full-account restore defaults.
- Main UI Backup all export fires and uses the all-export contract.
- Extension `/chatbooks` inherits and verifies the same Backup all behavior.
- OpenWebUI hydration can reuse a visible import job scope without normal-path
  manual conversation ID paste.

Existing WebUI E2E coverage must no longer treat a backup-all export that does
not fire as acceptable.

## Data Flow

### Backup All

1. User opens Chatbooks Backup & Import.
2. User chooses Backup all.
3. UI shows scope summary and limitations.
4. User starts export.
5. Client sends omitted `content_selections` or `content_selections: {}`.
6. API expands to the full user-account scope.
7. Export job records included account-data categories and warns only when a
   stored external pointer has no stored file bytes to include.
8. UI job tracker shows progress, completion, warnings, and Download.

### Selective Export

1. User opens advanced/selective export.
2. User selects content types and IDs.
3. Client sends a non-empty `content_selections` allowlist.
4. API exports exactly the requested types and IDs.
5. Empty arrays inside the allowlist exclude that type.
6. If the selected allowlist resolves to zero total exportable items, the UI
   blocks submission and the API rejects direct requests.

### Archive Import

1. User selects a `.chatbook` or compatible archive.
2. UI previews archive.
3. Default import flags are valid.
4. User starts import.
5. API restores account data present in the archive and records warnings/skips
   only for unsupported or unavailable restore targets.

### OpenWebUI Hydration

1. User imports OpenWebUI JSON or DB.
2. Import result/job exposes enough scope to identify imported conversations.
3. Hydration panel offers last/selected import job scope.
4. User reviews visible scope.
5. User enters server-local data root.
6. User previews attachments.
7. User runs hydration job after the preview is current.

## Error Handling

- Pointer-only media sources must be visible before export where known and in
  job results after export.
- Unsupported import targets must produce visible warnings, not silent omission.
- Selective export with zero total selected/exportable items must fail with a
  clear 4xx validation message rather than producing an empty archive.
- Archive import must not fail by default because of unsupported options.
- Hydration must not create a job until a current preview exists for the visible
  scope.
- If no reusable OpenWebUI import scope exists, the UI should explain that the
  user can import first or use manual advanced IDs.

## Documentation Requirements

Update:

- `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
- `Docs/API-related/Chatbook_API_Documentation.md`
- any route/page copy or test expectations that say `Chatbooks Playground`

Docs must explicitly state:

- omitted or `{}` export selections mean full user-account export
- non-empty selections are allowlists
- zero-item allowlists are invalid and do not mean "backup all"
- full user-account export includes media records and associated stored/derived
  media data
- pointer-only media sources export as stored pointers, not unavailable source
  file contents
- archive import restores all account data present in the archive by default
- Settings is not the full backup/restore workflow if the shortcut remains

## Compatibility And Rollout

The all-export contract changes the meaning of `{}` from the current effective
empty-export behavior to full user-account export.

Required rollout work:

- API documentation callout.
- User guide callout.
- Release note or migration note for API clients.
- Tests proving omitted, `{}`, allowlist, empty-array, and zero-item rejection
  behavior.

## Success Criteria

The remediation is successful when:

- a first-time user can create a full user-account backup from WebUI or
  extension without selecting per-type IDs
- a first-time user can restore a `.chatbook` archive with defaults
- Settings no longer implies full backup/restore unless it actually performs it
- OpenWebUI hydration no longer requires normal-path manual conversation ID
  paste
- tests fail if the reviewed P0/P1 behaviors regress

## References

- `Docs/Reviews/CHATBOOKS_BACKUP_IMPORT_UAT_UX_REVIEW_2026_07_09.md`
- `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
- `Docs/API-related/Chatbook_API_Documentation.md`
- `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/Settings/chatbooks.tsx`
- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
