# Chat Workspace Owner-Fenced Checkpoints

Tracking: TASK-13398.12 (formerly TASK-13396.12 before the dev ID collision).
The human approved the existing-store approach and this written specification
on 2026-09-30. Implementation and native acceptance are still in progress.

## Scope

Extend the existing workspace chat-session record, not the global Playground
session or a new database. Preserve the current split-key/optional IndexedDB
storage and bounds. Staging remains transient; this change restores conversation
selection and the unsent composer draft, not queued sends or automatic retries.

## Record And Qualification

Carry the verified target/owner key, current workspace chat reference, validated
history-selection reference, draft, messages and existing history/server IDs.
Reuse serverChatMirrorOwnerKey and the existing H1 reference/controller contract;
never store credentials. Clone/serialize these fields through existing helpers.
An empty conversation may carry a draft with a null history reference; it must
also have no message rows or history/server ID. This grants no history authority.

Read only the exact workspace/reference checkpoint after workspace hydration and
current account/target qualification. Reject a missing or mismatched owner,
workspace/reference, or unqualified history reference. Do not adopt unowned legacy
rows or fall back from a missing reference-specific entry to the workspace base
entry. Rejected records remain untouched and do not grant mutation authority.

## Handoff And Precedence

Save the outgoing qualified workspace session before switching or unmounting.
Only save while that captured owner/workspace/reference is still current; a stale
effect must not copy the incoming chat into the outgoing checkpoint.

Explicit route history intent, New Chat, and edits/selections made during an
asynchronous restore take precedence over the checkpoint. Re-check generation,
owner and workspace/reference after each await. Account/target changes cancel
restoration and clear the mounted sensitive view through existing boundaries.

Restore the reference through the existing H1 controller's owner validation,
not by treating cached message rows or a server ID as mutation authority. Restore
the draft only when it cannot replace newer typing. No automatic send on restore.
New Chat changes the reference and cannot resurrect the old base-key session.

## Integration And ADRs

Apply the same session qualification to Research Workspace's existing save/read
handoff so its shared records cannot become an unowned cross-surface fallback.
Keep native-fork and selected-ancestry behavior unchanged.

ADR-008 covers split persistence; ADR-049 covers owner-qualified selected history;
ADR-050 covers native-fork authority. This is an application of those decisions,
not a replacement: no new persistence architecture or owner-adoption policy.
Accepted ADR files remain unchanged.

## Verification

Test first: exact-key/no-legacy-fallback reads, owner/target/workspace/reference
mismatch, serialization/clone round trips, failed hydration, account/workspace
ABA, New Chat/explicit route precedence, typing during restore, cancellation and
outgoing handoff. Keep default unrelated chat surfaces unchanged.

Actual Chrome CDP UAT must demonstrate authenticated fresh-document reload,
workspace A/B switching, retained drafts, selected conversation identity, and
no context bleed. Use the real API/database/local model and native browser storage.
Regression mocks do not count as browser acceptance.

## Self-Review

No new store, database, fallback authority, credential persistence or automatic
retry is introduced. Shared record readers must be updated together; store edits
sequence after TASK-13396.10 and Panel integration after TASK-13396.8.
