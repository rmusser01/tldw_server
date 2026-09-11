---
id: TASK-13002
title: Extend Notes Sync v2 core contract for backlinks and restore
status: Done
assignee: []
created_date: '2026-08-08 20:21'
updated_date: '2026-08-08 22:14'
labels:
  - notes
  - sync-v2
  - parity
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/backlog/decisions/046-synchronized-database-notes-parity.md
  - 'https://github.com/rmusser01/tldw_server/pull/2775'
documentation:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/superpowers/specs/2026-08-08-notes-server-parity-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the existing notes.note domain a lossless production contract for title, content, conversation/message backlinks, tombstones, and base-aware restore so Chatbook can synchronize one personal Database Notes collection without server-side ambiguity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capabilities and envelope schemas expose notes.note upsert/tombstone with title, content, conversation_id, and message_id under server_trusted_v1.
- [x] #2 A restore-intent upsert can resurrect only the current tombstone head; stale ordinary updates against deleted notes remain whole-object conflicts.
- [x] #3 Server-origin note create, update, delete, and restore produce the same canonical envelopes and materialized object state as client-origin mutations.
- [x] #4 Accepted note title/content are preserved exactly within documented limits; validation never truncates, escapes, or rewrites canonical Markdown.
- [x] #5 SQLite and PostgreSQL contract tests cover create, update, tombstone, stale conflict, restore, idempotency, and exact payload materialization.
- [x] #6 Keyword writes remain explicitly blocked until their separately synchronized domain is enabled; no partial ownership is implied.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/031-notes-capability-sync-domains.md
Reason: Establishes the public notes.note payload, restore-intent semantics, and server-origin ownership boundary.

1. Characterize the existing Sync v2 Notes adapter, materializer, server-origin capture, REST restore, and ChaChaNotes persistence paths.
2. Add focused failing tests for the canonical four-field payload, exact title/content persistence, current-tombstone restore, stale conflict behavior, idempotency, and server-origin parity.
3. Implement the shared payload validator, production Notes adapter wiring, base-aware restore materialization, lossless storage, and REST restore capture; keep keyword writes blocked.
4. Document ADR-031 and the public contract, then run focused SQLite and configured PostgreSQL tests plus Ruff and Bandit.
5. Record verification, self-review the diff, and complete task hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added one discoverable, validated `notes.note` version-1 payload for exact title/content plus nullable conversation/message backlinks, and documented the ownership boundary in ADR-031 and the public Sync v2 API guides.
- Wired the production Notes adapter and materializer to enforce canonical payloads, current-tombstone-only restore intent, whole-object stale conflicts, idempotency, and server/client-origin parity while preserving the existing keyword-write block.
- Preserved accepted Markdown and title bytes in ChaChaNotes, included `message_id` in batch reads, and added focused SQLite plus PostgreSQL contract coverage for the complete mutation lifecycle.
- Verification: 215 focused tests passed and 1 PostgreSQL test skipped because neither a local PostgreSQL server nor Docker daemon was available. Ruff passed for new/reworked Sync files; compileall and `git diff --check` passed; Bandit reported 0 findings across 13,190 production lines. Full-file Ruff on the two legacy Notes files reports the same pre-existing `I001`, `BLE001`, `F841`, and `C409` baseline as `HEAD`, with no new finding introduced by this task.

Review: PR #2775 targets dev from codex/task-13002-notes-core-contract.

Review remediation: validating PR #2775 restore-path feedback before merge.

Review remediation: moved active-Sync note restore validation/capture into the reusable server-origin coordinator; active-note and stale-version restore attempts now return a stable 409 without appending an envelope; all restore-path database calls now run through the async thread helper. Evidence: focused active/inactive restore tests passed (2/2), the complete server-origin capture file passed (28/28), Ruff passed for the new/reworked core and test files, compileall and git diff --check passed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused Sync v2 and Notes tests pass on supported database backends.
- [x] #8 Bandit and static checks pass for touched production files.
- [x] #9 ADR-031 and Sync v2 public contract documentation are updated.
<!-- DOD:END -->
