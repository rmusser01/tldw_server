---
id: TASK-13151
title: Document Personal Context Profile server operations and architecture
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-01 14:47'
updated_date: '2026-09-02 08:19'
labels: []
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish accurate server operator and developer documentation for the canonical Personal Context peer, authenticated API, encrypted storage, and current Sync-v2 boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An operator guide covers authentication, master-key setup, Chatbook linking, export, server purge behavior, and current operational limitations.
- [x] #2 A developer guide maps Shared Core parity, per-user storage, real key-custody ownership, services, API routes, Sync-v2 adapters, conflict metadata, purge fencing, the ten-item extension checklist, and targeted tests.
- [x] #3 The existing Personal Context API reference accurately distinguishes shipped Sync-v2 support from missing server-origin publication and purge acknowledgement.
- [x] #4 User, developer, and API indexes plus MkDocs navigation make the guides discoverable and cross-link stable Chatbook documentation.
- [x] #5 Generated published documentation is reproducible and strict MkDocs, endpoint, custody, bootstrap, materializer, composed-app, contract, link, and diff checks pass after the final rebase.
- [x] #6 Offline/queued, locked, incompatible, version-conflict, first-link semantic-collision, post-link semantic-collision, and purge-pending guidance is explicit and consistent with Chatbook.
- [ ] #7 Server guides and API limit cross-peer convergence to the eligible snapshot from successful user-approved first-link reconciliation, distinguish protocol capability from the absent ongoing Personal Context client caller, state that Manual Sync is Notes/Chat only, and state that server REST mutations are not published back.
- [ ] #8 Server guides disclose pre-approval bootstrap metadata and transient remote-content download, adaptive interview egress/disclosure timing, Chatbook HTTP/TLS probe-runtime differences, partial local-removal and recovery limits, and incomplete purge distribution/acknowledgement without inventing unsupported recovery controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare the merged server guides and API reference against the corrected Chatbook documentation design at commit 7698e29f91393ec3e930808ef4178926364634c4.
2. Correct the operator, developer, and API documentation to distinguish reviewed first-link publication from the absent ongoing Personal Context client lifecycle, and document bootstrap, removal, adaptive-interview, transport, and purge limits where relevant.
3. Regenerate Docs/Published only through Helper_Scripts/refresh_docs_published.sh and verify canonical/generated parity.
4. Run targeted Personal Context documentation, link, contract, parity, and strict MkDocs checks; inspect the final diff and commit the correction.

ADR required: no
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: This is a shipped-behavior documentation correction under the existing Personal Context authority, encryption, and Sync boundary; it makes no new architecture decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initial operator, developer, API, discovery, and generated documentation landed in PR #2858. This follow-up corrects the shipped-behavior boundary against Chatbook design commit 7698e29f91393ec3e930808ef4178926364634c4: cross-peer equality is limited to the eligible snapshot from successful user-approved first-link reconciliation; later Chatbook mutations remain in an undrained encrypted local outbox because no shipped ongoing Personal Context caller exists and Manual Sync covers Notes/Chat only; server REST mutations are not published back. The guides now disclose pre-approval bootstrap metadata plus transient server record/proposal download, content-free durable review/UI state, adaptive interview egress and provider-display timing, Chatbook HTTP/TLS probe-runtime differences, partial local removal and absent recovery import, canonical key-cleanup retry limits, and incomplete purge distribution/acknowledgement. The exact shared four-bullet block matches the canonical Chatbook specification byte-for-byte. Canonical source and generated published copies match byte-for-byte, and a second Helper_Scripts/refresh_docs_published.sh run preserved the identical generated diff. Verification with /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python 3.11.13: check_top_guides_docs_path_hygiene.py passed; check_public_private_boundary.py reported OK; strict MkDocs completed successfully; the focused eight-file Personal Context suite collected 78 tests and passed 78 with 4 warnings in 24.67s; exact shared-block, required-claim, stale-claim, source-only-link, canonical/generated parity, and git diff whitespace guards passed. ADR required: no. ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md. Reason: documentation correction only; no storage, authority, encryption, or Sync architecture changed. Bandit: not applicable because only Markdown source, generated Markdown, and Backlog metadata changed. Lesson learned: the server transport's five Personal Context domains prove protocol capability, not that a production client schedules or exposes an ongoing lifecycle; documentation checks must verify callers and UI reachability separately. Known skips/blockers: full server suite not run because the approved documentation plan requires only the focused contract/API/Sync suite; no implementation concern emerged from those 78 tests.

Spec-review P1 follow-up: removed the developer-guide opening's incorrect attribution that the server publishes the reviewed first-link snapshot. The opening now limits server participation to returning the bootstrap snapshot, accepting/materializing Chatbook's approved first-link envelopes, and recording link completion; later-envelope handling remains labeled transport capability without a shipped Chatbook caller. Regenerated the published guide from the canonical source. Review verification: canonical/generated parity passed; the exact shared four-bullet block remained byte-identical to Chatbook design commit 7698e29f91393ec3e930808ef4178926364634c4; stale-claim and required-responsibility guards passed; strict MkDocs completed successfully; changed-path scope and whitespace checks passed. TASK-13151 remains In Progress pending final review and PR closeout.

Quality-review Important follow-up: narrowed server recovery export to the current canonical manifest/scopes/records and explicitly excluded proposals, runtime policy, encrypted workspace mappings, keys, receipts, Sync state, and other operational state; documented that no supported server import/restore workflow exists. Distinguished owned-workspace REST scope creation and encrypted local mapping from unbound Sync-received workspace scopes, including the absence of a current mapping API for existing inbound scopes. Split future-client capability/caller/queue/status/conflict responsibilities from companion-server REST publication and purge production/distribution/ack tracking, with purge acknowledgement identified as shared cross-peer work. Narrowed profile_purge_pending to mutations reaching the existing-profile writable boundary after earlier authentication/validation/ownership/lookup gates and documented that manifest recreation is unsupported. Verification: regenerated all Published counterparts; canonical/generated and exact shared-block byte parity passed; strict MkDocs, path hygiene, and public/private boundary checks passed; the eight-file focused suite passed 78 tests with 8 warnings in 18.94s; recovery, mapping, responsibility, purge, stale-claim, source-link, seven-path scope, and whitespace guards passed. TASK-13151 remains In Progress for final review and PR closeout.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
