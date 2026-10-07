# TASK-13512 — Sidepanel copy destination ownership

Approved Stage4 continuation on latest dev1fc353, after catalog checkpoint839d7a04ee. Reuse existing useChatDraftOwner/current composer restoration and H1 and draft destination fences. No payload format, persistence authority, or account provenance claim is added.

## Stage 1: Reproduce destination races
**Goal**: Show cold restoration loss, held initial/explicit owner retirement and imported-context carryover.
**Success Criteria**: Actual receiver checks fail before production edits.
**Tests**: Existing real useSimpleForm/router receiver harness plus actual composer restoration/account checks.
**Status**: Complete

## Stage 2: Reuse owned composer boundaries
**Goal**: Wait for restored owned drafts; hold existing owner/H1 and draft destination fences around read/apply/consume; clear page context on destination retirement.
**Success Criteria**: Delayed or old callbacks preserve current text and the unread package; hydrated conflicts use existing choices.
**Tests**: Cold ready transition, owned destination change, held explicit choice, context departure, existing exact-text/TTL/route checks.
**Status**: Complete

## Stage 3: Verify and review
**Goal**: Run affected checks and independent reviews; record the bounded outcome.
**Success Criteria**: Related tests, both client gates, no-added touched lint/diff/task checks pass.
**Tests**: Receiver/services/composer suites; shared types and extension compilation. Bandit N/A unless Python source changes.
**Status**: In Progress

Source provenance/unknown-owner review, receipts, cross-origin qualified references and native docking remain parent Stage4 work.

Failure-first: four cold/retirement/context checks, five canceled-read/stale-Cancel checks, and three readiness/consume checks reproduce the distinct races. Reuse the receiver attempt Map and real composer/account/H1 guards; pass the captured guard through the service consume read to its final remove boundary.

The production Form now previews exact copied text as plain text and uses neutral guidance for a retired destination with an empty composer. Actual shared composer restoration/account watcher composition and extracted production Form callbacks/JSX supplement receiver/storage tests; no full native or packaged-extension proof is claimed.

Source review found the same canceled-conflict recovery defect on same-URL navigation and an overly broad assistant-preparation guard. Both failed before correction. Retirement is consolidated for navigation/readiness; the production capture now covers owner, mounted selected-history controller and draft IDs/address/revision, allowing unrelated file retrieval/setup changes.
