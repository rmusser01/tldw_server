# PR 3436 Review And Merge

**Task:** TASK-15690 (formerly TASK-15493 and TASK-15466). User requests rebase onto latest `dev`, address all issues
and comments, use `superpowers:requesting-code-review` instead of Qodo, then merge.
The requester explicitly answered "waived" to the human Change-summary gate;
record that exception for this PR only, without inventing human-authored prose.
ADR assessment: existing ADR-105 governs; no new architecture decision is planned.
Preserve all functionality, analyzer behavior and existing architecture.

## Stage 1: Rebase
**Goal:** Preserve the reviewed implementation on latest fetched dev `cb70f0c5`.
**Success Criteria:** Reversible backup, no lost upstream or user work, clean rebase.
**Tests:** Range-diff and changed-path review; current task/ADR namespace remains unique.
**Status:** Complete. All 77 commits rebased without conflict; original published
head retained in the named backup. Upstream changes are frontend-only.

## Stage 2: Address Review And CI
**Goal:** Triage all discussion, reviews and failed CI; run an independent reviewer.
**Success Criteria:** No unresolved substantive findings; no Qodo request or bypass.
**Tests:** Regression-first fixes, focused CI reproductions and scoped fix review.
**Status:** Complete. The full independent review identified nine Important
findings; each was reproduced and repaired. Independent scoped repair review
approved all nine with no new findings at immutable tree `15f301f9`.
Failed CI roots were repaired without scanner/baseline relaxation: service shard
coverage, test-quality assertions/fixtures, published ADR snapshot/links, private
coercion, and singleton ownership. Bot comments contain no substantive findings;
no Qodo invocation. Technical verification/CI remain Stage 3/4 gates.

## Stage 3: Verify And Publish
**Goal:** Verify the final source, record limits and update PR3436 safely.
**Success Criteria:** Relevant regressions, native/security/consumer gates and normal
hooks pass; existing unrelated failures are proven and reported, not suppressed.
**Tests:** Appropriate pytest, Ruff/compile/format, Bandit, inventory/task/ID checks.
**Status:** In Progress. Frozen-source service verification: 118 passed, zero
skips, including official PostgreSQL cases. Actual consumers: 770 passed,
12 optional skips. Quality/workflow/docs: 288 passed. Strict MkDocs and shard
coverage pass. All 117 PR Python paths pass Ruff/compile; all 39 repair paths
pass format check. Six older unchanged paths retain format-only differences.
Endpoint-complete Bandit: eight unchanged LOW, zero MEDIUM/HIGH or scan errors.
Combined scraping run was interrupted after 3,538 passes/26 optional skips at
native enhanced-settings `/host`; isolated exact case passes in 3.21s. The full
native partition then passed 74 cases, zero skips, in 127.38s. The complementary
partition first found one remaining obsolete enhanced-policy empty-list test;
its typed-report/no-fetch correction passed 94 focused cases and independent
delta review. Final complementary rerun: 4,149 passed/28 optional or environment
skips in 193.21s. Together the disjoint partitions account for all 4,251 collected
cases; the interrupted run is not relabelled green. Publication is next.

Default-config bare pytest collected 89,397 successfully with configured importlib;
a bounded smoke was interrupted after 771 passes/12 skips. The older duplicate
Meetings collection error occurs with addopts cleared, not the default command.
No full-repository-green or diagnosed native-stall root cause is claimed.
Advisory mypy diagnostics remain disclosed; no broad typing rewrite or ignore.
Final metadata/inventory checks: 16 passed. Production remains identical to the
independently approved tree; only the reviewed test correction, generated line
references and execution receipts followed. Normal commit/push hooks and remote
required checks remain necessary; Stage 4 is not pre-approved by local results.

## Stage 4: Merge And Verify
**Goal:** Merge only the reviewed final head once repository gates permit.
**Success Criteria:** Actual GitHub merge receipt and resulting dev commit verified;
PR-specific summary waiver recorded; no unapproved check bypass or unrelated cleanup.
**Tests:** Latest base/head/check/review audit; verify merged state and merge commit.
**Status:** Not Started.

## November 13 CI Follow-Up

- Integrated dev `4b6b9ec446`, including its VN membership fixture, MCP spy
  fixture, and Ubuntu runner fixes. Resolved generated conflicts without
  changing ADR-096. Renumbered only the colliding integration record.
- Removed the runtime foundation-to-scraping import. The scraping domain
  publishes its gate in the existing service container without an inherited
  lock; checkpoint decoder tests patch only the checkpoint module's binding.
- Provision Chromium for the native integration shards and FFmpeg for the
  Auth integration shard that exercises VN runtime startup. The three other
  shard failures were HTTP 504s while downloading uv before tests ran.
- Local receipts: 351 CI-equivalent layering/workflow/scraping/MCP tests,
  12 VN cases including PostgreSQL, and 74 serial Chromium cases passed.
  The initial concurrent native run had seven failures; it is not counted
  as a pass. The task-format test must be rerun on the committed tree.
- GitHub must re-run the gates on the new head before merge.

## November 14 CI Follow-Up

1. Integrate dev `15793ad67a` without rewriting the reviewed PR history.
   Status: complete; merged locally with no conflicts.
2. Restore native subresource dispatch on CI's Playwright 1.65.2 and Chrome
   159. Keep HTTP profiles, caller credentials, and the existing guard,
   transfer accounting, and cleanup. Test: the native fixture must load
   child resources with Chromium's request-specific fetch metadata.
   Status: complete. All nine hosted failures reproduced; local A/B isolated
   the context-wide navigation metadata overrides. The shared filter gives
   Chromium ownership of fetch metadata and retains cross-origin credential
   stripping. Both the native header and mocked policy regressions were red
   before the fix. Independent review approved the final delta.
3. Verify both supported test environments, request independent review,
   then publish. Retry the unchanged Chat shard on the updated head before
   deciding whether it needs a separate change. Verify required checks and
   the actual merge on GitHub. Status: in progress. The 48 native tests pass
   on Playwright 1.65.2 and 1.58.0. The older-version run also passed 194
   other cases with only the stale generated inventory failing before refresh.
   The refreshed inventory passed all 12 tests. The updated dev backend gate
   found nine endpoint-module patches over its existing ceilings. Moved the
   stubs to service collaborators; the legacy media shim and assertions are
   unchanged. The focused endpoint/ratchet run passed 137 cases and independent
   review approved. The final backend step passed 1,129 contracts (one skip)
   and 193 WebSocket cases; production Bandit found no issues. Regenerated the
   four affected inventory line references and reran all 12 inventory tests
   successfully. Chat and hosted CI remain open.

## November 27 Rebase

1. Retain the reviewed tip locally and replay all 84 PR changes on current
   `dev`, including the edits originally made in the two merge commits.
   Status: complete at base `2053ec5ad4`.
2. Resolve shared endpoint, CI and docs changes without dropping behavior.
   Regenerate derived inventories and reserve a conflict-free identifier for
   this PR's ADR and integration task. Run the new env-read and other backend
   ratchets, native browser tests and affected service tests. Status: 349
   focused contracts and endpoint tests, including both env-read ratchets,
   and 74 native Playwright 1.65.2 tests passed. Bandit found no issues on the
   49 PR production files. Review found the Python Chromium runner was being
   validated using the frontend lockfile; pin only the affected Linux shards
   and verify the separate Python runner contract. Task and ADR IDs were
   checked against dev and open PRs, and the accepted decision is unchanged.
3. Independently review the integration delta and publish using an exact head
   lease. Arm protected auto-merge and verify the resulting commit on `dev`.
   Status: pending.
