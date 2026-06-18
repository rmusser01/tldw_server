# Workspace Container Contract Plan

Date: 2026-06-17
Task: TASK-2380
GitHub: https://github.com/rmusser01/tldw_server/issues/1988

## Stage 1: Existing Contract Survey
**Goal**: Ground the Workspace container contract in current tldw_server and tldw_chatbook vocabulary.
**Success Criteria**: Existing backend models, membership types, active-context eligibility, ACP bridge surfaces, and Chatbook operating-context fields are identified.
**Tests**: Read-only inspection of Workspace docs, Workspace core code, ACP bridge code, and Chatbook workspace PRD/foundation files.
**Status**: Complete

## Stage 2: Canonical Contract Document
**Goal**: Add a docs-only canonical Workspace container contract for Phase 2 implementation.
**Success Criteria**: The contract defines Workspace identity, lifecycle, metadata, archive/delete semantics, authority/status, import provenance, membership semantics, transfer policies, runtime binding vocabulary, active context rules, global browsing rules, and mappings to existing backend/frontend surfaces.
**Tests**: Markdown inspection and `git diff --check`.
**Status**: Complete

## Stage 3: Linkage And Tracking
**Goal**: Make the contract discoverable from existing Workspace docs and tracking records.
**Success Criteria**: The Workspaces core README and canonical-model decision point to the new contract; TASK-2380 records touched files and verification.
**Tests**: `rg` checks for the new contract path and required acceptance phrases.
**Status**: Complete

## Stage 4: Verification And PR
**Goal**: Verify docs-only changes and open a PR against `dev`.
**Success Criteria**: Docs checks pass, Bandit skip is documented because no backend code changed, branch is committed and pushed, and the PR links/closes issue #1988.
**Tests**: `git diff --check`; targeted `rg` acceptance checks.
**Status**: Complete

## Verification

- `git diff --check`
- `rg` acceptance check for Chatbook reference, global visibility, active-context eligibility, Phase 2 resource types, transfer policies, runtime binding follow-ups, and assigned child issues.
- Bandit: not run because this is a docs-only Backlog/Markdown change with no Python code touched.
- PR: https://github.com/rmusser01/tldw_server/pull/2381
