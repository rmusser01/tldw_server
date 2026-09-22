# PR2979 latest dev integration

Task: TASK-13260.278.18. Native/full UAT remains paused.

## Stage 1: Preserve and rebase
**Goal**: Preserve the PR branch and rebase onto dev8045fa2956f22a5bb95ccba113dc7236f17e62de.
**Success Criteria**: Recovery ref exists, original checkout unchanged, upstream share-link security/RAG Notes/CI changes coexist with UAT repairs.
**Tests**: Clean index/status, ancestry and before/after range-diff.
**Status**: In Progress

## Stage 2: Verify affected behavior
**Goal**: Check the overlapping Chat, RAG and workflow code.
**Success Criteria**: Focused suites pass, PostgreSQL is exercised through the official fixture where applicable, no new security findings.
**Tests**: Share-token security, slides Notes retrieval, UAT Chat integration, workflow and frontend guard tests; Bandit on overlapping Python scope.
**Status**: Not Started

## Stage 3: Publish the verified branch
**Goal**: Update PR2979 with the rebased branch and accurate evidence.
**Success Criteria**: Current browser CI evidence retained before publishing; push uses explicit lease; generated captures excluded; PR remains explicit about unresolved gates.
**Tests**: Remote head/base identity, tracked-file audit, CI/check review.
**Status**: Not Started
