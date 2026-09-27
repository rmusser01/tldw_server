# PR2761 embedding-storage erasure correction

Tracking: TASK-13013.8. Release scope narrowed by the requester on 2026-09-14.

When the Chroma manager is unavailable, erasure now resolves the configured
storage path using the existing non-creating DatabasePaths helper. It returns
zero only for confirmed absence; existing, unresolved or uninspectable storage
raises a fixed error. This prevents false success for normalized paths such as
`~` without changing deletion, retention or execution behavior.

Five regression cases remain in `test_dsr_release_recovery_regressions.py`:
normalized tilde storage, absent storage, existing storage, failed inspection
and failed path resolution. Existing DSR service/API suites remain unchanged.

The broader interrupted-request recovery tests and operator procedure are
outside this release. Their historical implementation is preserved at commit
`833f43367f` in this document and the same test file for TASK-13013.8 follow-up.
This release makes no new lifecycle, backup-erasure or recovery certification.
