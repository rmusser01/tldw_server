## Stage 1: Reproduce Windows portability failures
**Goal**: Add narrow regression tests that capture the Windows auth bundle path issue and the helper script console-safety requirement.
**Success Criteria**: New tests fail against the current implementation for the expected reasons.
**Tests**: `pytest tldw_Server_API/tests/Admin/test_bundle_ops.py -k windows`, `pytest tldw_Server_API/tests/Helper_Scripts/test_download_embedding_models.py`
**Status**: Complete

## Stage 2: Fix auth backup path resolution and helper output
**Goal**: Normalize Windows SQLite auth DB paths correctly and make helper progress output ASCII-safe.
**Success Criteria**: Bundle export resolves Windows auth DB paths to valid filesystem paths; helper logging does not emit non-ASCII console glyphs.
**Tests**: Same as Stage 1
**Status**: Complete

## Stage 3: Verify touched scope
**Goal**: Run focused verification for the patched behavior and summarize residual CI risk.
**Success Criteria**: Focused tests pass locally; remaining CI issues are categorized clearly.
**Tests**: targeted pytest bundle for touched files
**Status**: In Progress
