## Stage 1: Verify and Reproduce
**Goal**: Confirm which PR #854 review findings are still valid on the current branch and capture them with focused failing tests.
**Success Criteria**: Stale findings are identified; new or existing tests fail for the still-valid redaction, probe, SSH, and unsupported-mode issues.
**Tests**: `pytest` on targeted `tldw_Server_API/tests/VLLM_Management/*` and `tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py` slices.
**Status**: Complete

## Stage 2: Security and API Hardening
**Goal**: Fix the verified security and API-surface issues with minimal, local changes.
**Success Criteria**: Sensitive values are redacted, unsafe probe/SSH inputs are rejected, and unsupported execution modes are blocked.
**Tests**: Focused `pytest` targets for API, jobs service, SSH runner, and subprocess logging behavior.
**Status**: Complete

## Stage 3: Follow-up Triage
**Goal**: Reassess the remaining open review findings after Stage 2 and decide which are already stale versus which need a second implementation slice.
**Success Criteria**: Remaining findings are documented as fixed, stale, or deferred to a clearly bounded next slice with evidence from current code/tests.
**Tests**: Focused verification only for any still-valid follow-up changes.
**Status**: In Progress
