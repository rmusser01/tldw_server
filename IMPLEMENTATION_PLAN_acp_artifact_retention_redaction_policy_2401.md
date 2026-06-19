## Stage 1: Audit Persistence And Redaction Surfaces
**Goal**: Identify where ACP sessions, messages, events, artifacts, diagnostics, audit records, and promoted workspace artifacts are stored and exposed.
**Success Criteria**: The policy update is grounded in specific code paths and separates full-fidelity owner/admin views from support-safe redacted views.
**Tests**: Focused source inspection with `rg`/`sed`; no code execution required.
**Status**: Complete

## Stage 2: Document The Release Policy
**Goal**: Add an explicit ACP retention and redaction policy to the production readiness docs.
**Success Criteria**: The policy states retention boundaries, operator controls, redaction guarantees, and known non-guarantees for artifacts, events, diagnostics, audit records, and transcript previews.
**Tests**: `git diff --check` and targeted `rg` review.
**Status**: Complete

## Stage 3: Align Operator And Certification Guidance
**Goal**: Make user/setup and certification docs point to the same policy language.
**Success Criteria**: Certification/public-evidence guidance and setup caveats avoid implying raw payload retention or blanket redaction guarantees.
**Tests**: Targeted `rg` review for stale or conflicting retention/redaction wording.
**Status**: Complete

## Stage 4: Tracker Closeout
**Goal**: Update Backlog.md and GitHub trackers with evidence and any follow-up issues required by the audit.
**Success Criteria**: TASK-2390, issue #2401, and parent #2398 record the final status and verification; implementation gaps are split out if discovered.
**Tests**: `git status`, `git diff --check`, and docs-only Bandit rationale.
**Status**: Complete
