## Stage 1: Verify Review Findings
**Goal**: Separate still-valid PR #1907 review findings from findings already fixed by merged PR #1906.
**Success Criteria**: Each actionable item is mapped to current `dev` evidence.
**Tests**: Source inspection against current branch.
**Status**: Complete

## Stage 2: Patch Watchlists Status Truth
**Goal**: Preserve audio-specific status fields, count status-only failed audio outputs, and keep health fixtures typed.
**Success Criteria**: Audio status UI and overview attention counts cannot be masked by generic report status or missing request flags.
**Tests**: Watchlists output metadata and overview service tests.
**Status**: Complete

## Stage 3: Patch Backend Detail And Redaction Safety
**Goal**: Preserve legacy flat filter counters and redact JSON-style source secrets before source errors are stored.
**Success Criteria**: Run details retain real filter counts; source error sanitizer removes JSON key/value secrets.
**Tests**: Watchlists run detail and operator recovery tests.
**Status**: Complete

## Stage 4: Metadata Cleanup And Verification
**Goal**: Normalize Backlog metadata and run targeted frontend/backend/security checks.
**Success Criteria**: TASK metadata parses consistently; verification results are recorded in TASK-468.
**Tests**: Targeted Vitest, targeted pytest, Bandit on touched backend files.
**Status**: Complete
