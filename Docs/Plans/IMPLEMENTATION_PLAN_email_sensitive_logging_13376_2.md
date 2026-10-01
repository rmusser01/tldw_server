# Email sensitive logging plan (TASK-13376.2)

## Stage 1: Audit and reproduce
**Goal**: Identify email-sensitive INFO+ log values and prove leaks with real Loguru capture.
**Success Criteria**: Upload/parser/persistence/native database/search success and failure paths have synthetic sentinel coverage.
**Tests**: Filename/body/header/credential/metadata sentinels; exception text echoes data.
**Status**: Complete

## Stage 2: Minimal safe diagnostics
**Goal**: Replace sensitive values with static event outcomes, counts and bounded exception class names.
**Success Criteria**: No sentinel in captured output, exception object/extras or diagnostic tracebacks; operational failures remain visible.
**Tests**: Red-green logging regression cases and existing email suites.
**Status**: Complete

## Stage 3: Verify and document
**Goal**: Record scoped audit, test/Ruff/Bandit evidence and integrate shared-file findings.
**Success Criteria**: Tests and security checks pass; audit names scope and limits; Backlog task current.
**Tests**: Focused suite, Ruff, Bandit.
**Status**: Complete
