## Stage 1: Local Tool Configuration
**Goal**: Confirm OpenCode and Aider are configured to use the local llama.cpp OpenAI-compatible server at `127.0.0.1:9099`.
**Success Criteria**: Both tools can resolve the configured local model, and direct smoke checks either complete or record a concrete blocker.
**Tests**: `curl /v1/models`, `opencode models llama.cpp`, Aider configuration smoke command.
**Status**: Complete

## Stage 2: ACP Entrypoint Certification
**Goal**: Determine which installed tools expose an ACP-compatible stdio entrypoint and run certification where possible.
**Success Criteria**: OpenCode and Aider each have explicit ACP evidence: pass results, or a documented blocker when no ACP entrypoint exists.
**Tests**: `acp_certification_smoke.py --agent-profile <profile> --run` and backend live-E2E for any profile that passes direct stdio certification.
**Status**: Complete

## Stage 3: Repository Metadata and Docs
**Goal**: Update shipped compatibility metadata, docs, and task notes to match the verified evidence.
**Success Criteria**: `agents.yaml`, the compatibility matrix, and TASK-498 describe the current OpenCode/Aider state without overstating support.
**Tests**: Focused ACP registry/helper tests and docs/metadata checks.
**Status**: Complete

## Stage 4: Final Verification
**Goal**: Verify the touched scope and prepare the branch for review.
**Success Criteria**: Focused tests, relevant runtime checks, Bandit/doc skip rationale, and `git diff --check` are recorded.
**Tests**: Focused pytest set, Bandit if Python code is touched, `git diff --check`.
**Status**: Complete
