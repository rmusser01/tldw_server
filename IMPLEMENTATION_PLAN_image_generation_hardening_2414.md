## Stage 1: Request Boundary Regressions
**Goal**: Capture the reviewed Image Generation failures with focused tests before implementation.
**Success Criteria**: Tests fail for unsafe SwarmUI URL handling, workflow request validation, blocking adapter dispatch, bounded output extraction, Stable Diffusion log redaction, and reference-image metadata reuse.
**Tests**: `tests/Image_Generation/`, `tests/Workflows/adapters/test_content_adapters.py`, and `tests/Files/test_files_image_endpoint.py` focused regressions.
**Status**: Complete

## Stage 2: Shared Image Output and Validation Helpers
**Goal**: Centralize image output validation, byte caps, and request validation so file artifacts and workflows enforce the same constraints.
**Success Criteria**: Provider output extraction rejects oversized or unknown image payloads before returning results; workflow requests reject invalid dimensions, steps, formats, and disallowed extra params.
**Tests**: Focused Image Generation adapter and workflow tests pass.
**Status**: Complete

## Stage 3: Provider Adapter Hardening
**Goal**: Harden provider-specific risks without changing public behavior for valid requests.
**Success Criteria**: SwarmUI blocks off-origin image URLs with cookies, Stable Diffusion no longer logs prompts/paths/secrets, and remote image URL extraction honors the shared output cap.
**Tests**: SwarmUI, Stable Diffusion, OpenRouter, Novita, Together, and Model Studio focused tests pass.
**Status**: Complete

## Stage 4: Reference Image Listing Efficiency
**Goal**: Avoid unnecessary reference-image storage reads and full decodes during picker listing.
**Success Criteria**: Candidate rows expose stored file size, oversized rows are skipped before storage access, and dimension probing reads image headers without forcing a full decode while preserving fallback behavior.
**Tests**: Reference-image unit tests and files reference-image endpoint tests pass.
**Status**: Complete

## Stage 5: Verification and Task Closeout
**Goal**: Verify focused behavior, security scan touched code, and record results in Backlog.
**Success Criteria**: Focused tests and Bandit complete with no new findings in touched scope, and `TASK-2414` has final notes.
**Tests**: Focused pytest commands plus Bandit over touched source files.
**Status**: Complete
