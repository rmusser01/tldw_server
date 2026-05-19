# WebClipper Review Fixes Implementation Plan

## Stage 1: Regression Coverage
**Goal**: Capture the confirmed WebClipper bugs in backend unit and integration tests before changing behavior.
**Success Criteria**: New tests fail against the current implementation for visible-body preservation, visible-vs-full-extract selection, workspace-note resync, keyword convergence, truncation messaging, and fatal-save HTTP status handling.
**Tests**:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py`
**Status**: Complete

## Stage 2: Service and Endpoint Fixes
**Goal**: Update `WebClipperService` and the WebClipper endpoint to preserve clip body fidelity, keep workspace/canonical state in sync, and surface fatal canonical save failures correctly.
**Success Criteria**:
- Enrichment writeback preserves full visible body.
- Save uses `visible_body` as the note body when provided.
- Re-saving updates the existing workspace note and converges note keywords.
- Truncation messaging no longer falsely claims a missing attachment.
- Fatal canonical save failures no longer return HTTP `200`.
**Tests**:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py`
**Status**: Complete

## Stage 3: Verification and Security Check
**Goal**: Re-run the focused WebClipper suite and check the touched Python scope with Bandit.
**Success Criteria**: Focused tests pass, and Bandit is either clean or any environment limitation is reported explicitly.
**Tests**:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_db.py`
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/WebClipper/service.py tldw_Server_API/app/api/v1/endpoints/web_clipper.py -f json -o /tmp/bandit_webclipper_review_fixes.json`
**Status**: Complete
