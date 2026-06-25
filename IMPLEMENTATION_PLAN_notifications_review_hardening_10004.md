## Stage 1: Regression Coverage
**Goal**: Add focused tests for the validated Notifications review findings.
**Success Criteria**: Tests fail against current code for unbounded email delivery, unsafe failure details, and email helper timeout/delegation behavior.
**Tests**: `tldw_Server_API/tests/Notifications/test_notifications_service.py`, `tldw_Server_API/tests/Notifications/test_email_delivery.py`
**Status**: Complete

## Stage 2: Service Boundary Hardening
**Goal**: Bound and sanitize `NotificationsService` email delivery inputs and failure outputs.
**Success Criteria**: Recipient fanout, attachments, filenames, and error details are constrained before delivery.
**Tests**: Focused Notifications service tests.
**Status**: Complete

## Stage 3: Email Helper Consolidation
**Goal**: Remove the duplicate raw SMTP sending path from `email_delivery.py` while preserving the public helper.
**Success Criteria**: `send_notification_email()` delegates to AuthNZ email service and SMTP config exposes an explicit timeout for compatibility checks.
**Tests**: Focused email delivery tests.
**Status**: Complete

## Stage 4: Documentation and Verification
**Goal**: Align README scope and record verification.
**Success Criteria**: README describes actual core package ownership; targeted tests and Bandit pass or blockers are documented.
**Tests**: Notifications tests and Bandit on touched production files.
**Status**: Complete
