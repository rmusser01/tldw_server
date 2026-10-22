# C04 workspace agent handoff cookie transport repair

Task: TASK-13260.278.18.83.45.

The first real quickstart Create agent task failed with CSRF token validation failed. The handoff uses direct fetch while the existing ACP transport supplies cookie-session CSRF for unsafe methods. Reuse that transport for POST and rollback DELETE; preserve payloads, errors, direct credentials and hosted behavior. Keep backend CSRF enforcement unchanged.

ADR check: ADR required: no. This repairs an omitted use of the established ACP cookie transport; it changes no security or workspace policy. Existing ADR-007 governs the canonical Research workspace handoff.

## Stage 1: Reproduce the omitted header
**Goal**: A rendered handoff reaches the real shared browser transport boundary.
**Success Criteria**: Cookie-backed POST and rollback DELETE fail on current source; direct and hosted controls remain meaningful.
**Tests**: Actual handoff component with cookie CSRF server-boundary assertions and success/rollback paths.
**Status**: Complete

## Stage 2: Reuse the established ACP transport
**Goal**: Supply current CSRF on cookie writes with existing credential routing.
**Success Criteria**: New causal cases and existing handoff/client/connection tests pass in UI and WebUI configurations; lint and syntax clear, original contracts retained.
**Tests**: Targeted causal tests, complete affected handoff and ACP transport suites.
**Status**: Complete

## Stage 3: Independent review and live recovery
**Goal**: Review the exact combined patch and validate the normal handoff on the corrected build.
**Success Criteria**: Source review clear, first failed submit retained, ONE declared corrected-build recovery creates the bound task, real run/result and retained history qualified within existing workflow budgets.
**Tests**: Root normal browser handoff/run after combined reviewed build; no auth injection or unchanged resend.
**Status**: In Progress

Causal baseline: configured UI six controls, 3 failed and 3 passed, natural exit 1, 4.254 seconds outer. Both cookie handoffs were refused by the CSRF boundary; rollback DELETE lacked cookie credentials/header. Direct API-key, JWT and hosted controls passed. No setup failure or live retry.

Initial fixed check: single-user cookie POST/DELETE and direct/hosted controls passed, but an authored multi-user cookie expectation failed. The actual browser-networking contract deliberately limits cookie-session transport to single-user quickstart; multi-user uses JWT. Corrected that invalid new fixture to a supported quickstart JWT compatibility control; no production security policy changed. Only single-user POST and rollback DELETE are valid causal defects.

Final scoped checks: six supported causal/compatibility UI controls passed (5.509 seconds outer); original handoff/client/connection UI 87 passed (27.311 seconds outer), 93 distinct checks across split runs. Configured WebUI four files 93 passed together (41.178 seconds outer). Scoped ESLint current/baseline production and new test: zero errors, zero warnings, no ignored files; both TSX syntax parse clean. Inherited Next pages discovery diagnostic disclosed. TypeScript-only change: Bandit not applicable. New test formatting preserved its meaningful TS syntax tree (formatter parentheses and blank JSX whitespace normalized). Backend/shared ACP source and all existing tests unchanged. Independent review and corrected-build live recovery still open.

Operator-only corrections: the first formatter guard included SourceFile.text and so rejected formatting before any test write; fixed comparator excludes source text and normalizes only parentheses/blank JSX whitespace. An initial task append lacked PYTHONPATH and failed before writing; the supported invocation below succeeded. No test/application/live action followed either operator error.
