# PR2761 CodeQL remediation plan

**Tracking:** TASK-13013.3.1 (release parent TASK-13013.3).
**Goal:** Address every open release CodeQL finding with a verified repair or an individually reviewed, reproducible false-positive disposition.
**Source:** `28797892b1e021dad55cbc736532e702f0963ae8`; retain all release ancestry and approved legal dates.

## Stage 1: Inventory exact findings
**Goal:** Save alert identities, source/sink traces, analyzed revisions, and rule counts.
**Success Criteria:** Every alert maps to an owned investigation; mixed-revision scans are explicitly identified.
**Tests:** GitHub alert/SARIF reconciliation and source location/hash comparisons.
**Status:** Complete for initial snapshot; refresh after changes.

Initial snapshot: 443 open alerts: 241 Actions, 152 Python path injection,
18 other Python, and 32 JavaScript. Actions analysis covers the current source;
the initial Python/JavaScript analyses cover preceding `43165c8c82`.

## Stage 2: Reproduce and repair source deficiencies
**Goal:** Repair demonstrated security failures without changing unrelated behavior.
**Success Criteria:** Failing behavior tests turn green; relevant neighboring tests and scoped lint/Bandit pass.
**Tests:** Real loopback redirect tests, temporary-directory snapshot escape tests, and further source-specific boundary tests.
**Status:** Initial batch verified; rescan follow-up repairs in progress.

The current-source scan exposed operator-key persistence in the shared UAT
initializer and a session-directory alias in snapshot listing/quota. The UAT
helper now uses document-memory storage, verified against native Chromium
storage; snapshot directories reject symlinks before canonicalization.
Research writers are receiving the equivalent cross-session directory check.
Fresh scan traces remain individually reviewed before classification.

Independent ownership: Python path/file boundaries; Python HTTP/XPath/hash/regex
boundaries; frontend transport/storage/DOM boundaries; Actions event and checkout
trust boundaries. Parent integrates changes and manages source metadata and GitHub.

## Stage 3: Review each analyzer disposition
**Goal:** Distinguish unreachable analyzer flows from real or unresolved risks.
**Success Criteria:** Every proposed false-positive closure has its exact alert ID, trace-specific explanation, source hashes and executable evidence. Shared causes may share tests, but every alert is individually mapped and checked.
**Tests:** Event-specific checkout selection and real fetch-only Git workflow probes; sanitizer, persistence, hash and path containment invariants.
**Status:** In Progress.

The requester approved the 416 remaining reviewed repository-wide dispositions.
All completed and were independently reconciled with GitHub state. Together
with the earlier alert 2671, this is 417 verified dispositions (405 false
positives, 12 synthetic-test findings). See the
[disposition ledger](../../Evidence/PR2761-codeql-dispositions.md).
Real defects and main checkpoint alerts 2281/2282 remain excluded.

Do not disable queries, lower thresholds, delete legitimate behavior/tests, or
blanket-dismiss alerts. Apply individual false-positive dispositions only after
reviewing the actual trace and verifying the relevant boundary. True findings
must be repaired and rescanned. Unproven findings remain open.

## Stage 4: Integrate and verify the final candidate
**Goal:** Commit verified batches, refresh the protected-source record, and obtain current-source analysis.
**Success Criteria:** Every finding is repaired or individually resolved; all required checks refer to the final source; release PR accurately records remaining non-CodeQL gates.
**Tests:** Focused suites, lint/Bandit, source/manifest equality, required CI and complete Python/JavaScript/Actions scans.
**Status:** In Progress; source/evidence commits and protected manifest refresh underway.

No main merge or publication until the separate release gates are satisfied.
