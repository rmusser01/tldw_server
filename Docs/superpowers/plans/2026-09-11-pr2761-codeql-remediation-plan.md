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
**Status:** Initial and rescan follow-up repairs verified; hosted rescan pending.

The current-source scan exposed operator-key persistence in the shared UAT
initializer and a session-directory alias in snapshot listing/quota. The UAT
helper now uses document-memory storage, verified against native Chromium
storage; snapshot directories reject symlinks before canonicalization.
Research writers have the equivalent cross-session directory check. Whisper
again treats tilde-prefixed model input literally under its managed root, with
no OS account lookup. Shared containment comparisons preserve original path
spelling; checkpoint lexical checks precede filesystem resolution and retain
the canonical postcheck. Fresh scan traces remain individually reviewed before
classification.

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
Three later PR-only false positives, 2675/2676/2679, were independently reviewed
and resolved with committed proof. The ledger totals 420 dispositions. New real
rescan findings 2673/2674/2677/2678 were repaired, not dismissed.

Do not disable queries, lower thresholds, delete legitimate behavior/tests, or
blanket-dismiss alerts. Apply individual false-positive dispositions only after
reviewing the actual trace and verifying the relevant boundary. True findings
must be repaired and rescanned. Unproven findings remain open.

## Stage 4: Integrate and verify the final candidate
**Goal:** Commit verified batches, refresh the protected-source record, and obtain current-source analysis.
**Success Criteria:** Every finding is repaired or individually resolved; all required checks refer to the final source; release PR accurately records remaining non-CodeQL gates.
**Tests:** Focused suites, lint/Bandit, source/manifest equality, required CI and complete Python/JavaScript/Actions scans.
**Status:** In Progress; source/evidence commits and protected manifest refresh underway.

The verified follow-up source is `3f9866a860033b70b3434319fadfdd37b12819a2`.
Its protected manifest covers 7,117 files, SHA-256
`e38f39788bdbf6ee691a27cc91357e0d92a6b21a47355d745409938ea6e66f76`.
Release date and Countdown start are unchanged. Validation includes 61 UAT
tests plus the real Chromium storage proof, 105 Whisper tests, 150 combined
path tests, independent review and scoped lint/security checks; counts overlap.
Final-source hosted analysis remains necessary before claiming alert closure.

The `009505c415` JavaScript scan cleared real UAT storage findings. Two synthetic
regression writes were independently resolved, bringing the ledger to 422.
The final Python follow-up closes a Windows case-only sibling escape and
preliminary filesystem probes: compare original canonical spelling, require
exact lexical root spelling before candidate probes, check links parent-first,
and retain canonical checks. Absolute/root case aliases fail closed by design.
The combined suite passes 162 tests; Ruff and scoped Bandit pass. Native Windows
execution remains unverified. This follow-up needs its own source/manifest
binding and hosted scan; no query or model suppression is used.

No main merge or publication until the separate release gates are satisfied.

Final verified path source: `968ad1aaf95fccac966cfb31a8ac981508befb88`.
Independent review passes 36 focused cases with matching frozen file hashes.
The 7,117-file protected manifest remains byte-identical; source record and test
pins are rebound to this commit. Legal dates/digests are unchanged. Hosted
analysis must now verify this final source, including the earlier unrecognized
main-shared paths; no global dismissal of their actual main defects is used.
