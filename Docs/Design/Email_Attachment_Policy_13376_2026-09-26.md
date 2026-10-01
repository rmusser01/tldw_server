# Email attachment extraction policy (TASK-13376.3)

The supported extraction path is attached EML messages. PDF, image, audio,
document and arbitrary binary attachments remain metadata-only. ZIP and MBOX
expand root messages and apply the same policy to each message. PST/OST messages
are parsed when optional pypff is available; its attachment descriptors remain
metadata-only because the adapter does not export their content. Missing pypff
continues to return the deterministic existing unsupported-format error.

## Contract

- Existing `ingest_attachments=false` remains the default. Omitted
  `extract_attachments` inherits that flag; explicit true enables nested EML
  extraction and explicit false disables it even when the legacy flag is true.
- Omitted `attachment_mime_allowlist` defaults to `message/rfc822`, including
  the existing `.eml` filename compatibility path. Explicit lists select declared MIME
  types and disable filename inference; an empty list allows none. `attachment_mime_denylist` defaults empty.
  Deny rules win over allow rules and the filename fallback. Exact MIME types,
  `type/*` and `*/*` are supported, normalized case-insensitively.
- Every descriptor retains its name, declared MIME type, size and available
  content-id/disposition. Additive extraction status/reason fields explain
  disabled/depth/selection/unsupported/size decisions. Binary bytes are never
  persisted. Metadata-only mode avoids nested serialization, eager payload decoding and child
  processor hooks. Parent-message analysis retains its existing separate toggle.
- Nested attachment bodies must not enter the parent body through MIME walking.
- Existing depth (API range 1..5), archive member-count, total uncompressed size,
  per-member size, unsafe-path and encrypted-ZIP checks remain. Selected nested
  EMLs additionally honor archive member byte/count limits before processing.
- Child processing continues to disable analysis. Enabling extraction does not
  enable chunking, embeddings, claims, or analysis.

Form fields are shared by `/media/add` and `/media/process-emails`. MIME fields
accept repeated fields or comma-separated values. Unsupported MIME selections
retain metadata and never dispatch a generic binary processor.

## Alternatives

Extending existing flags and functions preserves the current API and processor
surface with a small, pure normalization/selection helper. A registry of binary
processors or model extraction would add unsupported behavior and dependencies;
those are outside this change. A new policy class is unnecessary.

## Stage 1: Policy behavior tests
**Goal**: Synthetic EML, ZIP, MBOX and fake PST tests cover policy and hook safety.
**Success Criteria**: New assertions fail because policy behavior is missing.
**Tests**: Metadata-only interception, MIME allow/deny and filename fallback,
body isolation, depth/size/count limits, PST degradation.
**Status**: Complete

## Stage 2: Minimal implementation and propagation
**Goal**: Add flags, descriptors and normalization to existing parser/form flow.
**Success Criteria**: Policy tests pass; legacy defaults and EML recursion remain.
**Tests**: New tests and existing parser/form/endpoint tests.
**Status**: Complete

## Stage 3: Verification and handoff
**Goal**: Record focused tests, Ruff, Bandit and integration ownership details.
**Success Criteria**: Passing evidence and precise remaining parent integration.
**Tests**: Shared venv and worktree PYTHONPATH; no Gmail or model calls.
**Status**: Complete


## Verification and integration evidence

- Policy TDD baseline: 25 failures / 1 existing compatibility pass before changes;
  metric-hook baseline: 6 failures before adding hooks.
- Focused policy/parser suite: 42 passed (20.20s), including multipart option
  propagation, fake PST attachment read interception and container metric formats.
- Expanded parser/policy/process-emails/logging run: 76 passed, 2 skipped,
  1 deselected and 1 failure (162.93s). The failure is the logging task's new
  rollback test passing unsupported `metadata=` to the Media DB method; its owner
  was notified to correct that test. The deselection is the separate logging task's
  upload/persistence privacy regression awaiting the parent integration. All
  attachment-policy, parser and process-emails tests passed.
- Real PST tests skip because pypff and/or PST_FIXTURE_PATH are unavailable.
  Synthetic PST descriptor behavior and the missing-dependency error are tested.
- Ruff passed on the seven owned Python files; Bandit found zero issues across
  the six owned production files. `git diff --check` passed. Evidence logs are
  `/tmp/email_attachment_policy_green_13376.log`,
  `/tmp/email_attachment_final_13376.log` and
  `/tmp/bandit_email_attachment_13376.json`.
- Parser metrics record `eml` parsed/error only around `parse_eml_bytes`; archive
  delegates produce no aggregate success observations. Container guard/read/
  extraction failures record zip/mbox/pst/ost error durations separately.
- INFO+ parser failures emit static events and bounded exception types using the
  logging task's helper; filenames, payloads and tracebacks are excluded.
- Parent owns persistence.py option propagation and the effective extraction flag
  for `/media/add` child persistence, plus combined verification and commits.
  Parser/form ownership has been released. No provider/model calls or commits
  were made by this implementation task.
