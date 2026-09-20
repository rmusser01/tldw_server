---
id: ADR-002
title: Character-conversation behavior snapshot and fenced completion authority
date: '2026-08-27'
status: Accepted
---

# ADR-002: Character-conversation behavior snapshot and fenced completion authority

## Context

Character conversations currently depend on mutable character cards and related
records when assembling completion prompts. A client that reopens an older
conversation therefore cannot prove that the server will reproduce the behavior
that was active when the conversation was created. Treating a current card as a
historical substitute would silently change character behavior, while copying the
conversation into a client-owned chat would split authority and lose server-side
concurrency guarantees.

The Chatbook Roleplay design requires an exact, capability-gated resume path. Its
approved downstream contract is documented in
`rmusser01/tldw_chatbook` ADR-095 and
`Docs/superpowers/specs/2026-08-27-server-backed-roleplay-conversation-resume-design.md`.
TASK-13159 provides the stored historical behavior prerequisite; TASK-13160
provides the versioned completion and persistence contract.

## Decision

### Snapshot authority

- Creating a resume-eligible character conversation MUST atomically persist one
  immutable, schema-versioned behavior snapshot for all participants or roll back the
  conversation creation. An intentionally unsupported creator may instead persist an
  explicit `missing`/non-resumable record only when it cannot advertise readiness and
  contract append rejects before writing.
- The snapshot is the sole character-behavior authority for resumed completion.
  It materializes behavior-affecting values that would otherwise be read through
  mutable references, including participant/card prompt fields, prompt and
  generation presets, exemplars, lore/world-book state, memory/note context, and
  conversation behavior settings.
- Credentials, provider secrets, portrait or attachment binaries, and other
  non-behavioral or secret material MUST NOT be stored in the snapshot.
- Canonical serialization, a schema version, a digest, and an enforced size bound
  make the snapshot auditable and safe to validate before use.
- Existing conversations without a valid snapshot remain explicitly legacy and
  preview-only. The server MUST NOT silently backfill them from current mutable
  sources.

### Mutable settings and fences

- Behavior-affecting conversation settings remain server-owned. Applying a
  supported mutation materializes the resulting behavior values and increments a
  monotonic `settings_version`.
- The atomic creation factory initializes settings version 1 with the effective
  provider, model, and closed set of explicit sampling values resolved at creation.
  Credentials and availability remain runtime state. If those effective values cannot
  be validated, the conversation is explicitly non-resumable and append fails before
  writing; later deployment defaults are never used as a historical substitute.
- Every successful message-history mutation, including add, edit, soft/hard delete,
  restore, branch/tail selection, and assistant insertion, advances one monotonic
  conversation `history_version` in the same transaction. Tail identity/version
  alone is not a complete history fence because an earlier ancestor can change.
- Conversation detail exposes snapshot status, snapshot schema version and
  digest, `settings_version`, `history_version`, and authoritative message-tail
  identity/version fences.
- The first downstream Chatbook slice treats remote behavior settings as
  read-only. The server contract nevertheless versions supported external
  mutations so clients can detect and refresh after drift.

### Resume completion contract

- Support is proven only by authenticated capability discovery advertising
  `roleplay_resume_contract_version >= 1` plus the required base features
  `snapshot_completion`, `fenced_completion`, `idempotent_user_append`, and
  `nonstream_assistant_persist`. Route or OpenAPI presence is not sufficient.
- A client first appends a user message using a caller-selected idempotency ID.
  Repeating an identical append returns its authoritative result; conflicting
  reuse fails with a structured conflict. The initial append also compare-and-swaps
  expected snapshot digest, settings version, prior tail ID/version, and
  `history_version`; drift or non-resumable policy returns a structured conflict and
  inserts no user row or branch.
- Snapshot completion accepts no implicit prompt append and no current-card,
  local, or request-time behavior override. Before provider dispatch it verifies
  the expected snapshot digest, settings version, exact input user-message ID,
  authoritative tail ID/version, and complete `history_version`.
- A server-issued generation fence binds the provider call to that verified
  state, including `history_version`. Assistant persistence MUST compare-and-swap
  the same fence after generation.
- If the conversation changes during generation, no assistant is persisted and
  no branch is created. Non-streaming responses return the generated text with
  `saved: false`; streaming clients retain already-delivered text and receive the
  same structured unsaved outcome.
- Non-streaming assistant persistence is part of the base contract. Streaming
  persistence is available only when `stream_assistant_persist` is separately
  advertised; otherwise clients use non-streaming completion.
- Stream persistence requires a short-lived, domain-separated HMAC-SHA256 grant
  emitted after generation and bound to authenticated owner, scope, conversation,
  exact input/parent, stable assistant ID, the complete generation fence, and final
  content digest. Signing keys come only from a dedicated server-only current/secondary
  Roleplay secret; the client-known single-user API key, public keys, defaults, and
  placeholders are never signing material. Without valid server-only material the
  stream feature is not advertised. Persist verifies the signature and every binding before CAS;
  identical replay is idempotent, while tamper, expiry, or cross-context reuse fails
  without revealing another user's record.
- Successful persistence returns stable authoritative assistant identity and its
  authoritative user parent. Missing/invalid snapshot, policy rejection, drift,
  unknown outcome, and validation-degraded results are structured and fail
  closed.
- All production character-conversation creation paths use one atomic snapshot
  factory or explicitly create a non-resumable `missing` record. Active Sync-origin
  paths remain non-resumable until their server-origin mutation and fenced
  completion semantics are atomic; per-conversation readiness must reject Resume
  before a user append could commit to such a conversation.

## Consequences

- Historical character behavior becomes reproducible without keeping mutable
  card records alive or trusting a client-side copy.
- Conversation state has one authority: the server. Clients may project a bounded
  transcript for display, but may not create a durable local shadow or fall back
  to local completion.
- Conversation creation and supported behavior-setting mutations become more
  expensive because they materialize and validate snapshot content.
- Legacy conversations cannot be resumed through this contract unless a future,
  explicitly designed migration supplies trustworthy historical data.
- Completion needs two conflict checks: one before provider cost is incurred and
  one before the assistant is committed. A concurrent mutation may therefore
  produce useful but deliberately unsaved text.
- Caller-selected user append adds an earlier compare-and-swap so a concurrent
  writer cannot turn the intended continuation into an implicit sibling branch.
- The capability version allows additive evolution without downstream route
  guessing; incompatible changes require a new contract version or feature gate.

## Alternatives considered

- **Read the current character card on resume.** Rejected because it violates the
  historical-behavior guarantee after edits or deletion.
- **Silently snapshot legacy conversations when first resumed.** Rejected because
  present-day sources cannot prove historical behavior.
- **Copy a remote conversation into a local chat.** Rejected because it creates
  dual ownership and makes remote concurrency and persistence ambiguous.
- **Check versions only before generation.** Rejected because state may change
  during a provider call and cause an assistant to be attached to the wrong tail.
- **Fence only the tail row.** Rejected because editing or deleting an earlier
  message changes provider history without changing the tail identity/version.
- **Always persist generated text by creating a branch after conflict.** Rejected
  because branching is not part of the approved contract and would hide drift.
