# Release Notes

This page is the release notes index placeholder for published versions.

## Unreleased

### Presentation Studio

- Added standalone HTML + JavaScript presentations as a separately gated,
  default-off project kind. Generation runs asynchronously through Jobs and an
  administrator-selected built-in provider adapter. Saved projects remain
  readable when generation or provider egress is disabled.
- The WebUI edits the complete document as inert text, offers a bounded
  text-only Safe outline, uses explicit strong-ETag saves, and downloads exact
  bytes as an attachment. It never previews or executes the document. Opening
  a downloaded HTML file outside tldw can execute its JavaScript and should be
  treated accordingly.
- The browser extension remains source-free for standalone projects and offers
  a metadata-only handoff to the canonical WebUI. Operators should complete the
  schema-v2 backup and default-off rollout steps in
  `Docs/Deployment/Standalone_HTML_Presentations.md` before enabling
  generation.

## 0.1.41 - 2026-07-16

### Research and source-grounded learning

- Added source-grounded spaced repetition and advanced quiz controls, plus
  Paperless source review and saved-view workflows.
- Expanded the Research Workspace foundations for source discovery and PDF
  handoff, with beginner UAT coverage, workspace status rails, and current
  decision and reconciliation records.

### Skills and service prompt groundwork

- Added Skills MCP catalog rendering, render-tool binding, WebUI and extension
  parity, and focused Skills UAT quality gates.
- Documented the reviewed groundwork for user-customizable service prompts as
  follow-on work rather than presenting it as a completed user feature.

### Operations and reliability

- Hardened Quick Ingest advanced transport when no server URL is persisted,
  default and render-loop behavior, preset provider selection, and E2E
  analysis-provider guards.
- Closed Chatbooks post-merge UAT issues, strengthened Watchlists briefing
  contracts across backend and shared product surfaces, and added Jobs
  admission and operations extraction.
- Fixed Parakeet ONNX feature configuration, extension launch and cancellation
  races, frontend CodeQL paths, and related release-review regressions.

### Security, authentication, and provider egress

- Improved single-user API-key device persistence, media authentication refresh
  behavior, metadata-only ingestion guards, and stale-session recovery after a
  WebUI relaunch.
- Centralized checked outbound access for configured local providers and public
  custom OpenAI-compatible adapters.

## 0.1.40 - 2026-07-10

### Chatbooks

- Chatbooks Backup & Import now treats omitted `content_selections` or `content_selections: {}` as a full user-account export. Non-empty selections are explicit allowlists, and zero-item allowlists are rejected.
- Full-account exports are driven by the account-data inventory and include media records, derived media data, bundled tldw-stored account file artifacts, embeddings, settings, prompts, evaluations, generated documents, and sensitive values required for restore. External source URLs/paths remain pointer metadata unless tldw stored the bytes.
- Chatbook archive imports now restore all restorable archive data present by default. OpenWebUI imports remain reference-first and use the OpenWebUI attachment hydration workflow for copied images/files.
- The WebUI and browser extension now use the primary Backup all flow with a redacted scope summary, while Settings links to Chatbooks Backup & Import instead of acting as the full backup/restore workflow.

### Character Chat

- Resolved the Character Chat GA database-health release dependency tracked by
  `TASK-429` and PR #1862: corrupt per-user `ChaChaNotes` databases are reported
  through sanitized health metadata, startup warm-up fails open, and operators
  are directed to `Docs/Operations/ChaChaNotes_DB_Recovery.md` for recovery
  steps. This removes the release-blocking R11 backend recovery gap for
  first-class character chat.

### Release hardening

- Fixed Guardian notification timestamp handling, visual identity ZIP unsafe-path
  detection on Windows, legacy audio WebSocket config handling, macOS PyAudio CI
  setup, sandbox concurrency test stability, CodeQL alert paths, and media
  ingest worker startup defaults.

For release process details, see `Docs/Release_Checklist.md`.
