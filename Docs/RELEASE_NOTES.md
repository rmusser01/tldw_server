# Release Notes

This page is the release notes index placeholder for published versions.

## Unreleased

No published changes yet.

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
