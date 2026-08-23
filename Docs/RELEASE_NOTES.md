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

### Character Chat

- Resolved the Character Chat GA database-health release dependency tracked by
  `TASK-429` and PR #1862: corrupt per-user `ChaChaNotes` databases are reported
  through sanitized health metadata, startup warm-up fails open, and operators
  are directed to `Docs/Operations/ChaChaNotes_DB_Recovery.md` for recovery
  steps. This removes the release-blocking R11 backend recovery gap for
  first-class character chat.

For release process details, see `Docs/Release_Checklist.md`.
