# Release Notes

This page is the release notes index placeholder for published versions.

## Unreleased

### Character Chat

- Resolved the Character Chat GA database-health release dependency tracked by
  `TASK-429` and PR #1862: corrupt per-user `ChaChaNotes` databases are reported
  through sanitized health metadata, startup warm-up fails open, and operators
  are directed to `Docs/Operations/ChaChaNotes_DB_Recovery.md` for recovery
  steps. This removes the release-blocking R11 backend recovery gap for
  first-class character chat.

For release process details, see `Docs/Release_Checklist.md`.
