# VN pack storage preflight

[GitHub #2025](https://github.com/rmusser01/tldw_server/issues/2025) · TASK-15670

The generation preflight reports the selected pack's live, owner-attributed file registrations. It counts each registration once and reports how much of that total is referenced by approved items. The counts come from generated-file metadata. Finding unregistered or missing files on disk is separate work under #2025.

Estimate the bytes for the default all-slot generation batch from the slot dimensions, with pack defaults as fallback, and four bytes per pixel. Apply the local image-generation bounds and report how many variants lack an estimate. Actual file sizes can vary with provider output dimensions and encoding. Compare a complete estimate with the shared Storage quota service. Generation continues to enforce quota against the actual bytes it writes.

Show these readings beside the existing generation controls. Keep the configuration results available when storage is unavailable.

No new ADR is required. [ADR-064](../ADR/064-usage-quotas-per-user-limits.md) governs quota resolution and enforcement. This work adds an advisory read to the existing VN preflight.
