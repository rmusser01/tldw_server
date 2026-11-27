# VN asset retention and orphan detection

[GitHub #2025](https://github.com/rmusser01/tldw_server/issues/2025) · TASK-15671

The owner can ask the existing cleanup endpoint for an age-aware dry run. Rejected items become eligible after 30 days and hidden items after 7 days. The clock starts at the latest item update, with creation as a fallback. Drafts, approved items, and items without a trustworthy age stay protected in this preview. Active generation, restores, and exports defer the preview for the affected pack. Existing shared-file, published script, and play-session checks still apply. The Visual Identity import path stores its own copy under the owner's Visual Identity storage.

An owner-scoped storage report pages through items, registrations, or generated image files. It distinguishes a missing registration, missing bytes, a registration with no known VN reference, and bytes with no registration or live item. A 24-hour settling period and active-job deferral protect in-flight writes. Trash, soft-deleted packs, and runtime references still count as evidence of ownership or use. File scans follow the VN image writer's filename convention and skip symbolic links and preview archives.

The report and age-aware preview only inspect data. Existing explicit cleanup remains available for reviewed choices and keeps its existing idempotency contract.

ADR assessment follows [ADR-020](../ADR/020-db-management-per-user-paths-and-content-backend.md) for paths and [ADR-094](../ADR/094-jobs-resource-admission-and-domain-publication.md) for publication boundaries. This change adds diagnostics over those sources and requires no new ADR.
