# VN asset cleanup audit

[GitHub #2025](https://github.com/rmusser01/tldw_server/issues/2025) · TASK-15672

A confirmed cleanup keeps its owner, pack, selection, and outcome in the existing cleanup receipt. The API commits a checkpoint before each deletion attempt. Requests using previously completed idempotency keys still replay their stored results. Previews leave no history entry.

The result distinguishes confirmed registration removal, item removal, and physical unlink. If an operation raises, cleanup stops and records the affected item, file, and failed stage. The same request replays that partial result. When the process stops or a checkpoint cannot be completed, the history shows the last durable progress and any item whose final state is uncertain. The original key remains in progress so it cannot trigger another deletion attempt.

An owner-scoped, paginated history remains available after a pack is soft-deleted. Prior completed receipts are listed with their stored results. An admission interrupted before its first checkpoint appears with its owner and pack but has no selection or result yet. The recorded selection includes only a bounded list of valid item IDs. The history omits file paths, raw errors, confirmation strings, and idempotency keys.

The design follows the owner-local database boundary in [ADR-020](../ADR/020-db-management-per-user-paths-and-content-backend.md) and the durable domain receipt boundary in [ADR-094](../ADR/094-jobs-resource-admission-and-domain-publication.md). It extends the existing receipt lifecycle and requires no new ADR.
