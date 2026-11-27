# VN asset runtime integrity

[GitHub #2024](https://github.com/rmusser01/tldw_server/issues/2024) · TASK-15677

Use the existing owner-local content rules to check the generated file behind a VN asset. Check the registration against the owner and item, require supported file metadata and policy access, and resolve the file within the owner's outputs. A regular, nonempty file must match the registered size when a positive size was recorded. Legacy registrations can use the item's MIME type and may lack a recorded size.

Single and bulk approval check the selected items, then compare their file references during the review write. A changed reference aborts the review transaction. The v1 manifest endpoint checks the entries emitted by the existing approved-only builder. A failed check returns HTTP 409 with issue codes and the affected item IDs. Readiness uses the same captured pack metadata for required-slot coverage and for file checks. Successful runtime manifests use the `vn_asset_manifest.v1` schema. Portability backups can include draft and incomplete packs.

The checks describe file state at the time of the request. Content URLs still enforce ownership and access when opened. [ADR-020](../ADR/020-db-management-per-user-paths-and-content-backend.md) governs owner-local paths. The existing storage and manifest boundaries cover this change.
