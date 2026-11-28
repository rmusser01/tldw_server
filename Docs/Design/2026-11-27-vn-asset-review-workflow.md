# VN asset review workflow

[GitHub #2023](https://github.com/rmusser01/tldw_server/issues/2023) · TASK-15686

Use the existing review board and authenticated item preview endpoint. Load an image when the reviewer opens its card preview. Allow one open preview at a time. Closing it, navigating to a different pack, or changing the asset's file releases its request and temporary URL. Request history records the preview's media type and size.

Filter cards by review status and submit only checked items in view. Keep selections for visible items on refresh and after a failed mutation. Clear the submitted selection after a successful review and return focus to the filter when the action still has focus. Preferred variants can be set on approved items.

Verify the current owner before sending a preview or review request. Keep one review write in flight per server, owner, and pack. A same-account re-verification keeps the loaded review snapshot and its selection. Apply a confirmed mutation before refreshing details so a failed refresh can be reported separately from a failed review.

Use the WebUI's existing API client and owner-scoped VN routes. This slice needs no new ADR, API contract, or storage.
