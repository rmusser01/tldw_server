import { describe, expect, it } from "vitest"

import { getMessageDeleteFeedback } from "../message-delete-feedback"

describe("saved message Delete recovery", () => {
  it.each([
    "__proto__",
    "constructor",
    "toString",
    "Unknown provider text",
    null
  ])("uses safe unconfirmed copy for %j", (issue) => {
    expect(getMessageDeleteFeedback(issue).key).toBe("deleteMessage.failed")
  })
  it.each([
    [
      { details: { detail: "message_has_descendants" }, status: 409 },
      "descendants"
    ],
    [{ detail: { error_code: "message_changed" } }, "changed"],
    [{ code: "response_in_progress" }, "busy"],
    [{ code: "request_config_scope_changed" }, "scope"],
    [{ detail: "unsupported_native_sync_delete" }, "sync"],
    [{ status: 404 }, "unavailable"],
    ["message_delete_refresh_failed", "refreshFailed"]
  ])("classifies a declared failure %j", (issue, category) => {
    expect(getMessageDeleteFeedback(issue).key).toBe(
      `deleteMessage.${category}`
    )
  })
})
