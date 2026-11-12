import { describe, expect, it } from "vitest"
import { isDefinitiveNativeBranchRejection } from "../chat-history-selection"

const selection = {
  conversation_id: "owned",
  messages: [{ id: "selected-answer" }]
}
const rejection = () => ({
  status: 409,
  details: {
    detail: {
      status: "stale_selection",
      code: "history_branch_changed",
      conversation_id: "owned",
      parent_message_id: "selected-answer" as string | null,
      leaf_ids: ["new-answer"],
      history_version: 4
    }
  }
})

describe("native branch refusal authority", () => {
  it.each([false, true])(
    "qualifies the exact latest-leaf rejection with an empty selection=%s",
    (empty) => {
      const error = rejection()
      if (empty) error.details.detail.parent_message_id = null
      expect(
        isDefinitiveNativeBranchRejection(
          error,
          empty ? { ...selection, messages: [] } : selection
        )
      ).toBe(true)
    }
  )
  it.each([
    { name: "message only", error: new Error("history_branch_changed") },
    { name: "string HTTP status", error: { ...rejection(), status: "409" } },
    { name: "unrelated HTTP status", error: { ...rejection(), status: 503 } },
    { name: "unwrapped provider object", error: rejection().details.detail },
    ...[
      { status: "unsupported_history_capability" },
      { code: "future_refusal" },
      { conversation_id: "foreign" },
      { parent_message_id: "other-answer" },
      { leaf_ids: [] },
      { leaf_ids: ["new-answer", "new-answer"] },
      { leaf_ids: [""] },
      { history_version: "4" },
      { history_version: -1 },
      { history_version: Number.MAX_SAFE_INTEGER + 1 }
    ].map((change) => ({
      name: JSON.stringify(change),
      error: {
        ...rejection(),
        details: { detail: { ...rejection().details.detail, ...change } }
      }
    })),
    {
      name: "array detail",
      error: { status: 409, details: { detail: [rejection().details.detail] } }
    }
  ])("keeps $name uncertain", ({ error }) => {
    expect(isDefinitiveNativeBranchRejection(error, selection)).toBe(false)
  })
})
