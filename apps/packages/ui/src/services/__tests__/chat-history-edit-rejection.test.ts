import { describe, expect, it } from "vitest"
import { isDefinitiveNativeEditRejection } from "../chat-history-selection"
import { TldwApiError } from "../tldw/api-error"
const selection = {
  conversation_id: "owned",
  cursor: { kind: "before_message" as const, message_id: "target" },
  selection_digest: "original-selection"
}
const target = {
  version: 1 as const,
  message_id: "target",
  message_revision: "opaque-original"
}
const detail = () => ({
  status: "stale_selection",
  code: "history_edit_target_changed",
  conversation_id: "owned",
  message_id: "target",
  message_revision: "opaque-original",
  selection_digest: "original-selection"
})
const bg = () => ({ status: 409, details: { detail: detail() } })
describe("protected Edit refusal authority", () => {
  it.each(["background", "direct"])(
    "accepts the exact original %s transport refusal",
    (envelope) => {
      const error =
        envelope === "direct"
          ? new TldwApiError("changed", 409, detail())
          : bg()
      expect(isDefinitiveNativeEditRejection(error, selection, target)).toBe(
        true
      )
    }
  )
  it.each([
    new Error("history_edit_target_changed"),
    { ...bg(), status: "409" },
    { ...bg(), status: 503 },
    detail(),
    { status: 409, details: { detail: [detail()] } },
    { ...bg(), detail: { ...detail(), message_revision: "other" } },
    ...[
      "status",
      "code",
      "conversation_id",
      "message_id",
      "message_revision",
      "selection_digest"
    ].flatMap((key) => [
      { status: 409, details: { detail: { ...detail(), [key]: undefined } } },
      { status: 409, details: { detail: { ...detail(), [key]: "foreign" } } }
    ])
  ])(
    "keeps an incomplete, foreign or ambiguous response uncertain (%j)",
    (error) => {
      expect(isDefinitiveNativeEditRejection(error, selection, target)).toBe(
        false
      )
    }
  )
  it("requires the retained matching before-message cursor", () => {
    expect(
      isDefinitiveNativeEditRejection(
        bg(),
        {
          ...selection,
          cursor: { kind: "before_message", message_id: "foreign" }
        },
        target
      )
    ).toBe(false)
  })
})
