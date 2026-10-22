import { beforeEach, describe, expect, it } from "vitest"

import {
  isSavedReplyActionRefusal,
  nativeSavedReplyTarget,
  rememberSavedReplyRefusal,
  resetSavedReplyRefusals,
  savedReplyActionAvailability,
  savedReplyRegenerateUnavailableReason,
  savedReplyStatusOffersActions
} from "../saved-reply-actions"
import type { HistorySelectionController } from "../useHistorySelection"
import type { Message } from "@/store/option"

const t = (_key: string, fallback: string) => fallback

const view = {
  view_session_id: "view",
  owner_key: "owner",
  conversation_id: "chat-1",
  interpretation: { kind: "parent_graph_v1" as const },
  cursor: { kind: "after_message" as const, message_id: "a1" },
  selection_revision: 1
}

const node = (id: string, role: string, parent_id: string | null) => ({
  id,
  revision: "1",
  role,
  parent_id,
  settled: true
})

const captured = (overrides: { reply?: Record<string, unknown>; interpretation?: unknown } = {}) => ({
  status: "captured" as const,
  snapshot: { nodes: [] },
  rows: [node("u1", "user", null), node("a1", "assistant", "u1")],
  selected_content: [
    { id: "u1", revision: "1", message: "Question", images: [] },
    { id: "a1", revision: "1", message: "Answer", images: [], ...overrides.reply }
  ],
  view: { ...view, interpretation: overrides.interpretation ?? view.interpretation },
  purpose: "send" as const,
  storage_context_digest: "storage"
})

const controller = (state: Record<string, unknown>) =>
  ({ getCurrent: () => ({ owner: null, view: null, capture: null, ...state }) }) as unknown as HistorySelectionController

const ready = (overrides: Parameters<typeof captured>[0] = {}, owner = { kind: "native" }) =>
  controller({ status: "ready", owner, view, capture: captured(overrides) })

const visible: Message[] = [
  { id: "u1", isBot: false, name: "You", message: "Question", sources: [] },
  { id: "a1", isBot: true, name: "Assistant", message: "Answer", sources: [] }
]

describe("saved reply action availability (CM-01)", () => {
  it("keeps ordinary actions for a chat without selected saved history", () => {
    expect(savedReplyActionAvailability({ historySelection: null, messages: visible })).toEqual({ regenerate: null, continue: null })
    expect(savedReplyActionAvailability({ historySelection: controller({ status: "idle" }), messages: visible }))
      .toEqual({ regenerate: null, continue: null })
    expect(savedReplyActionAvailability({
      historySelection: controller({ status: "ready" }), messages: visible, temporaryChat: true
    })).toEqual({ regenerate: null, continue: null })
  })

  it("offers Regenerate and Continue on a ready saved reply that matches the thread", () => {
    expect(savedReplyActionAvailability({ historySelection: ready(), messages: visible })).toEqual({ regenerate: null, continue: null })
    expect(nativeSavedReplyTarget(ready(), visible)?.reply.id).toBe("a1")
  })

  it("matches saved rows through their server message IDs", () => {
    const local = visible.map((row) => ({ ...row, id: `local-${row.id}`, serverMessageId: row.id }))
    expect(nativeSavedReplyTarget(ready(), local)?.input.id).toBe("u1")
  })

  it("reports a loading saved chat instead of refusing on click", () => {
    const availability = savedReplyActionAvailability({ historySelection: controller({ status: "loading", owner: { kind: "native" } }), messages: visible })
    expect(availability).toEqual({ regenerate: "loading", continue: "loading" })
    expect(savedReplyRegenerateUnavailableReason(availability.regenerate, t)).toMatch(/still loading/)
  })

  // The send path regenerates a device history (local owner) through its own
  // boundary; Continue still needs a native saved reply.
  it("offers Regenerate but not Continue for a local owner", () => {
    expect(savedReplyActionAvailability({ historySelection: ready({}, { kind: "local" }), messages: visible }))
      .toEqual({ regenerate: null, continue: "unsupported" })
  })

  it("does not offer Regenerate for a local reply that does not answer the question before it", () => {
    const capture = captured()
    const historySelection = controller({
      status: "ready",
      owner: { kind: "local" },
      view,
      capture: { ...capture, rows: [capture.rows[0], { ...capture.rows[1], parent_id: "elsewhere" }] }
    })
    expect(savedReplyActionAvailability({ historySelection, messages: visible }).regenerate).toBe("unsupported")
  })

  it.each([
    ["a legacy linear history", ready({ interpretation: { kind: "legacy_linear_v1", projection_id: "p" } })],
    ["a history that needs review", controller({ status: "legacy_review_required", owner: { kind: "native" } })],
    ["an unsupported selection", controller({ status: "unsupported_history_capability", owner: { kind: "native" } })]
  ])("disables both actions for %s", (_label, historySelection) => {
    const availability = savedReplyActionAvailability({ historySelection, messages: visible })
    expect(availability).toEqual({ regenerate: "unsupported", continue: "unsupported" })
    expect(savedReplyRegenerateUnavailableReason(availability.regenerate, t)).toBe(
      "This saved reply can't be regenerated here. Use New Branch to try again from this point."
    )
  })

  it("disables both actions when the visible reply no longer matches the saved one", () => {
    const edited = [visible[0], { ...visible[1], message: "Edited locally" }]
    expect(savedReplyActionAvailability({ historySelection: ready(), messages: edited }))
      .toEqual({ regenerate: "unsupported", continue: "unsupported" })
  })

  it.each([
    ["tool calls", { tool_calls: [{ id: "call-1" }] }],
    ["an image", { images: ["data:image/png;base64,AA=="] }],
    ["no text", { message: "  " }]
  ])("keeps Regenerate but offers no Continue for a reply with %s", (_label, reply) => {
    const messages = [visible[0], { ...visible[1], message: (reply as { message?: string }).message ?? "Answer" }]
    expect(savedReplyActionAvailability({ historySelection: ready({ reply }), messages }))
      .toEqual({ regenerate: null, continue: "unsupported" })
  })

  it("keeps Retry for a reply that stopped early but offers no Continue on it", () => {
    const interrupted = [visible[0], { id: "local-draft", isBot: true, name: "Assistant", message: "Part", sources: [], generationInfo: { interrupted: true } }]
    expect(savedReplyActionAvailability({ historySelection: ready(), messages: interrupted as Message[] }))
      .toEqual({ regenerate: null, continue: "unsupported" })
  })
})

describe("recorded reply status actions (CM-03)", () => {
  it("lets the last saved reply of a server chat offer its Continue and Retry", () => {
    expect(savedReplyStatusOffersActions(ready(), visible)).toBe(true)
  })

  it.each([
    ["no selected saved history", null],
    ["a new chat", controller({ status: "idle" })],
    ["a loading saved chat", controller({ status: "loading", owner: { kind: "native" } })],
    ["a local owner", ready({}, { kind: "local" })],
    ["a legacy linear history", ready({ interpretation: { kind: "legacy_linear_v1", projection_id: "p" } })]
  ])("keeps the recorded status passive for %s", (_label, historySelection) => {
    expect(savedReplyStatusOffersActions(historySelection, visible)).toBe(false)
  })

  it("keeps the recorded status passive when the visible reply is not the saved one", () => {
    expect(savedReplyStatusOffersActions(ready(), [visible[0], { ...visible[1], message: "Edited locally" }])).toBe(false)
  })
})

describe("remembered saved-reply refusals (CM-01)", () => {
  beforeEach(() => resetSavedReplyRefusals())
  const intent = (action?: "continue") => ({
    owner_key: "owner", conversation_id: "chat-1", message_id: "a1", message_revision: "1",
    input_message_id: "u1", input_message_revision: "1", isCurrent: () => true, isOriginCurrent: () => true,
    ...(action ? { action } : {})
  })
  const refusal = (code: string, status = "unsupported_history_capability") =>
    ({ status: 409, details: { detail: { status, code } } })

  it.each([
    [refusal("unsupported_history_regeneration"), true],
    [refusal("unsupported_history_continuation"), true],
    [refusal("unsupported_history_regeneration", "stale_selection"), false],
    [refusal("unsupported_history_context_memory"), false],
    [{ status: 500, details: { detail: { status: "unsupported_history_capability", code: "unsupported_history_regeneration" } } }, false],
    [new Error("unsupported_history_regeneration"), false]
  ])("recognizes only the server's definitive refusal %#", (error, expected) => {
    expect(isSavedReplyActionRefusal(error)).toBe(expected)
  })

  it("disables both actions on a reply whose Regenerate was refused", () => {
    rememberSavedReplyRefusal(intent())
    expect(savedReplyActionAvailability({ historySelection: ready(), messages: visible }))
      .toEqual({ regenerate: "unsupported", continue: "unsupported" })
  })

  it("withdraws only Continue after a refused Continue", () => {
    rememberSavedReplyRefusal(intent("continue"))
    expect(savedReplyActionAvailability({ historySelection: ready(), messages: visible }))
      .toEqual({ regenerate: null, continue: "unsupported" })
  })

  it("does not carry a refusal to another revision of the reply", () => {
    rememberSavedReplyRefusal({ ...intent(), message_revision: "2" })
    expect(savedReplyActionAvailability({ historySelection: ready(), messages: visible }))
      .toEqual({ regenerate: null, continue: null })
  })
})
