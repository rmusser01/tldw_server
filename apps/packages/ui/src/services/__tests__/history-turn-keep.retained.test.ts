import i18n from "i18next"
import { beforeAll, describe, expect, it } from "vitest"

import type { HistoryTurnRecovery } from "@/db/dexie/types"
import { decodeChatErrorPayload } from "@/utils/chat-error-message"
import {
  isRetainedPartialReply,
  retainedReplyMessage
} from "../history-turn-keep"

const admission = {
  version: 1 as const,
  owner_key: "owner",
  conversation_id: "chat",
  input_message_id: "question",
  input_message_revision: "1",
  selection_digest: "digest"
}

const turn = (extra: Partial<HistoryTurnRecovery>): HistoryTurnRecovery =>
  ({
    operation_id: "op",
    origin_view: {} as HistoryTurnRecovery["origin_view"],
    selection_digest: "digest",
    request_context_digest: "context",
    owner_key: "owner",
    conversation_id: "chat",
    created_at: 1,
    input_text: "Question",
    input_images: [],
    result_text: "",
    state: "accepted_unsent",
    input_id: "question",
    assistant_id: "reply",
    admission,
    ...extra
  }) as HistoryTurnRecovery

const failure = {
  status: 502,
  code: "provider_unavailable",
  message: "The chat service provider is currently unavailable.",
  provider: "custom-openai-api",
  model: "local-uat-chat"
}

describe("retained replies (CS-04, CM-02, CC-02)", () => {
  beforeAll(async () => {
    await i18n.init({ lng: "en", resources: {} })
  })

  it("keeps a partial reply of a server chat that only this device holds", () => {
    expect(isRetainedPartialReply(turn({ outcome: "interrupted", result_text: "Part" }), "native")).toBe(true)
    // A local chat stores its partial reply with the marker instead.
    expect(isRetainedPartialReply(turn({ outcome: "interrupted", result_text: "Part" }), "local")).toBe(false)
    expect(isRetainedPartialReply(turn({ outcome: "complete", result_text: "Done" }), "native")).toBe(false)
  })

  it("keeps a reply stopped before it started, and a failure the server reported", () => {
    for (const owner of ["native", "local"] as const) {
      expect(isRetainedPartialReply(turn({ outcome: "stopped" }), owner)).toBe(true)
      expect(isRetainedPartialReply(turn({ outcome: "interrupted", failure }), owner)).toBe(true)
      // An early end with nothing to show and no known cause stays in review.
      expect(isRetainedPartialReply(turn({ outcome: "interrupted" }), owner)).toBe(false)
    }
  })

  it("keeps a server-settled turn only once the server confirmed its question and the outcome is known", () => {
    const server = { persistence: "server" as const }
    expect(isRetainedPartialReply(turn({ ...server, outcome: "stopped", result_text: "Part" }), "native")).toBe(true)
    expect(isRetainedPartialReply(turn({ ...server, outcome: "interrupted", failure }), "native")).toBe(true)
    // A dropped stream may still have been saved by the server: review.
    expect(isRetainedPartialReply(turn({ ...server, outcome: "interrupted", result_text: "Part" }), "native")).toBe(false)
    expect(isRetainedPartialReply(turn({ ...server, outcome: "stopped", admission: undefined }), "native")).toBe(false)
    expect(isRetainedPartialReply(turn({ ...server, outcome: "stopped", settled_message_id: "saved" }), "native")).toBe(false)
  })

  it("shows a kept failure as an error reply naming the provider and model", () => {
    const message = retainedReplyMessage(
      turn({ outcome: "interrupted", failure, interruption_reason: failure.message })
    )
    expect(message).toMatchObject({ id: "reply", isBot: true, parentMessageId: "question" })
    expect(message.generationInfo).toMatchObject({ interrupted: true })
    expect(decodeChatErrorPayload(message.message)).toMatchObject({
      summary: "Couldn't reach Custom OpenAI API (local-uat-chat)",
      recoveryAction: "open-model-selector"
    })
  })

  it("shows a stopped reply with what it wrote so far", () => {
    const message = retainedReplyMessage(turn({ outcome: "stopped", result_text: "Part" }))
    expect(message.message).toBe("Part")
    expect(message.generationInfo).toMatchObject({ interrupted: true, stopped: true })
  })
})
