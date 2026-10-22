import { describe, expect, it } from "vitest"

import { NativeHistoryBranchChoice } from "@/hooks/chat/native-character-preflight"
import {
  adoptFailureAdmission,
  definitiveHistoryTurnFailure
} from "../history-turn-failure"

const providerUnavailable = (extra: Record<string, unknown> = {}) =>
  Object.assign(new Error("The chat service provider is currently unavailable."), {
    status: 502,
    details: {
      detail: {
        error_code: "provider_unavailable",
        message: "The chat service provider is currently unavailable.",
        ...extra
      }
    }
  })

const settled = {
  aborted: false,
  scopeInvalidated: false,
  replyCompleted: false,
  fullText: "",
  selectedModel: "custom-openai-api:local-uat-chat"
}

describe("history turn failures (CC-02)", () => {
  it("settles a turn the server answered with an error before any reply", () => {
    expect(
      definitiveHistoryTurnFailure({
        ...settled,
        error: new Error("Stream completion failed", { cause: providerUnavailable() }),
        historyTurn: { persistence: "server" }
      })
    ).toEqual({
      status: 502,
      code: "provider_unavailable",
      message: "The chat service provider is currently unavailable.",
      provider: "custom-openai-api",
      model: "local-uat-chat"
    })
  })

  it("leaves stops, partial replies, unsaved client questions and unknown errors to their own handling", () => {
    const error = providerUnavailable()
    const client = { persistence: "client" as const, admission: { input_message_id: "q" } as never }
    expect(definitiveHistoryTurnFailure({ ...settled, error, historyTurn: client, aborted: true })).toBeUndefined()
    expect(definitiveHistoryTurnFailure({ ...settled, error, historyTurn: client, fullText: "Part" })).toBeUndefined()
    expect(definitiveHistoryTurnFailure({ ...settled, error, historyTurn: client, scopeInvalidated: true })).toBeUndefined()
    // A client-settled question that was never saved is a failed save, not a failed reply.
    expect(definitiveHistoryTurnFailure({ ...settled, error, historyTurn: { persistence: "client" } })).toBeUndefined()
    expect(
      definitiveHistoryTurnFailure({ ...settled, error: new TypeError("network error"), historyTurn: client })
    ).toBeUndefined()
    expect(
      definitiveHistoryTurnFailure({
        ...settled,
        error: new NativeHistoryBranchChoice(() => true),
        historyTurn: { persistence: "server" }
      })
    ).toBeUndefined()
  })

  it("adopts a saved question only from a server-settled native turn's error", () => {
    const owner = { kind: "native", conversation_id: "chat", owner_key: "owner" } as never
    const turn = {
      owner,
      persistence: "server" as const,
      selection: { selection_digest: "digest" } as never,
      capture: { snapshot: { nodes: [] } } as never
    }
    // No admission in the error, or one that does not belong to this turn.
    expect(adoptFailureAdmission(turn, providerUnavailable())).toBeUndefined()
    expect(
      adoptFailureAdmission(turn, providerUnavailable({ tldw_history_admission_v1: { input_message_id: "forged" } }))
    ).toBeUndefined()
    // Client-settled turns already know their question.
    expect(
      adoptFailureAdmission(
        { ...turn, persistence: "client" },
        providerUnavailable({ tldw_history_admission_v1: { input_message_id: "q" } })
      )
    ).toBeUndefined()
  })
})
