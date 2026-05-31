import { describe, expect, it } from "vitest"
import {
  buildQuickChatPopoutState,
  parseQuickChatPopoutState
} from "../popout-state"

describe("quick chat popout state", () => {
  it("stores normalized source route context", () => {
    const state = buildQuickChatPopoutState(
      {
        messages: [],
        modelOverride: null,
        assistantMode: "docs_rag"
      },
      "chrome-extension://abc/options.html#/research-workspace?tab=chat"
    )

    expect(state.sourceRoute).toBe("/research-workspace")
  })

  it("parses valid popout state with route context", () => {
    const parsed = parseQuickChatPopoutState({
      messages: [
        {
          id: "m1",
          role: "user",
          content: "How do I do this on this page?",
          timestamp: 123
        }
      ],
      modelOverride: "gpt-4o-mini",
      assistantMode: "docs_rag",
      sourceRoute: "/knowledge"
    })

    expect(parsed).not.toBeNull()
    expect(parsed?.assistantMode).toBe("docs_rag")
    expect(parsed?.sourceRoute).toBe("/knowledge")
    expect(parsed?.messages).toHaveLength(1)
  })

  it("rejects invalid popout state payloads", () => {
    const parsed = parseQuickChatPopoutState({
      messages: [{ id: 123, role: "user", content: "x", timestamp: 1 }],
      assistantMode: "docs_rag",
      sourceRoute: "/chat"
    })
    expect(parsed).toBeNull()
  })
})
