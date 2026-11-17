import React from "react"
import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ACPChatPanel, parseUpdateMessage } from "../ACPChatPanel"
import { getSessionMessageCount, getSessionTokenUsage } from "../sessionMetrics"
import { useACPSessionsStore } from "@/store/acp-sessions"
import type { ACPUpdate } from "@/services/acp/types"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback || key })
}))

const native = (type: string, content: unknown): ACPUpdate => ({
  timestamp: new Date("2026-10-09T00:00:00Z"),
  type,
  data: { sessionId: "owned", update: { sessionUpdate: type, content } }
})

describe("native ACP text consumers", () => {
  const originalScroll = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollIntoView")
  beforeEach(() => {
    useACPSessionsStore.getState().reset()
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() })
  })
  afterEach(() => {
    if (originalScroll) Object.defineProperty(HTMLElement.prototype, "scrollIntoView", originalScroll)
    else Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView")
  })

  it.each(["agent_message_chunk", "user_message_chunk"])("renders %s through the real role renderer", (type) => {
    useACPSessionsStore.getState().createSession({ cwd: "/tmp/project" })
    render(<ACPChatPanel state="connected" isConnected updates={[native(type, { type: "text", text: "Controlled native summary" })]}
      connect={vi.fn()} sendPrompt={vi.fn()} cancel={vi.fn()} />)
    expect(screen.getByText("Controlled native summary")).toBeInTheDocument()
    expect(screen.getByText(type === "agent_message_chunk" ? "Agent" : "You")).toBeInTheDocument()
    expect(screen.queryByText(/sessionUpdate/)).not.toBeInTheDocument()
  })

  it("keeps exact authored native text at the parser boundary", () => {
    expect(parseUpdateMessage(native("agent_message_chunk", { type: "text", text: "  exact\ntext  " }), "Cancelled"))
      .toEqual({ role: "assistant", content: "  exact\ntext  " })
  })

  it.each([{ type: "image", data: "opaque" }, { type: "text", text: { private: "object" } }, null])(
    "retains nontext or malformed native data as a diagnostic (%j)", (content) => {
      const update = native("agent_message_chunk", content)
      expect(parseUpdateMessage(update, "Cancelled")).toEqual({ role: "system", content: JSON.stringify(update.data, null, 2) })
    })

  it("counts native text messages without fabricating usage and preserves stored zero", () => {
    const state = useACPSessionsStore.getState()
    const id = state.createSession({ cwd: "/tmp/project" })
    const agent = native("agent_message_chunk", { type: "text", text: "Assistant" })
    const user = native("user_message_chunk", { type: "text", text: "User" })
    state.addUpdate(id, agent)
    state.addUpdate(id, user)
    state.addUpdate(id, { type: "usage_update", data: { update: { sessionUpdate: "usage_update", used: 0 } } })
    const session = useACPSessionsStore.getState().getSession(id)!
    expect(session.messageCount).toBe(2)
    expect(getSessionMessageCount(session)).toBe(2)
    expect(getSessionTokenUsage(session)).toBeNull()
    expect(getSessionTokenUsage({ ...session, usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } })).toBe(0)
  })

  it("preserves final legacy text interpretation", () => {
    expect(parseUpdateMessage({ timestamp: new Date(), type: "assistant_text", data: { text: "Legacy" } }, "Cancelled"))
      .toEqual({ role: "assistant", content: "Legacy" })
  })
})
