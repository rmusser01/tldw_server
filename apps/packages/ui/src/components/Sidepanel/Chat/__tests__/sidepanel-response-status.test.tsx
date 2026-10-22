import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import {
  SidepanelResponseStatus,
  resolveSidepanelPrimaryAction
} from "../sidepanel-response-status"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      String(fallback ?? _key).replace(/\{\{(\w+)\}\}/g, (_match, name) =>
        String(options?.[name] ?? "")
      )
  })
}))

const t = (_key: string, fallback: string) => fallback

describe("side panel primary action while a reply is pending (XS-10)", () => {
  it("says Send next and offers Stop before the first token arrives", () => {
    const action = resolveSidepanelPrimaryAction({
      isSending: true,
      isConnectionReady: true,
      sendTitle: "Send",
      t
    })
    expect(action.label).toBe("Send next")
    expect(action.showStop).toBe(true)
    expect(action.queuesAfterReply).toBe(true)
    expect(action.ariaLabel).toBe("Send next, after the current reply")
    expect(action.title).toBe("Send this message after the current reply finishes.")
  })

  it("sends normally and hides Stop when nothing is pending", () => {
    const action = resolveSidepanelPrimaryAction({
      isSending: false,
      isConnectionReady: true,
      sendTitle: "Press Enter to send",
      t
    })
    expect(action).toMatchObject({
      label: "Send",
      showStop: false,
      queuesAfterReply: false,
      ariaLabel: "Send message",
      title: "Press Enter to send"
    })
  })

  it("never promises to queue while offline", () => {
    const action = resolveSidepanelPrimaryAction({
      isSending: false,
      isConnectionReady: false,
      t
    })
    expect(action.label).toBe("Send")
    expect(action.queuesAfterReply).toBe(false)
    expect(action.title).toBe(
      "Connect to your tldw server before sending. Your draft has not been queued."
    )
  })

  it("explains a blocked Send before anything else", () => {
    const action = resolveSidepanelPrimaryAction({
      isSending: true,
      isConnectionReady: true,
      disabledReason: "Saving the character’s opening.",
      t
    })
    expect(action.title).toBe("Saving the character’s opening.")
  })
})

describe("side panel waiting status (XS-10)", () => {
  const pending = [
    { isBot: false, message: "Hello" },
    { isBot: true, message: "▋" }
  ]

  it("names the model while the reply has no text yet", () => {
    render(
      <SidepanelResponseStatus
        isSending
        messages={pending}
        selectedModel="tldw:mock-gpt-4o"
        models={[{ model: "tldw:mock-gpt-4o", nickname: "Mock GPT-4o" }]}
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent("Waiting for Mock GPT-4o…")
  })

  it("falls back to the model id when the catalog has no name for it", () => {
    render(<SidepanelResponseStatus isSending messages={pending} selectedModel="llama3:8b" />)
    expect(screen.getByRole("status")).toHaveTextContent("Waiting for llama3:8b…")
  })

  it("disappears once the reply starts", () => {
    render(
      <SidepanelResponseStatus
        isSending
        messages={[...pending.slice(0, 1), { isBot: true, message: "Hi there▋" }]}
        selectedModel="tldw:mock-gpt-4o"
      />
    )
    expect(screen.queryByTestId("chat-waiting-status")).not.toBeInTheDocument()
  })

  it("stays hidden when nothing is pending", () => {
    render(<SidepanelResponseStatus isSending={false} messages={pending} selectedModel="m" />)
    expect(screen.queryByTestId("chat-waiting-status")).not.toBeInTheDocument()
  })
})
