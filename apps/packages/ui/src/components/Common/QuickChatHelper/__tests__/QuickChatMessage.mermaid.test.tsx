// @vitest-environment jsdom
import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { QuickChatMessage } from "../QuickChatMessage"

const markdownCalls = vi.hoisted(() => [] as Array<Record<string, unknown>>)
const storageState = vi.hoisted(() => ({
  values: new Map<string, unknown>()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageState.values.has(key) ? storageState.values.get(key) : defaultValue,
    vi.fn()
  ]
}))

vi.mock("@/components/Common/Markdown", () => ({
  default: (props: Record<string, unknown>) => {
    markdownCalls.push(props)
    return <div data-testid="mock-markdown">{String(props.message ?? "")}</div>
  }
}))

const assistantMessage = {
  id: "assistant-1",
  role: "assistant" as const,
  content: "```mermaid\ngraph TD\n  A-->B\n```",
  timestamp: Date.now()
}

describe("QuickChatMessage Mermaid rendering gates", () => {
  beforeEach(() => {
    markdownCalls.length = 0
    storageState.values.clear()
  })

  it("enables Mermaid for a completed assistant message by default", async () => {
    render(<QuickChatMessage message={assistantMessage} />)

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]).toEqual(
      expect.objectContaining({
        enableMermaidDiagrams: true
      })
    )
  })

  it("does not enable Mermaid while an assistant message is streaming", async () => {
    render(
      <QuickChatMessage
        message={assistantMessage}
        isStreaming
        isLast
      />
    )

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]?.enableMermaidDiagrams).not.toBe(true)
  })

  it("does not render Markdown for user messages", async () => {
    render(
      <QuickChatMessage
        message={{
          id: "user-1",
          role: "user",
          content: "```mermaid\ngraph TD\n  A-->B\n```",
          timestamp: Date.now()
        }}
      />
    )

    expect(screen.getByText(/graph TD/)).toBeInTheDocument()
    await waitFor(() => {
      expect(markdownCalls).toHaveLength(0)
    })
  })

  it("does not enable Mermaid when the chat setting is disabled", async () => {
    storageState.values.set("renderMermaidDiagrams", false)

    render(<QuickChatMessage message={assistantMessage} />)

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]?.enableMermaidDiagrams).not.toBe(true)
  })
})
