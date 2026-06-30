// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundUserMessageBubble } from "../PlaygroundUserMessage"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue, vi.fn()]
}))

vi.mock("antd", () => ({
  Image: ({ src, alt }: { src?: string; alt?: string }) => (
    <span role="img" aria-label={alt ?? ""} data-src={src ?? ""} />
  ),
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    cancel: vi.fn(),
    isSpeaking: false,
    speak: vi.fn()
  })
}))

vi.mock("../HumanMessge", () => ({
  HumanMessage: ({ message }: { message: string }) => <div>{message}</div>
}))

vi.mock("../EditMessageForm", () => ({
  EditMessageForm: () => <div data-testid="edit-message-form" />
}))

vi.mock("../DocumentChip", () => ({
  DocumentChip: () => <div data-testid="document-chip" />
}))

vi.mock("../DocumentFile", () => ({
  DocumentFile: () => <div data-testid="document-file" />
}))

vi.mock("@/utils/chat-style", () => ({
  buildChatTextClass: () => ""
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: "standard" })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: { setReplyTarget: (...args: unknown[]) => void }) => unknown
  ) =>
    selector({
      setReplyTarget: vi.fn()
    })
}))

const renderUserMessage = (
  props: Partial<React.ComponentProps<typeof PlaygroundUserMessageBubble>> = {}
) =>
  render(
    <PlaygroundUserMessageBubble
      message="Review this research plan."
      isBot={false}
      name="You"
      currentMessageIndex={0}
      totalMessages={1}
      onRegenerate={() => undefined}
      onEditFormSubmit={() => undefined}
      isProcessing
      isStreaming={false}
      {...props}
    />
  )

describe("Playground user message design-system badges", () => {
  it("renders system and message-type chips through the shared Badge", () => {
    renderUserMessage({
      role: "system",
      message_type: "summary",
      messageId: "message-1"
    })

    expect(screen.getByTestId("chat-message")).toHaveAttribute("data-role", "system")
    expect(screen.getByText("Review this research plan.")).toBeInTheDocument()
    expect(screen.getByTestId("playground-system-message-badge")).toHaveAttribute(
      "data-ds-component",
      "Badge"
    )
    expect(screen.getByTestId("playground-system-message-badge")).toHaveTextContent(
      "System prompt"
    )
    expect(screen.getByTestId("playground-message-type-badge")).toHaveAttribute(
      "data-ds-component",
      "Badge"
    )
    expect(screen.getByTestId("playground-message-type-badge")).toHaveTextContent(
      "Summary"
    )
  })

  it("renders a human-readable fallback for unknown message-type chips", () => {
    renderUserMessage({
      message_type: "custom_review",
      messageId: "message-2"
    })

    const badge = screen.getByTestId("playground-message-type-badge")

    expect(badge).toHaveAttribute("data-ds-component", "Badge")
    expect(badge).toHaveTextContent("Custom review")
    expect(badge).not.toHaveTextContent("copilot.custom_review")
  })
})
