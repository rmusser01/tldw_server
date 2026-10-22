// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

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
  Image: ({ src, alt, onError }: { src?: string; alt?: string; onError?: () => void }) => (
    // eslint-disable-next-line @next/next/no-img-element -- Native img delivers the browser error event instead of a non-image placeholder.
    <img alt={alt ?? ""} src={src ?? ""} onError={onError} />
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

const uiMode = vi.hoisted(() => ({ value: "standard" }))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: uiMode.value })
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

  it("renders document processing metadata below the user message", () => {
    renderUserMessage({
      messageId: "message-3",
      metadataExtra: {
        documentProcessing: {
          status: "processing",
          files: [
            {
              id: "file-1",
              filename: "scan.pdf",
              mode: "ocr_pages",
              status: "processing"
            }
          ]
        }
      }
    })

    expect(screen.getByText("Processing documents")).toBeInTheDocument()
    expect(screen.getByText("scan.pdf")).toBeInTheDocument()
  })
})

// CM-10 (#3101): user turns must read apart from replies, not as a second
// card of the same colour and width.
describe("Playground user message turn styling", () => {
  afterEach(() => {
    uiMode.value = "standard"
  })

  const bubble = () =>
    screen.getByText("Review this research plan.").closest(".message-bubble") as HTMLElement

  it("tints the user bubble with the primary colour", () => {
    renderUserMessage({ messageId: "message-4" })

    expect(bubble().style.backgroundImage).toContain("--color-primary")
  })

  it("caps the user bubble at three quarters of the row in casual mode", () => {
    uiMode.value = "casual"
    renderUserMessage({ messageId: "message-5" })

    expect(bubble().className).toMatch(/(^|\s)sm:max-w-\[75%\](\s|$)/)
  })

  it("keeps the wider bubble in Pro mode", () => {
    uiMode.value = "pro"
    renderUserMessage({ messageId: "message-6" })

    expect(bubble().className).not.toContain("max-w-[75%]")
  })
})

it("retains the question and names an unavailable attachment after image failure", () => {
  renderUserMessage({ images: ["http://chat.test/missing.png"], messageId: "missing-attachment" })
  fireEvent.error(screen.getByRole("img", { name: "Uploaded Image" }))
  expect(screen.getByRole("status")).toHaveTextContent("Image unavailable. Attach this image again before continuing.")
  expect(screen.getByText("Review this research plan.")).toBeVisible()
})
