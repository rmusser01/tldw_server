// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
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

vi.mock("antd", async (importOriginal) => ({
  ...(await importOriginal<typeof import("antd")>()),
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

it("retains the question and names an unavailable attachment after image failure", () => {
  renderUserMessage({ images: ["http://chat.test/missing.png"], messageId: "missing-attachment" })
  fireEvent.error(screen.getByRole("img", { name: "Uploaded Image" }))
  expect(screen.getByRole("status")).toHaveTextContent("Image unavailable. Attach this image again before continuing.")
  expect(screen.getByText("Review this research plan.")).toBeVisible()
})

it("keeps an active user edit mounted when the message view changes", async () => {
  const props = {
    message: "Review this research plan.", isBot: false, name: "You",
    currentMessageIndex: 0, totalMessages: 1, onRegenerate: vi.fn(),
    onEditFormSubmit: vi.fn(), isProcessing: false, messageId: "retained-edit"
  }
  const { rerender } = render(<PlaygroundUserMessageBubble {...props} messageView="transcript" />)
  fireEvent.click(screen.getByRole("button", { name: "More actions" }))
  fireEvent.click(await screen.findByRole("button", { name: "edit" }))
  const form = screen.getByTestId("edit-message-form")
  rerender(<PlaygroundUserMessageBubble {...props} messageView="conversation" />)
  expect(screen.getByTestId("edit-message-form")).toBe(form)
  rerender(<PlaygroundUserMessageBubble {...props} messageView="research" />)
  expect(screen.getByTestId("edit-message-form")).toBe(form)
})
