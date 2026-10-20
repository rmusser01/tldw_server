// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundSendControl } from "../PlaygroundSendControl"
import { isComposerDraftEmpty } from "../composer-send-readiness"

const t = (_key: string, fallback?: string) => fallback ?? _key

const baseProps = {
  isProMode: false,
  isMobileViewport: true,
  isSending: false,
  isConnectionReady: true,
  sendWhenEnter: true,
  onSendWhenEnterChange: vi.fn(),
  sendLabel: "Send",
  compareNeedsMoreModels: false,
  onStopStreaming: vi.fn(),
  onStopListening: vi.fn(),
  onSubmitForm: vi.fn(),
  sendMenuOpen: false,
  onSendMenuChange: vi.fn(),
  t
}

describe("PlaygroundSendControl accessibility", () => {
  it("keeps the ready send action distinct from its adjacent delivery-options trigger", () => {
    render(<PlaygroundSendControl {...baseProps} />)

    expect(
      screen.getByRole("button", { name: "Send message" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open message delivery options" })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Open send options" })
    ).not.toBeInTheDocument()
    expect(screen.queryAllByRole("button", { name: /send/i })).toHaveLength(1)
  })

  it("disables Send and explains why while a fresh chat's history selection is not idle (CS-01, #3106)", () => {
    const { rerender } = render(
      <PlaygroundSendControl {...baseProps} isHistorySelectionPending />
    )

    const send = screen.getByRole("button", { name: "Send message" })
    expect(send).toBeDisabled()
    expect(send).toHaveAttribute(
      "title",
      "Conversation history isn't ready yet. Wait for it to load, or start a new chat."
    )

    rerender(<PlaygroundSendControl {...baseProps} isHistorySelectionPending={false} />)
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
  })
})

describe("unavailable delivery", () => {
  it("shows a disabled Send instead of offering an offline queue", () => {
    render(<PlaygroundSendControl {...baseProps} isConnectionReady={false} />)
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
    expect(screen.queryByRole("button", { name: /send next/i })).not.toBeInTheDocument()
  })
})

describe("while a reply is being written (CM-02)", () => {
  it("labels Stop in words and offers to send the next message after the reply", () => {
    const onStopStreaming = vi.fn()
    render(<PlaygroundSendControl {...baseProps} isSending onStopStreaming={onStopStreaming} />)

    // The real locale names it "Stop Streaming"; this test's t() echoes the key.
    const stop = screen.getByRole("button", { name: /stop/i })
    expect(stop).toHaveTextContent(/^Stop$/)
    stop.click()
    expect(onStopStreaming).toHaveBeenCalledTimes(1)

    const next = screen.getByRole("button", { name: "Send next, after the current reply" })
    expect(next).toHaveTextContent("Send next")
    expect(next).toHaveAttribute("title", "Send this message after the current reply finishes.")
    expect(screen.queryByText(/^queue$/i)).not.toBeInTheDocument()
  })
})

describe("an empty composer (CC-02)", () => {
  it("disables Send and says why until there is something to send", () => {
    const { rerender } = render(<PlaygroundSendControl {...baseProps} isDraftEmpty />)
    const send = screen.getByRole("button", { name: "Send message" })
    expect(send).toBeDisabled()
    expect(send).toHaveAttribute("title", "Type a message to send.")

    rerender(<PlaygroundSendControl {...baseProps} isDraftEmpty={false} />)
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
  })

  it("counts text, an image, attachments, page context and pinned results as something to send", () => {
    const empty = {
      message: "  ",
      image: "",
      documentCount: 0,
      fileCount: 0,
      hasImportedContext: false,
      pinnedResultCount: 0,
      fileRetrievalEnabled: false
    }
    expect(isComposerDraftEmpty(empty)).toBe(true)
    expect(isComposerDraftEmpty({ ...empty, message: "Hi" })).toBe(false)
    expect(isComposerDraftEmpty({ ...empty, image: "data:image/png;base64,AA" })).toBe(false)
    expect(isComposerDraftEmpty({ ...empty, documentCount: 1 })).toBe(false)
    expect(isComposerDraftEmpty({ ...empty, fileCount: 1 })).toBe(false)
    expect(isComposerDraftEmpty({ ...empty, hasImportedContext: true })).toBe(false)
    expect(isComposerDraftEmpty({ ...empty, pinnedResultCount: 2 })).toBe(false)
    // With file retrieval on, pinned results are not added to the message.
    expect(isComposerDraftEmpty({ ...empty, pinnedResultCount: 2, fileRetrievalEnabled: true })).toBe(true)
  })
})
