// @vitest-environment jsdom
/**
 * CC-07 (#3101): Send was the visually weakest control in the composer: a
 * small grey button with an 11px uppercase label. It is now the composer's
 * primary action with a 13px sentence-case label, and it dims while there is
 * nothing to send.
 */
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundSendControl } from "../PlaygroundSendControl"
import { hasSendableDraft } from "../composer-draft-content"

const t = (_key: string, fallback?: string) => fallback ?? _key

const casualDesktop = {
  isProMode: false,
  isMobileViewport: false,
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

describe("PlaygroundSendControl emphasis", () => {
  it("presents Send as the primary action at a regular size in casual mode", () => {
    render(<PlaygroundSendControl {...casualDesktop} />)

    const send = screen.getByRole("button", { name: "Send message" })
    expect(send).toHaveClass("ant-btn-primary")
    expect(send).not.toHaveClass("ant-btn-sm")
  })

  it("labels Send in 13px sentence case rather than small capitals", () => {
    render(<PlaygroundSendControl {...casualDesktop} />)

    const label = screen.getByText("Send")
    expect(label).not.toHaveClass("uppercase")
    expect(label).toHaveClass("text-[13px]")
  })

  it("dims Send while there is nothing to send and restores it once there is", () => {
    const { rerender } = render(
      <PlaygroundSendControl {...casualDesktop} hasDraftContent={false} />
    )
    const quiet = screen.getByRole("button", { name: "Send message" })
    expect(quiet).toHaveAttribute("data-draft-empty", "true")
    // Quieter, but still readable: a plain button, not a faded primary one
    // whose label fails contrast (axe color-contrast on /chat).
    expect(quiet).not.toHaveClass("ant-btn-primary")
    expect(quiet).not.toHaveClass("opacity-60")

    rerender(<PlaygroundSendControl {...casualDesktop} hasDraftContent />)
    const ready = screen.getByRole("button", { name: "Send message" })
    expect(ready).not.toHaveAttribute("data-draft-empty")
    expect(ready).toHaveClass("ant-btn-primary")
  })

  it("does not dim Send a second time when it is already disabled", () => {
    render(
      <PlaygroundSendControl
        {...casualDesktop}
        isConnectionReady={false}
        hasDraftContent={false}
      />
    )

    const send = screen.getByRole("button", { name: "Send message" })
    expect(send).toBeDisabled()
    expect(send).not.toHaveClass("opacity-60")
    expect(send).not.toHaveAttribute("data-draft-empty")
  })
})

// CC-03 (#3101): the disabled Send must not say "Connect…" while the first
// server check is still running and the rest of the composer says Connecting….
describe("PlaygroundSendControl while the server connection is pending", () => {
  it("says it is still connecting rather than asking to connect", () => {
    render(
      <PlaygroundSendControl
        {...casualDesktop}
        isConnectionReady={false}
        connectionPending
      />
    )

    expect(screen.getByRole("button", { name: "Send message" })).toHaveAttribute(
      "title",
      "Still connecting to your tldw server. Your draft has not been sent."
    )
  })

  it("asks to connect once a check has failed", () => {
    render(
      <PlaygroundSendControl
        {...casualDesktop}
        isConnectionReady={false}
        connectionPending={false}
      />
    )

    expect(screen.getByRole("button", { name: "Send message" })).toHaveAttribute(
      "title",
      "Connect to your tldw server before sending. Your draft has not been queued."
    )
  })
})

describe("hasSendableDraft", () => {
  it("is false for an empty or whitespace-only draft with nothing attached", () => {
    expect(hasSendableDraft({ message: "" })).toBe(false)
    expect(hasSendableDraft({ message: "   \n" })).toBe(false)
  })

  it("counts text, an image, attachments or imported page context", () => {
    expect(hasSendableDraft({ message: "hi" })).toBe(true)
    expect(hasSendableDraft({ message: "", image: "data:image/png;base64,x" })).toBe(true)
    expect(hasSendableDraft({ message: "", attachmentCount: 1 })).toBe(true)
    expect(hasSendableDraft({ message: "", hasImportedContext: true })).toBe(true)
  })

  it("counts pinned search results, which send on their own", () => {
    expect(hasSendableDraft({ message: "", pinnedResultCount: 2 })).toBe(true)
    expect(hasSendableDraft({ message: "", pinnedResultCount: 0 })).toBe(false)
  })
})
