// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundSendControl } from "../PlaygroundSendControl"

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
})
