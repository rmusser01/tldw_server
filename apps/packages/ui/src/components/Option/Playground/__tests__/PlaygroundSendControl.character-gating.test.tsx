// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundSendControl } from "../PlaygroundSendControl"

const t = (_key: string, fallback?: string) => fallback ?? _key

const baseProps = {
  isProMode: false,
  isMobileViewport: false,
  isSending: false,
  isConnectionReady: true,
  sendWhenEnter: true,
  onSendWhenEnterChange: vi.fn(),
  sendLabel: "SEND",
  compareNeedsMoreModels: false,
  onStopStreaming: vi.fn(),
  onStopListening: vi.fn(),
  onSubmitForm: vi.fn(),
  sendMenuOpen: false,
  onSendMenuChange: vi.fn(),
  t
}

describe("PlaygroundSendControl character chat gating", () => {
  it("turns blocked character SEND into a setup action without submitting", async () => {
    const user = userEvent.setup()
    const onSubmitForm = vi.fn()
    const onAction = vi.fn()
    const onStopListening = vi.fn()

    render(
      <PlaygroundSendControl
        {...baseProps}
        onSubmitForm={onSubmitForm}
        onStopListening={onStopListening}
        characterChatSendBlocker={{
          active: true,
          title: "Configure the selected model provider before chatting as Ada",
          actionLabel: "Open model settings",
          onAction
        }}
      />
    )

    await user.click(screen.getByRole("button", { name: /open model settings/i }))

    expect(onStopListening).toHaveBeenCalledTimes(1)
    expect(onAction).toHaveBeenCalledTimes(1)
    expect(onSubmitForm).not.toHaveBeenCalled()
  })
})
