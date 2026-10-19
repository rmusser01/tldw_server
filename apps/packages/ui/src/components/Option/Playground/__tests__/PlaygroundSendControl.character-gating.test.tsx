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
  it("B02 disables character Send while the selected card is loading", async () => {
    const onAction = vi.fn()
    const onSubmitForm = vi.fn()
    const blocker = { active: true, title: "Loading character. Your draft is retained.", actionLabel: "Loading character", disabled: true, onAction }
    render(<PlaygroundSendControl {...baseProps} onSubmitForm={onSubmitForm} characterChatSendBlocker={blocker} />)
    const button = screen.getByRole("button", { name: "Loading character" })
    expect(button).toBeDisabled()
    await userEvent.setup().click(button)
    expect(onAction).not.toHaveBeenCalled()
    expect(onSubmitForm).not.toHaveBeenCalled()

})

  it.each([false, true])("keeps unsupported history Send and Queue unavailable (busy=%s)", async isSending => {
    const submit = vi.fn()
    const user = userEvent.setup()
    render(<PlaygroundSendControl {...baseProps} isSending={isSending} onSubmitForm={submit}
      submitDisabledReason="Tracked persona chat is unavailable here. Use a persona prompt overlay in a regular chat." />)
    const button = screen.getByRole("button", { name: isSending ? /queue/i : /send/i })
    expect(button).toBeDisabled()
    expect(button).toHaveAttribute("title", expect.stringContaining("Tracked persona chat is unavailable"))
    await user.click(button)
    expect(submit).not.toHaveBeenCalled()
  })

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

  it("keeps the mobile blocker action compact so the composer input keeps width", () => {
    render(
      <PlaygroundSendControl
        {...baseProps}
        isMobileViewport
        characterChatSendBlocker={{
          active: true,
          title: "Configure the selected model provider before chatting as Ada",
          actionLabel: "Open model settings",
          onAction: vi.fn()
        }}
      />
    )

    expect(
      screen.getByRole("button", { name: "Open model settings" })
    ).toBeInTheDocument()
    expect(screen.queryByText("Open model settings")).not.toBeInTheDocument()
  })
})
