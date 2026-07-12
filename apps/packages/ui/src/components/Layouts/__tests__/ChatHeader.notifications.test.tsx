import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"

import { ChatHeader } from "../ChatHeader"

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />
}))
vi.mock("~/assets/icon.png", () => ({ default: "icon.png" }))
vi.mock("../HeaderShortcuts", () => ({ HeaderShortcuts: () => null }))

const t = ((_: string, fallback?: string) => fallback || "") as TFunction

const props = (overrides: Partial<React.ComponentProps<typeof ChatHeader>> = {}) => ({
  t,
  temporaryChat: false,
  historyId: "history-1",
  chatTitle: "Chat",
  isEditingTitle: false,
  onTitleChange: vi.fn(),
  onTitleEditStart: vi.fn(),
  onTitleCommit: vi.fn(),
  onOpenCompanionHome: vi.fn(),
  onOpenCommandPalette: vi.fn(),
  onOpenShortcutsModal: vi.fn(),
  onOpenSettings: vi.fn(),
  shortcutsExpanded: false,
  onToggleShortcuts: vi.fn(),
  commandKeyLabel: "Ctrl+",
  onOpenNotifications: vi.fn(),
  notificationState: "active" as const,
  notificationCount: 3,
  onRetryNotifications: vi.fn(),
  ...overrides
})

describe("ChatHeader notification lifecycle", () => {
  it("names the active count and hides the visual badge from assistive technology", () => {
    render(<ChatHeader {...props()} />)

    const trigger = screen.getByRole("button", { name: "Notifications, 3 unread" })
    expect(trigger).toHaveAttribute("aria-haspopup", "dialog")
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    expect(trigger).toHaveAttribute("aria-controls", "chat-header-notification-status")
    expect(screen.getByText("3")).toHaveAttribute("aria-hidden", "true")
  })

  it.each([
    ["connecting", "Notifications, connecting"],
    ["degraded", "Notifications, reconnecting"],
    ["auth-required", "Notifications, sign-in required"],
    ["unavailable", "Notifications unavailable"]
  ] as const)("exposes %s with a truthful accessible name", (state, name) => {
    render(<ChatHeader {...props({ notificationState: state, notificationCount: 0 })} />)
    expect(screen.getByRole("button", { name })).toBeEnabled()
  })

  it("opens with native keyboard activation, retries once, and returns focus on Escape", async () => {
    const user = userEvent.setup()
    const onRetryNotifications = vi.fn()
    render(
      <ChatHeader
        {...props({ notificationState: "unavailable", onRetryNotifications })}
      />
    )
    const trigger = screen.getByRole("button", { name: "Notifications unavailable" })
    trigger.focus()
    await user.keyboard("{Enter}")

    expect(trigger).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByRole("dialog", { name: "Notifications unavailable" })).toBeInTheDocument()
    await user.keyboard("{Escape}")
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()

    await user.keyboard(" ")
    await user.click(screen.getByRole("button", { name: "Try again" }))
    expect(onRetryNotifications).toHaveBeenCalledTimes(1)

    await user.keyboard("{Escape}")
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })

  it("announces a lifecycle transition once through a polite status region", () => {
    const view = render(<ChatHeader {...props({ notificationState: "connecting" })} />)
    view.rerender(<ChatHeader {...props({ notificationState: "degraded" })} />)

    const status = screen.getByRole("status")
    expect(status).toHaveAttribute("aria-live", "polite")
    expect(status).toHaveTextContent("Notifications are reconnecting")
  })
})
