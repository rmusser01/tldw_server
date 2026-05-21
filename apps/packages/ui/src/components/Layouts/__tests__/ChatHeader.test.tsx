import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"
import { ChatHeader } from "../ChatHeader"

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Input: ({ value, onChange, ...rest }: any) => (
    <input
      value={value}
      onChange={onChange}
      {...rest}
    />
  )
}))

vi.mock("~/assets/icon.png", () => ({
  default: "icon.png"
}))

vi.mock("../HeaderShortcuts", () => ({
  HeaderShortcuts: ({ expanded }: { expanded: boolean }) => (
    <div data-testid="header-shortcuts" data-expanded={expanded ? "true" : "false"} />
  )
}))

const t = ((
  key: string,
  fallback?: string,
  values?: Record<string, unknown>
) => {
  const base = fallback || key
  if (!values) return base
  return Object.entries(values).reduce((acc, [name, value]) => {
    return acc.replaceAll(`{{${name}}}`, String(value))
  }, base)
}) as unknown as TFunction

const createProps = (overrides: Partial<React.ComponentProps<typeof ChatHeader>> = {}) => ({
  t,
  temporaryChat: false,
  historyId: "history-1",
  chatTitle: "Chat title",
  isEditingTitle: false,
  onTitleChange: vi.fn(),
  onTitleEditStart: vi.fn(),
  onTitleCommit: vi.fn(),
  onToggleSidebar: vi.fn(),
  sidebarCollapsed: false,
  onOpenCommandPalette: vi.fn(),
  onOpenShortcutsModal: vi.fn(),
  onOpenSettings: vi.fn(),
  onToggleTheme: vi.fn(),
  themeMode: "dark" as const,
  onStartSavedChat: vi.fn(),
  onStartTemporaryChat: vi.fn(),
  onStartCharacterChat: vi.fn(),
  activeCharacterName: null,
  shortcutsExpanded: false,
  onToggleShortcuts: vi.fn(),
  commandKeyLabel: "Ctrl+",
  ...overrides
})

describe("ChatHeader shortcut toggle", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("shows signpost button with 'Show shortcuts' when collapsed and requests expand", () => {
    const props = createProps({ shortcutsExpanded: false })
    render(<ChatHeader {...props} />)

    const toggleButton = screen.getByRole("button", { name: "Show shortcuts" })
    expect(toggleButton).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(toggleButton)

    expect(props.onToggleShortcuts).toHaveBeenCalledWith(true)
    expect(screen.getByTestId("header-shortcuts")).toHaveAttribute(
      "data-expanded",
      "false"
    )
  })

  it("shows signpost button with 'Hide shortcuts' when expanded and requests collapse", () => {
    const props = createProps({ shortcutsExpanded: true })
    render(<ChatHeader {...props} />)

    const toggleButton = screen.getByRole("button", { name: "Hide shortcuts" })
    expect(toggleButton).toHaveAttribute("aria-expanded", "true")
    fireEvent.click(toggleButton)

    expect(props.onToggleShortcuts).toHaveBeenCalledWith(false)
    expect(screen.getByTestId("header-shortcuts")).toHaveAttribute(
      "data-expanded",
      "true"
    )
  })

  it("applies focus-visible ring classes to key header controls", () => {
    const props = createProps()
    render(<ChatHeader {...props} />)

    const controls = [
      screen.getByRole("button", { name: "Collapse sidebar" }),
      screen.getByRole("button", { name: "Show shortcuts" }),
      screen.getByRole("button", { name: "New saved chat" }),
      screen.getByRole("button", { name: "Temporary chat (not saved)" }),
      screen.getByRole("button", { name: "Character chat" }),
      screen.getByRole("button", { name: "Open settings" }),
      screen.getByRole("button", { name: "Switch to light theme" }),
      screen.getByRole("button", { name: "Show keyboard shortcuts" })
    ]

    for (const control of controls) {
      expect(control.className).toContain("focus-visible:ring-2")
      expect(control.className).toContain("focus-visible:ring-focus")
      expect(control.className).toContain("focus-visible:ring-offset-2")
      expect(control.className).toContain("focus-visible:ring-offset-bg")
    }
  })

  it("shows a theme toggle control and triggers toggle callback", () => {
    const props = createProps({ themeMode: "light" })
    render(<ChatHeader {...props} />)

    const toggleButton = screen.getByRole("button", {
      name: "Switch to dark theme"
    })
    fireEvent.click(toggleButton)

    expect(props.onToggleTheme).toHaveBeenCalledTimes(1)
  })

  it("routes chat mode actions to the dedicated callbacks", () => {
    const props = createProps()
    render(<ChatHeader {...props} />)

    fireEvent.click(screen.getByRole("button", { name: "New saved chat" }))
    fireEvent.click(
      screen.getByRole("button", { name: "Temporary chat (not saved)" })
    )
    fireEvent.click(screen.getByRole("button", { name: "Character chat" }))

    expect(props.onStartSavedChat).toHaveBeenCalledTimes(1)
    expect(props.onStartTemporaryChat).toHaveBeenCalledTimes(1)
    expect(props.onStartCharacterChat).toHaveBeenCalledTimes(1)
  })

  it("hides chat session actions when callbacks are omitted", () => {
    render(
      <ChatHeader
        {...createProps({
          onStartSavedChat: undefined,
          onStartTemporaryChat: undefined,
          onStartCharacterChat: undefined
        })}
      />
    )

    expect(
      screen.queryByRole("button", { name: "New saved chat" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Temporary chat (not saved)" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Character chat" })
    ).not.toBeInTheDocument()
  })

  it("hides temporary and character quick actions below the small breakpoint", () => {
    const props = createProps()
    render(<ChatHeader {...props} />)

    expect(
      screen.getByRole("button", { name: "Temporary chat (not saved)" }).className
    ).toContain("hidden")
    expect(
      screen.getByRole("button", { name: "Temporary chat (not saved)" }).className
    ).toContain("sm:inline-flex")
    expect(screen.getByRole("button", { name: "Character chat" }).className).toContain(
      "hidden"
    )
    expect(screen.getByRole("button", { name: "Character chat" }).className).toContain(
      "sm:inline-flex"
    )
  })

  it("allows the chat header control row to wrap on narrow screens", () => {
    const props = createProps()
    render(<ChatHeader {...props} />)

    const headerRow = screen
      .getByRole("button", { name: "New saved chat" })
      .closest("header > div")
    const actionsRow = screen
      .getByRole("button", { name: "New saved chat" })
      .parentElement

    expect(headerRow?.className).toContain("flex-wrap")
    expect(actionsRow?.className).toContain("flex-wrap")
  })

  it("shows mode badges for temporary and active character", () => {
    const props = createProps({
      temporaryChat: true,
      activeCharacterName: "Rin"
    })
    render(<ChatHeader {...props} />)

    expect(screen.getByText("Temporary")).toBeInTheDocument()
    expect(screen.getByText("Character: Rin")).toBeInTheDocument()
  })

  it("hides session mode badge when not on chat route", () => {
    const props = createProps({
      showSessionModeBadge: false,
      temporaryChat: false,
      activeCharacterName: "Rin"
    })
    render(<ChatHeader {...props} />)

    expect(screen.queryByText("Saved")).not.toBeInTheDocument()
    expect(screen.queryByText("Character: Rin")).not.toBeInTheDocument()
  })

  it("hides chat title when chat title display is disabled", () => {
    const props = createProps({
      showChatTitle: false,
      chatTitle: "Chat title"
    })
    render(<ChatHeader {...props} />)

    expect(screen.queryByText("Chat title")).not.toBeInTheDocument()
  })

  it("shows share controls and status when provided", () => {
    const onOpenShareModal = vi.fn()
    const props = createProps({
      onOpenShareModal,
      shareStatusLabel: "2 active link(s)"
    })
    render(<ChatHeader {...props} />)

    fireEvent.click(
      screen.getByRole("button", {
        name: "Share conversation (2 active link(s))"
      })
    )

    expect(onOpenShareModal).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId("chat-header-share-status")).toHaveTextContent(
      "2 active link(s)"
    )
  })
})
