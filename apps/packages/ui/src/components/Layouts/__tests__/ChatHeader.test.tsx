import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import userEvent from "@testing-library/user-event"
import type { TFunction } from "i18next"
import { ChatHeader } from "../ChatHeader"

vi.mock("antd", async (importOriginal) => ({
  ...(await importOriginal<typeof import("antd")>()),
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Input: ({ value, onChange, onPressEnter, ...rest }: any) => (
    <input
      value={value}
      onChange={onChange}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          onPressEnter?.(event)
        }
      }}
      {...rest}
    />
  )
}))

vi.mock("~/assets/icon.png", () => ({
  default: "icon.png"
}))

vi.mock("../HeaderShortcuts", () => ({
  HeaderShortcuts: ({ expanded }: { expanded: boolean }) => (
    <div
      data-testid="header-shortcuts"
      data-expanded={expanded ? "true" : "false"}
    />
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

const createProps = (
  overrides: Partial<React.ComponentProps<typeof ChatHeader>> = {}
) => ({
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
  onOpenCompanionHome: vi.fn(),
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

  it("shows signpost button with 'Show pages' when collapsed and requests expand", () => {
    const props = createProps({ shortcutsExpanded: false })
    render(<ChatHeader {...props} showChatTitle={false} />)

    const toggleButton = screen.getByRole("button", { name: "Show pages" })
    expect(toggleButton).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(toggleButton)

    expect(props.onToggleShortcuts).toHaveBeenCalledWith(true)
    expect(screen.getByTestId("header-shortcuts")).toHaveAttribute(
      "data-expanded",
      "false"
    )
  })

  it("shows signpost button with 'Hide pages' when expanded and requests collapse", () => {
    const props = createProps({ shortcutsExpanded: true })
    render(<ChatHeader {...props} showChatTitle={false} />)

    const toggleButton = screen.getByRole("button", { name: "Hide pages" })
    expect(toggleButton).toHaveAttribute("aria-expanded", "true")
    fireEvent.click(toggleButton)

    expect(props.onToggleShortcuts).toHaveBeenCalledWith(false)
    expect(screen.getByTestId("header-shortcuts")).toHaveAttribute(
      "data-expanded",
      "true"
    )
  })

  it("shows a Companion Home button immediately before the signpost shortcut toggle", () => {
    const props = createProps()
    render(<ChatHeader {...props} showChatTitle={false} />)

    const homeButton = screen.getByRole("button", { name: "Companion Home" })
    const signpostButton = screen.getByRole("button", {
      name: "Show pages"
    })
    const headerBrand = homeButton.closest(".flex.items-center.gap-2.text-text")
    const headerButtons: HTMLElement[] = Array.from(
      headerBrand?.querySelectorAll("button") ?? []
    )

    expect(headerBrand).toBeTruthy()
    expect(headerBrand).toContainElement(homeButton)
    expect(headerBrand).toContainElement(signpostButton)
    expect(headerButtons.indexOf(signpostButton)).toBe(
      headerButtons.indexOf(homeButton) + 1
    )

    fireEvent.click(homeButton)
    expect(props.onOpenCompanionHome).toHaveBeenCalledTimes(1)
  })

  it("keeps the conversation title below the route heading", () => {
    render(<ChatHeader {...createProps()} />)
    expect(screen.getByRole("heading", { name: "Chat title", level: 2 })).toBeVisible()
  })

  it("keeps visible focus rings on primary conversation controls", () => {
    render(<ChatHeader {...createProps()} />)
    for (const name of [
      "Collapse sidebar",
      "Search commands - Open command palette",
      "New saved chat",
      "More actions"
    ]) {
      expect(screen.getByRole("button", { name }).className).toContain(
        "focus-visible:ring-2"
      )
    }
  })

  it("labels the search trigger as the command palette opener and keeps the shortcut hint visible", () => {
    const props = createProps()
    render(<ChatHeader {...props} />)

    const trigger = screen.getByRole("button", {
      name: "Search commands - Open command palette"
    })

    expect(trigger).toHaveAttribute("title", "Search commands")
    expect(trigger).toHaveTextContent("Search commands")
    expect(trigger).toHaveTextContent("Ctrl+K")

    fireEvent.click(trigger)
    expect(props.onOpenCommandPalette).toHaveBeenCalledTimes(1)
  })

  it("offers the theme toggle through More and preserves the existing action", async () => {
    const props = createProps({ themeMode: "light" })
    render(<ChatHeader {...props} />)
    fireEvent.click(screen.getByRole("button", { name: "More actions" }))
    fireEvent.click(
      await screen.findByRole("menuitem", { name: "Switch to dark theme" })
    )
    expect(props.onToggleTheme).toHaveBeenCalledTimes(1)
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
      screen.queryByRole("button", { name: "Temporary chat (on this device)" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Character chat" })
    ).not.toBeInTheDocument()
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
      persistenceKind: "server",
      activeCharacterName: "Rin"
    })
    render(<ChatHeader {...props} />)

    expect(
      screen.queryByTestId("chat-header-persistence-badge")
    ).not.toBeInTheDocument()
    expect(screen.queryByText("Saved on server")).not.toBeInTheDocument()
    expect(screen.queryByText("Character: Rin")).not.toBeInTheDocument()
  })

  // CS-03 (#3104): the header badge reports where the chat is actually saved,
  // derived from acknowledged state rather than from server connectivity.
  it("labels a saved chat without an acknowledged server copy as saved on this device", () => {
    render(<ChatHeader {...createProps({ temporaryChat: false })} />)

    const badge = screen.getByTestId("chat-header-persistence-badge")
    expect(badge).toHaveTextContent("Saved on this device")
    expect(badge).toHaveAttribute(
      "title",
      "Saved on this device only. This chat is not on your tldw server."
    )
    expect(screen.queryByText("Saved on server")).not.toBeInTheDocument()
  })

  it.each(["server", "temporary", "serverFailed"] as const)("keeps %s status labels readable with semantic backgrounds", persistenceKind => {
    render(<ChatHeader {...createProps({ temporaryChat: persistenceKind === "temporary", persistenceKind, themeMode: "light" })} />)
    const badge = screen.getByTestId("chat-header-persistence-badge")
    expect(badge.classList.contains("text-text")).toBe(true)
    expect(badge.className).toContain(persistenceKind === "server" ? "bg-success/10" : "bg-warn/10")
  })

  it("labels a chat whose server write was acknowledged as saved on server", () => {
    render(
      <ChatHeader
        {...createProps({ temporaryChat: false, persistenceKind: "server" })}
      />
    )

    const badge = screen.getByTestId("chat-header-persistence-badge")
    expect(badge).toHaveTextContent("Saved on server")
    expect(badge).toHaveAttribute(
      "title",
      "Saved on your tldw server and on this device."
    )
  })

  it.each([
    ["serverSaving", "Saving to server…"],
    ["serverFailed", "Couldn't save to server"]
  ] as const)(
    "does not claim the server while the server write is %s",
    (persistenceKind, label) => {
      render(
        <ChatHeader
          {...createProps({ temporaryChat: false, persistenceKind })}
        />
      )

      expect(
        screen.getByTestId("chat-header-persistence-badge")
      ).toHaveTextContent(label)
      expect(screen.queryByText("Saved on server")).not.toBeInTheDocument()
    }
  )

  it("keeps the temporary badge for temporary chats", () => {
    render(
      <ChatHeader
        {...createProps({ temporaryChat: true, persistenceKind: "temporary" })}
      />
    )

    const badge = screen.getByTestId("chat-header-persistence-badge")
    expect(badge).toHaveTextContent("Temporary")
    expect(badge).toHaveAttribute(
      "title",
      "Temporary messages can remain on this device across reloads until you close or clear the chat. They are not added to server history. Draft text may remain on this device."
    )
  })

  it("hides chat title when chat title display is disabled", () => {
    const props = createProps({
      showChatTitle: false,
      chatTitle: "Chat title"
    })
    render(<ChatHeader {...props} />)

    expect(screen.queryByText("Chat title")).not.toBeInTheDocument()
  })

  it("labels the title editor and keeps an untitled placeholder visible", () => {
    const props = createProps({
      chatTitle: "",
      isEditingTitle: true
    })

    render(<ChatHeader {...props} />)

    const titleInput = screen.getByRole("textbox", {
      name: "Rename conversation"
    })
    expect(titleInput).toHaveAttribute("placeholder", "Untitled")
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

describe("calm conversation header", () => {
  it("hands overlays a persistent focus-return target when launched from More", async () => {
    const user = userEvent.setup()
    let returnTarget: Element | null = null
    const props = createProps({
      onToggleShortcuts: () => {
        returnTarget = document.activeElement
      }
    })
    render(<ChatHeader {...props} />)
    const more = screen.getByRole("button", { name: "More actions" })
    await user.click(more)
    await user.click(
      await screen.findByRole("menuitem", { name: "Browse pages" })
    )
    expect(returnTarget).toBe(more)
  })

  it("leads with the conversation title as a heading beside persistence state", () => {
    render(<ChatHeader {...createProps()} />)
    expect(screen.getByRole("heading", { name: "Chat title" })).toBeVisible()
    expect(
      screen.getByTestId("chat-header-persistence-badge")
    ).toHaveTextContent("Saved on this device")
  })

  it("reveals secondary utilities through More and runs the existing settings action", async () => {
    const props = createProps()
    render(<ChatHeader {...props} />)
    expect(screen.queryByRole("menuitem", { name: "Open settings" })).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "More actions" }))
    fireEvent.click(
      await screen.findByRole("menuitem", { name: "Open settings" })
    )
    expect(props.onOpenSettings).toHaveBeenCalledTimes(1)
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "More actions" })
      ).toHaveAttribute("aria-expanded", "false")
    )
  })

  it("keeps saved chat direct while temporary and character entry remain available in More", async () => {
    const props = createProps()
    render(<ChatHeader {...props} />)
    fireEvent.click(screen.getByRole("button", { name: "New saved chat" }))
    expect(props.onStartSavedChat).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByRole("button", { name: "More actions" }))
    fireEvent.click(
      await screen.findByRole("menuitem", {
        name: "Temporary chat (on this device)"
      })
    )
    expect(props.onStartTemporaryChat).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByRole("button", { name: "More actions" }))
    fireEvent.click(
      await screen.findByRole("menuitem", { name: "Character chat" })
    )
    expect(props.onStartCharacterChat).toHaveBeenCalledTimes(1)
  })

  it("returns keyboard focus to More when the menu is dismissed with Escape", async () => {
    const user = userEvent.setup()
    render(<ChatHeader {...createProps()} />)
    const more = screen.getByRole("button", { name: "More actions" })
    await user.click(more)
    await screen.findByRole("menu")
    await user.keyboard("{Escape}")
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "More actions" })
      ).toHaveAttribute("aria-expanded", "false")
    )
    await waitFor(() => expect(more).toHaveFocus())
  })

  it("retains the existing app navigation outside the chat route", () => {
    render(
      <ChatHeader
        {...createProps({ showChatTitle: false, showSessionModeBadge: false })}
      />
    )
    expect(screen.getByText("tldw Assistant")).toBeVisible()
    expect(screen.getByRole("button", { name: "Companion Home" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Open settings" })).toBeVisible()
  })
})

it("labels history eligibility without claiming an acknowledged save", () => {
  render(
    <ChatHeader {...createProps({ historyId: null, temporaryChat: false })} />
  )
  expect(screen.getByText("History on")).toHaveAttribute(
    "title",
    "New messages can be stored in chat history."
  )
  expect(screen.queryByText("Saved")).not.toBeInTheDocument()
})

it("explains Temporary history without promising removal of retained local drafts", () => {
  render(<ChatHeader {...createProps({ temporaryChat: true })} />)
  expect(screen.getByText("Temporary")).toHaveAttribute(
    "title",
    "Temporary messages can remain on this device across reloads until you close or clear the chat. They are not added to server history. Draft text may remain on this device."
  )
})
