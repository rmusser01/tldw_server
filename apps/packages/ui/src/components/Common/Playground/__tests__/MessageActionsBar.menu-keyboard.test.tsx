// @vitest-environment jsdom
/**
 * AX-02 (#3122): a reply's "More actions" is a menu button (WAI-ARIA menu
 * pattern): the trigger says it opens a menu, the actions are menu items, and
 * Arrow, Home and End keys move between them. Generation info opens from the
 * keyboard, not only on hover.
 */
import React from "react"
import type { TFunction } from "i18next"
import { ConfigProvider } from "antd"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { MessageActionsBar } from "../MessageActionsBar"

const translate = ((key: string, fallback?: string) => fallback ?? key) as TFunction

const props = (overrides: Partial<React.ComponentProps<typeof MessageActionsBar>> = {}) =>
  ({
    t: translate,
    isProMode: false,
    isBot: true,
    showVariantPager: false,
    resolvedVariantIndex: 0,
    variantCount: 1,
    canSwipePrev: false,
    canSwipeNext: false,
    actionRowVisibility: "flex",
    isTtsEnabled: false,
    isSpeaking: false,
    onToggleTts: vi.fn(),
    hideCopy: true,
    copyPressed: false,
    onCopy: vi.fn(),
    canReply: true,
    onReply: vi.fn(),
    canSaveToNotes: true,
    canSaveToFlashcards: true,
    canGenerateDocument: false,
    onGenerateDocument: vi.fn(),
    onSaveKnowledge: vi.fn(),
    savingKnowledge: null,
    isLastMessage: false,
    hideEditAndRegenerate: true,
    onRegenerate: vi.fn(),
    onEdit: vi.fn(),
    editMode: false,
    showFeedbackControls: false,
    feedbackDisabled: false,
    feedbackDisabledReason: "",
    isFeedbackSubmitting: false,
    showThanks: false,
    onThumbUp: vi.fn(),
    onThumbDown: vi.fn(),
    onOpenDetails: vi.fn(),
    onDelete: vi.fn(),
    ...overrides
  }) satisfies React.ComponentProps<typeof MessageActionsBar>

const mount = (overrides?: Partial<React.ComponentProps<typeof MessageActionsBar>>) =>
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <MessageActionsBar {...props(overrides)} />
    </ConfigProvider>
  )

const openMenu = async () => {
  const user = userEvent.setup()
  const trigger = screen.getByRole("button", { name: "More actions" })
  await user.click(trigger)
  const menu = await screen.findByRole("menu", { name: "More actions" })
  return { user, trigger, menu }
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("message More actions menu", () => {
  it("is a menu button whose actions are menu items", async () => {
    mount()
    const trigger = screen.getByRole("button", { name: "More actions" })
    expect(trigger).toHaveAttribute("aria-haspopup", "menu")
    expect(trigger).toHaveAttribute("aria-expanded", "false")

    const { menu } = await openMenu()
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    expect(trigger).toHaveAttribute("aria-controls", menu.id)
    const items = within(menu).getAllByRole("menuitem")
    expect(items.map((item) => item.textContent)).toEqual([
      "Reply",
      "Save to Notes",
      "Save to Flashcards",
      "Delete"
    ])
    await waitFor(() => expect(items[0]).toHaveFocus())
  })

  it("moves between actions with the Arrow, Home and End keys", async () => {
    mount()
    const { menu } = await openMenu()
    const items = within(menu).getAllByRole("menuitem")
    await waitFor(() => expect(items[0]).toHaveFocus())

    fireEvent.keyDown(items[0], { key: "ArrowDown" })
    expect(items[1]).toHaveFocus()
    fireEvent.keyDown(items[1], { key: "End" })
    expect(items[3]).toHaveFocus()
    fireEvent.keyDown(items[3], { key: "ArrowDown" })
    expect(items[0]).toHaveFocus()
    fireEvent.keyDown(items[0], { key: "ArrowUp" })
    expect(items[3]).toHaveFocus()
    fireEvent.keyDown(items[3], { key: "Home" })
    expect(items[0]).toHaveFocus()
  })

  it("runs the focused action with Enter and returns focus to the trigger on Escape", async () => {
    const onReply = vi.fn()
    mount({ onReply })
    const { user, trigger, menu } = await openMenu()
    const [reply] = within(menu).getAllByRole("menuitem")
    await waitFor(() => expect(reply).toHaveFocus())
    await user.keyboard("{Enter}")
    expect(onReply).toHaveBeenCalledTimes(1)

    await user.click(trigger)
    const reopened = await screen.findByRole("menu", { name: "More actions" })
    await waitFor(() => expect(within(reopened).getAllByRole("menuitem")[0]).toHaveFocus())
    await user.keyboard("{Escape}")
    await waitFor(() => expect(trigger).toHaveFocus())
    expect(trigger).toHaveAttribute("aria-expanded", "false")
  })
})

// AX-02 review: the rest of the WAI-ARIA menu pattern.
describe("message More actions menu pattern", () => {
  it("keeps its items out of the Tab order (roving focus)", async () => {
    mount()
    const { menu } = await openMenu()
    for (const item of within(menu).getAllByRole("menuitem")) {
      expect(item).toHaveAttribute("tabindex", "-1")
    }
  })

  it("closes on Tab and returns focus to the trigger", async () => {
    mount()
    const { user, trigger, menu } = await openMenu()
    await waitFor(() => expect(within(menu).getAllByRole("menuitem")[0]).toHaveFocus())

    await user.tab()
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(trigger).toHaveFocus()
  })

  it("closes when focus leaves it for another control, which keeps focus", async () => {
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <MessageActionsBar {...props()} />
        <button type="button">Elsewhere</button>
      </ConfigProvider>
    )
    const { trigger, menu } = await openMenu()
    await waitFor(() => expect(within(menu).getAllByRole("menuitem")[0]).toHaveFocus())
    const elsewhere = screen.getByRole("button", { name: "Elsewhere" })

    elsewhere.focus()
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(elsewhere).toHaveFocus()
  })

  it("keeps a disabled action reachable, says why, and does not run it", async () => {
    const onToggleTts = vi.fn()
    mount({
      isTtsEnabled: true,
      ttsActionDisabled: true,
      ttsDisabledReason: "No voices on the server.",
      onToggleTts
    })
    const { trigger, menu } = await openMenu()
    const items = within(menu).getAllByRole("menuitem")
    const readAloud = within(menu).getByRole("menuitem", { name: "Read Aloud" })
    expect(readAloud).toHaveAttribute("aria-disabled", "true")
    expect(readAloud).not.toBeDisabled()
    expect(readAloud).toHaveAccessibleDescription("No voices on the server.")

    const previous = items[items.indexOf(readAloud) - 1]
    previous.focus()
    fireEvent.keyDown(previous, { key: "ArrowDown" })
    expect(readAloud).toHaveFocus()
    fireEvent.click(readAloud)
    expect(onToggleTts).not.toHaveBeenCalled()
    expect(trigger).toHaveAttribute("aria-expanded", "true")
  })

  it("exposes role-play steering toggles as checkable items", async () => {
    mount({
      isLastMessage: true,
      onMessageSteeringModeChange: vi.fn(),
      onMessageSteeringForceNarrateChange: vi.fn(),
      messageSteeringMode: "continue_as_user",
      messageSteeringForceNarrate: false,
      onClearMessageSteering: vi.fn()
    })
    const { menu } = await openMenu()
    const checked = (name: string) =>
      within(menu).getByRole("menuitemcheckbox", { name }).getAttribute("aria-checked")
    expect(checked("Continue as user")).toBe("true")
    expect(checked("Impersonate user")).toBe("false")
    expect(checked("Force narrate")).toBe("false")
  })

  it("keeps steered continues that run at once as plain actions", async () => {
    mount({
      isLastMessage: true,
      onRunSteeredContinue: vi.fn(),
      onMessageSteeringModeChange: vi.fn(),
      onMessageSteeringForceNarrateChange: vi.fn(),
      messageSteeringMode: "none",
      messageSteeringForceNarrate: true
    })
    const { menu } = await openMenu()
    expect(within(menu).getByRole("menuitem", { name: "Continue as user" })).toBeInTheDocument()
    expect(within(menu).getByRole("menuitem", { name: "Impersonate user" })).toBeInTheDocument()
    expect(within(menu).getByRole("menuitemcheckbox", { name: "Force narrate" })).toHaveAttribute(
      "aria-checked",
      "true"
    )
  })
})

describe("generation info", () => {
  it("opens from keyboard focus, not only on hover", async () => {
    const user = userEvent.setup()
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <button type="button">Before</button>
        <MessageActionsBar
          {...props({ generationInfo: { eval_count: 42 } })}
        />
      </ConfigProvider>
    )
    screen.getByRole("button", { name: "Before" }).focus()
    await user.tab()
    expect(screen.getByRole("button", { name: "generationInfo" })).toHaveFocus()
    const metrics = await screen.findByText("eval_count", { exact: false })
    // antd mounts the popup hidden and shows it a moment later.
    await waitFor(() => expect(metrics).toBeVisible())
  })

  // A click must not close what hovering or the click's own focus opened.
  const settle = () => new Promise((resolve) => setTimeout(resolve, 300))

  it("stays open when a mouse user hovers and then clicks Info", async () => {
    const user = userEvent.setup()
    mount({ generationInfo: { eval_count: 42 } })
    const info = screen.getByRole("button", { name: "generationInfo" })
    await user.hover(info)
    const metrics = await screen.findByText("eval_count", { exact: false })
    await waitFor(() => expect(metrics).toBeVisible())
    await user.click(info)
    await settle()
    expect(metrics).toBeVisible()
  })

  it("stays open when Info has focus and is then clicked", async () => {
    const user = userEvent.setup()
    mount({ generationInfo: { eval_count: 42 } })
    const info = screen.getByRole("button", { name: "generationInfo" })
    info.focus()
    const metrics = await screen.findByText("eval_count", { exact: false })
    await waitFor(() => expect(metrics).toBeVisible())
    await user.click(info)
    await settle()
    expect(metrics).toBeVisible()
  })
})
