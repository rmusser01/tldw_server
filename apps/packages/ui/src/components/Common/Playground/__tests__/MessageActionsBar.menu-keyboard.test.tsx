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
    expect(await screen.findByText("eval_count", { exact: false })).toBeVisible()
  })
})
