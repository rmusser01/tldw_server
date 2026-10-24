import React from "react"
import type { TFunction } from "i18next"
import { ConfigProvider, Popover, Tooltip } from "antd"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { IconButton } from "../../IconButton"
import { MessageActionsBar } from "../MessageActionsBar"

const translate = ((key: string, fallback?: string) => fallback ?? key) as TFunction

const mount = (children: React.ReactNode) => render(
  <ConfigProvider theme={{ token: { motion: false } }}>{children}</ConfigProvider>
)

const completedAssistant = () => ({
  t: translate,
  isProMode: false,
  isBot: true,
  showVariantPager: false,
  resolvedVariantIndex: 0,
  variantCount: 1,
  canSwipePrev: false,
  canSwipeNext: false,
  overflowChipVisibility: "hidden",
  actionRowVisibility: "flex",
  isTtsEnabled: false,
  isSpeaking: false,
  onToggleTts: vi.fn(),
  hideCopy: true,
  copyPressed: false,
  onCopy: vi.fn(),
  canReply: false,
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
  onOpenDetails: vi.fn()
}) satisfies React.ComponentProps<typeof MessageActionsBar>

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("real AntD message action triggers", () => {
  it("opens a native button Popover as an independent trigger fixture control", async () => {
    const user = userEvent.setup()
    mount(<Popover trigger="click" content={<button type="button">Native menu choice</button>}>
      <button type="button">Native More actions</button>
    </Popover>)
    expect(screen.queryByRole("button", { name: "Native menu choice" })).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Native More actions" }))
    expect(await screen.findByRole("button", { name: "Native menu choice" })).toBeVisible()
  })

  it.each([
    { label: "Save to Notes", makeFlashcard: false },
    { label: "Save to Flashcards", makeFlashcard: true }
  ])("opens the actual assistant More button and dispatches $label once", async ({ label, makeFlashcard }) => {
    const user = userEvent.setup()
    const props = completedAssistant()
    mount(<MessageActionsBar {...props} />)
    const trigger = screen.getAllByRole("button", { name: "More actions" })
      .find((button) => button.dataset.testid !== "message-actions-overflow-chip")
    if (!trigger) throw new Error("Actual IconButton More trigger is missing")
    expect(trigger.querySelector("svg.lucide-ellipsis")).not.toBeNull()
    expect(screen.queryByRole("button", { name: "Save to Notes" })).not.toBeInTheDocument()
    expect(props.onSaveKnowledge).not.toHaveBeenCalled()
    await user.click(trigger)
    const choice = await screen.findByRole("button", { name: label })
    await waitFor(() => expect(choice).toBeVisible())
    expect(screen.getByRole("button", { name: "Save to Notes" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Save to Flashcards" })).toBeVisible()
    await user.click(choice)
    expect(props.onSaveKnowledge).toHaveBeenCalledTimes(1)
    expect(props.onSaveKnowledge).toHaveBeenCalledWith(makeFlashcard)
    await waitFor(() => expect(choice).not.toBeVisible())
  })

  it("opens a native button Tooltip on hover as an independent event fixture control", async () => {
    const user = userEvent.setup()
    mount(<Tooltip title="Native hover explanation"><button type="button">Native hover</button></Tooltip>)
    await user.hover(screen.getByRole("button", { name: "Native hover" }))
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Native hover explanation")
  })

  it("opens the shared IconButton Tooltip on hover through the injected handler", async () => {
    const user = userEvent.setup()
    mount(<Tooltip title="Shared hover explanation"><IconButton ariaLabel="Shared hover">Icon</IconButton></Tooltip>)
    const trigger = screen.getByRole("button", { name: "Shared hover" })
    await user.hover(trigger)
    const tooltip = await screen.findByRole("tooltip")
    expect(tooltip).toHaveTextContent("Shared hover explanation")
    expect(tooltip.id).not.toBe("")
    expect(trigger).toHaveAttribute("aria-describedby", tooltip.id)
    await user.unhover(trigger)
    await waitFor(() => expect(tooltip).not.toBeVisible())
  })

  it("opens a native button Tooltip on keyboard focus as an independent event fixture control", async () => {
    const user = userEvent.setup()
    mount(<Tooltip trigger="focus" title="Native focus explanation"><button type="button">Native focus</button></Tooltip>)
    await user.tab()
    expect(screen.getByRole("button", { name: "Native focus" })).toHaveFocus()
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Native focus explanation")
  })

  it("opens the shared IconButton Tooltip on keyboard focus through the injected handler", async () => {
    const user = userEvent.setup()
    mount(<>
      <Tooltip trigger="focus" title="Shared focus explanation"><IconButton ariaLabel="Shared focus">Icon</IconButton></Tooltip>
      <button type="button">Next focus target</button>
    </>)
    await user.tab()
    expect(screen.getByRole("button", { name: "Shared focus" })).toHaveFocus()
    const tooltip = await screen.findByRole("tooltip")
    expect(tooltip).toHaveTextContent("Shared focus explanation")
    await user.tab()
    expect(screen.getByRole("button", { name: "Next focus target" })).toHaveFocus()
    await waitFor(() => expect(tooltip).not.toBeVisible())
  })

  it("keeps native default type, DOM ref, explicit click and disabled guards", async () => {
    const user = userEvent.setup()
    const ref = React.createRef<HTMLButtonElement>()
    const onClick = vi.fn()
    const onSubmit = vi.fn((event: React.FormEvent) => event.preventDefault())
    const view = mount(<form onSubmit={onSubmit}>
      <IconButton ref={ref} ariaLabel="Guarded action" onClick={onClick}>Icon</IconButton>
    </form>)
    const button = screen.getByRole("button", { name: "Guarded action" })
    expect(ref.current).toBe(button)
    expect(button).toHaveAttribute("type", "button")
    expect(button).toHaveAttribute("title", "Guarded action")
    await user.click(button)
    expect(onClick).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalled()
    view.rerender(<ConfigProvider theme={{ token: { motion: false } }}><form onSubmit={onSubmit}>
      <IconButton ref={ref} ariaLabel="Guarded action" onClick={onClick} disabled>Icon</IconButton>
    </form></ConfigProvider>)
    expect(button).toBeDisabled()
    expect(ref.current).toBe(button)
    await user.click(button)
    expect(onClick).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it("retains standard injected description and open-state attributes when aliases are absent", () => {
    mount(<>
      <IconButton ariaLabel="Described action" aria-describedby="action-description"
        aria-haspopup="menu" aria-expanded={false} aria-controls="action-menu">Icon</IconButton>
      <p id="action-description">Action explanation</p>
      <div id="action-menu" />
    </>)
    const button = screen.getByRole("button", { name: "Described action" })
    expect(button).toHaveAccessibleDescription("Action explanation")
    expect(button).toHaveAttribute("aria-haspopup", "menu")
    expect(button).toHaveAttribute("aria-expanded", "false")
    expect(button).toHaveAttribute("aria-controls", "action-menu")
  })

  it("preserves supplied legacy aliases and mandatory label precedence", () => {
    mount(<IconButton ariaLabel="Legacy action" aria-label="Ignored standard label"
      hasPopup="dialog" aria-haspopup="menu" ariaExpanded aria-expanded={false}
      ariaControls="legacy-dialog" aria-controls="standard-menu"
      dataTestId="legacy-action" title="Explicit action title">Icon</IconButton>)
    const button = screen.getByRole("button", { name: "Legacy action" })
    expect(button).toHaveAttribute("aria-haspopup", "dialog")
    expect(button).toHaveAttribute("aria-expanded", "true")
    expect(button).toHaveAttribute("aria-controls", "legacy-dialog")
    expect(button).toHaveAttribute("data-testid", "legacy-action")
    expect(button).toHaveAttribute("title", "Explicit action title")
  })
})
