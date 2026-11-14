// @vitest-environment jsdom
/**
 * CM-09 (#3101, G17): the reply's More menu groups its actions (reuse,
 * transform, delete) with dividers, gives every action a real icon instead
 * of a single letter, and says why an unavailable action is disabled.
 */
import React from "react"
import { render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MessageActionsBar } from "../MessageActionsBar"

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Popover: ({
    children,
    content
  }: {
    children: React.ReactNode
    content?: React.ReactNode
  }) => (
    <>
      {children}
      <div data-testid="overflow-menu">{content}</div>
    </>
  )
}))

const baseProps = () =>
  ({
    t: ((key: string, fallback?: string) => fallback || key) as any,
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
    copyPressed: false,
    onCopy: vi.fn(),
    canReply: false,
    onReply: vi.fn(),
    canSaveToNotes: true,
    canSaveToFlashcards: false,
    canGenerateDocument: false,
    onGenerateDocument: vi.fn(),
    onSaveKnowledge: vi.fn(),
    savingKnowledge: null,
    isLastMessage: true,
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

const menu = () => screen.getByTestId("overflow-menu")

/** Labels of the menu's actions, with dividers shown as "|". */
const menuOutline = () =>
  Array.from(menu().firstElementChild!.children).flatMap((node) => {
    if (node.getAttribute("role") === "separator") return ["|"]
    if (node.tagName === "BUTTON") return [node.textContent ?? ""]
    return []
  })

describe("MessageActionsBar overflow menu", () => {
  it("separates reuse, transform and delete actions with dividers", () => {
    render(
      <MessageActionsBar
        {...baseProps()}
        onNewBranch={vi.fn()}
        onContinue={vi.fn()}
        onQuickMessageAction={vi.fn()}
        onDelete={vi.fn()}
      />
    )

    expect(menuOutline()).toEqual([
      "New Branch",
      "Continue",
      "Save to Notes",
      "|",
      "Summarize this",
      "Translate this",
      "Make this shorter",
      "Explain this",
      "|",
      "Delete"
    ])
  })

  it("gives every action, including role-play steering, an icon instead of a letter", () => {
    render(
      <MessageActionsBar
        {...baseProps()}
        onQuickMessageAction={vi.fn()}
        onMessageSteeringModeChange={vi.fn()}
        onMessageSteeringForceNarrateChange={vi.fn()}
        messageSteeringMode="continue_as_user"
        onClearMessageSteering={vi.fn()}
      />
    )

    const actions = within(menu()).getAllByRole("button")
    expect(actions.length).toBeGreaterThan(8)
    for (const action of actions) {
      expect(action.querySelector("svg"), action.textContent ?? "").not.toBeNull()
    }
    for (const label of [
      "Continue as user",
      "Impersonate user",
      "Force narrate",
      "Summarize this",
      "Translate this",
      "Make this shorter",
      "Explain this"
    ]) {
      expect(within(menu()).getByRole("button", { name: label })).toBeInTheDocument()
    }
  })

  it("explains why Read aloud is unavailable", () => {
    const reason = "No TTS voices are available on the server."
    render(
      <MessageActionsBar
        {...baseProps()}
        isTtsEnabled
        ttsActionDisabled
        ttsDisabledReason={reason}
      />
    )

    const readAloud = within(menu()).getByRole("button", { name: "Read Aloud" })
    expect(readAloud).toBeDisabled()
    expect(readAloud).toHaveAccessibleDescription(reason)
    expect(within(menu()).getByText(reason)).toBeVisible()
  })
})
