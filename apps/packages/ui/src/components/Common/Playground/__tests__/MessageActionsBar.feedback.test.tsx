import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { MessageActionsBar } from "../MessageActionsBar"

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

const baseProps = () =>
  ({
    t: ((key: string, fallback?: string) => fallback || key) as any,
    isProMode: false,
    isBot: false,
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
    canSaveToNotes: false,
    canSaveToFlashcards: false,
    canGenerateDocument: false,
    onGenerateDocument: vi.fn(),
    onSaveKnowledge: vi.fn(),
    savingKnowledge: null,
    isLastMessage: false,
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

describe("MessageActionsBar feedback visibility", () => {
  it("shows feedback row for non-bot messages when showFeedbackControls is true", () => {
    const { container } = render(
      <MessageActionsBar
        {...baseProps()}
        showFeedbackControls
      />
    )

    expect(screen.getByText("Was this helpful?")).toBeInTheDocument()
    expect(container.firstElementChild).toHaveClass("justify-between")
  })

  // CM-10 (#3101): each reply shows one "..." (More actions). The feedback
  // details button keeps its place and name but uses its own icon.
  it("shows a single ellipsis per reply next to the feedback row", () => {
    const onOpenDetails = vi.fn()
    const { container } = render(
      <MessageActionsBar
        {...baseProps()}
        isBot
        canSaveToNotes
        showFeedbackControls
        onOpenDetails={onOpenDetails}
      />
    )

    expect(screen.getByRole("button", { name: "More actions" })).toBeInTheDocument()
    expect(container.querySelectorAll("button svg.lucide-ellipsis")).toHaveLength(1)

    const details = screen.getByRole("button", { name: "More feedback" })
    expect(details.querySelector("svg.lucide-ellipsis")).toBeNull()
    fireEvent.click(details)
    expect(onOpenDetails).toHaveBeenCalledTimes(1)
  })

  it("hides feedback row when showFeedbackControls is false", () => {
    const { container } = render(<MessageActionsBar {...baseProps()} />)

    expect(screen.queryByText("Was this helpful?")).toBeNull()
    expect(container.firstElementChild).toHaveClass("justify-end")
  })
})
