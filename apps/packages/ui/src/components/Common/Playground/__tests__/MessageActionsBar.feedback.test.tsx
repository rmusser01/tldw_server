import React from "react"
import { render, screen } from "@testing-library/react"
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

  it("hides feedback row when showFeedbackControls is false", () => {
    const { container } = render(<MessageActionsBar {...baseProps()} />)

    expect(screen.queryByText("Was this helpful?")).toBeNull()
    expect(container.firstElementChild).toHaveClass("justify-end")
  })
})
