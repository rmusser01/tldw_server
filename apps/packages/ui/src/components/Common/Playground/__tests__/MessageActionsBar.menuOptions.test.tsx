import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
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
      {content}
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
    canSaveToNotes: false,
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

describe("MessageActionsBar menu options", () => {
  it("keeps New Branch available even when temporary chat is enabled", () => {
    render(
      <MessageActionsBar
        {...baseProps()}
        temporaryChat
        onNewBranch={vi.fn()}
      />
    )

    expect(screen.getByText("New Branch")).toBeInTheDocument()
  })

  it("keeps Generate document available when enabled in bot menu", () => {
    render(
      <MessageActionsBar
        {...baseProps()}
        temporaryChat
        canGenerateDocument
      />
    )

    expect(screen.getByText("Generate document")).toBeInTheDocument()
  })

  it("runs steered continue actions immediately when callback is provided", () => {
    const onRunSteeredContinue = vi.fn()
    const onMessageSteeringModeChange = vi.fn()

    render(
      <MessageActionsBar
        {...baseProps()}
        onContinue={vi.fn()}
        onRunSteeredContinue={onRunSteeredContinue}
        onMessageSteeringModeChange={onMessageSteeringModeChange}
        onMessageSteeringForceNarrateChange={vi.fn()}
      />
    )

    fireEvent.click(screen.getByText("Continue as user"))
    fireEvent.click(screen.getByText("Impersonate user"))

    expect(onRunSteeredContinue).toHaveBeenNthCalledWith(
      1,
      "continue_as_user"
    )
    expect(onRunSteeredContinue).toHaveBeenNthCalledWith(
      2,
      "impersonate_user"
    )
    expect(onMessageSteeringModeChange).not.toHaveBeenCalled()
  })

  it("exposes quick transform actions and routes selection callbacks", () => {
    const onQuickMessageAction = vi.fn()
    render(
      <MessageActionsBar
        {...baseProps()}
        onQuickMessageAction={onQuickMessageAction}
      />
    )

    fireEvent.click(screen.getByText("Summarize this"))
    fireEvent.click(screen.getByText("Translate this"))
    fireEvent.click(screen.getByText("Make this shorter"))
    fireEvent.click(screen.getByText("Explain this"))

    expect(onQuickMessageAction).toHaveBeenNthCalledWith(1, "summarize")
    expect(onQuickMessageAction).toHaveBeenNthCalledWith(2, "translate")
    expect(onQuickMessageAction).toHaveBeenNthCalledWith(3, "shorten")
    expect(onQuickMessageAction).toHaveBeenNthCalledWith(4, "explain")
  })

  it("uses touch-sized controls for variant pager and working overflow at every width", () => {
    const onDelete = vi.fn()
    render(
      <MessageActionsBar
        {...baseProps()}
        showVariantPager
        variantCount={3}
        canSwipePrev
        canSwipeNext
        onDelete={onDelete}
      />
    )

    const prevButton = screen.getByTestId("variant-prev-button")
    const nextButton = screen.getByTestId("variant-next-button")
    const countLabel = screen.getByTestId("variant-count-label")
    const overflowTrigger = screen.getByRole("button", { name: "More actions" })

    expect(countLabel).toHaveTextContent("1 of 3")
    expect(prevButton.className).toContain("min-h-[44px]")
    expect(prevButton.className).toContain("min-w-[44px]")
    expect(nextButton.className).toContain("min-h-[44px]")
    expect(nextButton.className).toContain("min-w-[44px]")
    expect(overflowTrigger.className).toContain("min-h-[44px]")
    expect(overflowTrigger.className).toContain("min-w-[44px]")
    fireEvent.click(screen.getByRole("button", { name: "Delete" }))
    expect(onDelete).toHaveBeenCalledTimes(1)
  })
})
