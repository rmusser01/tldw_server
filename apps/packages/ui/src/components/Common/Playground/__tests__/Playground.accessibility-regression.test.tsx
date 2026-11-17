// @vitest-environment jsdom
import React from "react"
import axe from "axe-core"
import { render } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { LoadingStatus } from "../ActionInfo"
import { MessageActionsBar } from "../MessageActionsBar"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string, options?: Record<string, unknown>) => {
      const template = fallback || key
      if (!options) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

const baseActionProps = {
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
  hideCopy: false,
  copyPressed: false,
  onCopy: vi.fn(),
  canReply: false,
  onReply: vi.fn(),
  canSaveToWorkspaceNotes: false,
  onSaveToWorkspaceNotes: vi.fn(),
  canSaveToNotes: false,
  canSaveToFlashcards: false,
  canGenerateDocument: false,
  onGenerateDocument: vi.fn(),
  onSaveKnowledge: vi.fn(),
  savingKnowledge: null,
  generationInfo: null,
  isLastMessage: true,
  hideEditAndRegenerate: false,
  onRegenerate: vi.fn(),
  onNewBranch: vi.fn(),
  temporaryChat: false,
  hideContinue: false,
  onContinue: vi.fn(),
  onRunSteeredContinue: vi.fn(),
  messageSteeringMode: "none",
  onMessageSteeringModeChange: vi.fn(),
  messageSteeringForceNarrate: false,
  onMessageSteeringForceNarrateChange: vi.fn(),
  onClearMessageSteering: vi.fn(),
  onEdit: vi.fn(),
  editMode: false,
  showFeedbackControls: true,
  feedbackSelected: null,
  feedbackDisabled: false,
  feedbackDisabledReason: "",
  isFeedbackSubmitting: false,
  showThanks: false,
  onThumbUp: vi.fn(),
  onThumbDown: vi.fn(),
  onOpenDetails: vi.fn(),
  onDelete: vi.fn(),
  canPin: true,
  isPinned: false,
  onTogglePinned: vi.fn()
} satisfies React.ComponentProps<typeof MessageActionsBar>

async function expectNoCriticalA11yIssues(container: HTMLElement) {
  const results = await axe.run(container, {
    runOnly: {
      type: "rule",
      values: ["aria-allowed-attr", "aria-valid-attr", "aria-roles", "button-name"]
    }
  })

  expect(results.violations).toHaveLength(0)
}

describe("Playground accessibility regression smoke", () => {
  it("keeps loading status live-region semantics free of critical rule violations", async () => {
    const { container } = render(
      <LoadingStatus
        isProcessing
        isStreaming
        isSearchingInternet
        actionInfo="Searching internet..."
      />
    )

    await expectNoCriticalA11yIssues(container)
  })

  it("keeps message action controls accessible for keyboard and SR users", async () => {
    const { container } = render(<MessageActionsBar {...baseActionProps} />)

    await expectNoCriticalA11yIssues(container)
  })
})
